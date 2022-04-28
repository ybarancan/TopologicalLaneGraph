import torch
import torch.nn as nn
import torch.nn.functional as F
import baseline.Utils.utils as utils
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttConvLSTM(nn.Module):
    def __init__(self, opts, feats_channels, feats_dim, n_layers=2,
        hidden_dim=[64,16], kernel_size=3, time_steps=3,
        use_bn=True):
        """
        input_shape: a list -> [b, c, h, w]
        hidden_dim: Number of hidden states in the convLSTM
        """
        super(AttConvLSTM, self).__init__()

        self.opts = opts
        self.feats_channels = feats_channels
        self.grid_size = feats_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.time_steps = time_steps
        self.use_bn = use_bn

        assert(len(self.hidden_dim) == n_layers)

        # Not a fused implementation so that batchnorm
        # can be applied separately before adding

        self.conv_x = []
        self.conv_h = []
        if use_bn:
            self.bn_x = []
            self.bn_h = []
            self.bn_c = []

        for l in range(self.n_layers):
            hidden_dim = self.hidden_dim[l]
            if l != 0:
                in_channels = self.hidden_dim[l-1]
            else:
                in_channels = self.feats_channels + 3
                # +3 for prev 2 vertices and for the 1st vertex

            self.conv_x.append(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = 4*hidden_dim,
                    kernel_size = self.kernel_size,
                    padding = self.kernel_size//2,
                    bias = not use_bn
                )
            )

            self.conv_h.append(
                nn.Conv2d(
                    in_channels = hidden_dim,
                    out_channels = 4*hidden_dim,
                    kernel_size = self.kernel_size,
                    padding = self.kernel_size//2,
                    bias = not use_bn
                )
            )

            if use_bn:
                # Independent BatchNorm per timestep
                self.bn_x.append(nn.ModuleList([nn.BatchNorm2d(4*hidden_dim) for i in range(time_steps)]))
                self.bn_h.append(nn.ModuleList([nn.BatchNorm2d(4*hidden_dim) for i in range(time_steps)]))
                self.bn_c.append(nn.ModuleList([nn.BatchNorm2d(hidden_dim) for i in range(time_steps)]))

        self.conv_x = nn.ModuleList(self.conv_x)
        self.conv_h = nn.ModuleList(self.conv_h)

        if use_bn:
            self.bn_x = nn.ModuleList(self.bn_x)
            self.bn_h = nn.ModuleList(self.bn_h)
            self.bn_c = nn.ModuleList(self.bn_c)

        self.att_in_planes = sum(self.hidden_dim)
        self.conv_att = nn.Conv2d(
            in_channels = self.att_in_planes,
            out_channels = self.feats_channels,
            kernel_size = 1,
            padding = 0,
            bias = True
        )

        self.fc_att = nn.Linear(
            in_features = self.feats_channels,
            out_features = 1
        )
        '''
        self.conv_att_2 = nn.Conv2d(
            in_channels = self.feats_channels,
            out_channels = 1,
            kernel_size = 1,
            padding = 0,
            bias = True
        )
        '''
        self.fc_out = nn.Linear(
            in_features = self.grid_size**2 * self.hidden_dim[-1], 
            out_features = self.grid_size**2 + 1,    
        )

    def rnn_step(self, t, input, cur_state):
        """
        t: time step
        cur_state: [[h(l),c(l)] for l in self.n_layers]
        """
        out_state = []

        for l in range(self.n_layers):
            h_cur, c_cur = cur_state[l]

            if l == 0:
                inp = input
            else:
                inp = out_state[l-1][0]
                # Previous layer hidden
            
            conv_x = self.conv_x[l](inp)
            if self.use_bn:
                conv_x = self.bn_x[l][t](conv_x)
             
            conv_h = self.conv_h[l](h_cur)
            if self.use_bn:
                conv_h = self.bn_h[l][t](conv_h)

            i, f, o, u = torch.split((conv_h + conv_x), self.hidden_dim[l], dim=1) 

            c = F.sigmoid(f) * c_cur + F.sigmoid(i) * F.tanh(u)
            if self.use_bn:
                h = F.sigmoid(o) * F.tanh(self.bn_c[l][t](c))
            else:
                h = F.sigmoid(o) * F.tanh(c)

            out_state.append([h,c])

        return out_state

    def rnn_zero_state(self, shape):
        out_state = []

        for l in range(self.n_layers):
            h = torch.zeros(shape[0], self.hidden_dim[l], shape[2], shape[3], device=device)
            c = torch.zeros(shape[0], self.hidden_dim[l], shape[2], shape[3], device=device)
            out_state.append([h,c])

        return out_state

    def attention(self, feats, rnn_state):
        h_cat = torch.cat([state[0] for state in rnn_state], dim=1)
        conv_h = self.conv_att(h_cat)
        conv_h = F.relu(conv_h + feats)

        conv_h = conv_h.view(-1, self.feats_channels) 
        fc_h = self.fc_att(conv_h) # (N *grid_size*grid_size, 1)

        fc_h = fc_h.view(-1, self.grid_size*self.grid_size) # (N, grid_size*grid_size)

        att = F.softmax(fc_h, dim=-1).view(-1, 1, self.grid_size, self.grid_size) # (N, 1, grid_size, grid_size)

        feats = feats * att

        return feats, att

    def forward(self, feats, first_vertex, poly=None,
        temperature = 0.0, mode='train_ce',
        fp_beam_size=1,
        beam_size=1,
        first_log_prob=None,
        return_attention=False,
        use_correction=False):
        """
        feats: [b, c, h, w]
        first_vertex: [b, ]
        poly: [b, self.time_steps, ]
            : Both first_v and poly are a number between 0 to grid_size**2 + 1,
            : representing a point, or EOS as the last token
        temperature: < 0.01 -> Greedy, else -> multinomial with temperature
        return_attention: True/False
        use_correction: True/False
        mode: 'train_ce'/'train_rl'/'train_eval'/'train_ggnn'/test'/'
        """
        params = locals()
        params.pop('self')

        if beam_size == 1:
            return self.vanilla_forward(**params)
        else:
            return self.beam_forward(**params)



    def vanilla_forward(self, feats, first_vertex, poly=None,
        temperature = 0.0, mode='train_ce',
        fp_beam_size=1,
        beam_size=1,
        first_log_prob=None,
        return_attention=False,
        use_correction=False):
        """
        See forward
        """
        # TODO: The vanilla forward function is pretty messy since
        # it implements all the different modes that we have in
        # a single function. Need to find cleaner alternatives
        
        
        # logging.error('FIRST VERTEX')
        # logging.error(str(first_vertex))
      
        
        
        first_vertex = first_vertex.view(-1)
        batch_size = feats.size(0) * fp_beam_size

        # Expand feats
        if fp_beam_size > 1:
            feats = feats.unsqueeze(1)
            feats = feats.repeat([1, fp_beam_size, 1, 1, 1])
            feats = feats.view(batch_size, -1, self.grid_size, self.grid_size)

        # Setup tensors to be reused
        v_prev2 = torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device)
        v_prev1 = torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device)
        v_first = torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device)

        # Initialize
        v_prev1 = utils.class_to_grid(first_vertex, v_prev1, self.grid_size)
        v_first = utils.class_to_grid(first_vertex, v_first, self.grid_size)
        rnn_state = self.rnn_zero_state(feats.size())

        # Things to output
        if return_attention:
            out_attention = [torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device)]
        pred_polys = [first_vertex.to(torch.float32)]

        if not mode == 'test':
            logits = [torch.zeros(batch_size, self.grid_size**2 + 1, device=device)]

        if first_log_prob is None:
            log_probs = [torch.zeros(batch_size, device=device)]
        else:
            log_probs = [first_log_prob.view(-1)]

        lengths = torch.zeros(batch_size, device=device).to(torch.long)
        lengths += self.time_steps

        # Run AttConvLSTM
        for t in range(1,self.time_steps):
            att_feats, att = self.attention(feats, rnn_state)
            input_t = torch.cat((att_feats, v_prev2, v_prev1, v_first), dim=1)

            rnn_state = self.rnn_step(t, input_t, rnn_state)
            if return_attention:
                out_attention.append(att)

            h_final = rnn_state[-1][0] # h from last layer
            h_final = h_final.view(batch_size, -1)

            logits_t = self.fc_out(h_final)
            logprobs = F.log_softmax(logits_t, dim=-1)

            if temperature < 0.01:
                # No precision issues
                logprob, pred = torch.max(logprobs, dim=-1)
            else:
                probs = torch.exp(logprobs/temperature)
                pred = torch.multinomial(probs, 1)

                # Get logprob of the sampled vertex
                logprob = logprobs.gather(1, pred)

                # Remove the last dimension if not 1
                pred = torch.squeeze(pred, dim=-1)
                logprob = torch.squeeze(logprob, dim=-1)

            for b in range(batch_size):
                if lengths[b] != self.time_steps:
                    continue
                    # prediction has ended
                if pred[b] == self.grid_size**2:
                    # if EOS
                    lengths[b] = t+1
                    # t+1 because we want to keep the EOS
                    # for the loss as well (t goes from 0 to self.time_steps-1)

            if not 'test' in mode:
                logits.append(logits_t)
            
            v_prev2 = v_prev2.copy_(v_prev1)

            if mode == 'train_ce':
                v_prev1 = utils.class_to_grid(poly[:,t], v_prev1, self.grid_size)

            elif mode == 'train_ggnn' and use_correction:
                # GGNN trains on corrected polygons

                pred = self.correct_next_input(pred, poly[:,t])
                v_prev1 = utils.class_to_grid(pred, v_prev1, self.grid_size)

#            elif mode == 'tool':
#                if poly is not None and not expanded:
#                    if poly[0,t] != self.grid_size**2:
#                        pred = (poly[:,t]).repeat(fp_beam_size)
#                    else:
#                        expanded = True
##                        print 'Expanded beam at time: ', t
#                        logprob, pred = torch.topk(logprobs[0,:], fp_beam_size, dim=-1)                        

                v_prev1 = utils.class_to_grid(pred, v_prev1, self.grid_size)

            else:
                # Test mode or train_rl
                v_prev1 = utils.class_to_grid(pred, v_prev1, self.grid_size)

            pred_polys.append(pred.to(torch.float32))
            log_probs.append(logprob)

        out_dict = {}

        pred_polys = torch.stack(pred_polys) # (self.time_steps, b,)
        out_dict['pred_polys'] = pred_polys.permute(1,0)

        out_dict['rnn_state'] = rnn_state
        # Return last rnn state

        log_probs = torch.stack(log_probs).permute(1,0) # (b, self.time_steps)

        # Get logprob_sums
        logprob_sums = torch.zeros(batch_size)
        for b in torch.arange(batch_size, dtype=torch.int32):
            p = torch.sum(log_probs[b, :lengths[b]])
            lp = ((5. + lengths[b])/6.)**0.65
            logprob_sums[b] = p

        out_dict['logprob_sums'] = logprob_sums
        out_dict['feats'] = feats
        # Return the reshape feats based on beam sizes

        if not 'test' in mode:
            out_dict['log_probs'] = log_probs

            logits = torch.stack(logits) # (self.time_steps, b, self.grid_size**2 + 1)
            out_dict['logits'] = logits.permute(1,0,2)

            out_dict['lengths'] = lengths

        if return_attention:
            out_attention = torch.stack(out_attention)
            out_dict['attention'] = out_attention.permute(1,0,2,3,4)

        return out_dict

    def correct_next_input(self, pred, gt):
        x = pred % self.grid_size
        y = pred / self.grid_size

        x_gt = gt % self.grid_size
        y_gt = gt / self.grid_size

        dist = torch.abs(y_gt - y) + torch.abs(x_gt - x)
        need_correct = torch.gt(dist, self.opts['correction_threshold'])

        is_eos = torch.eq(gt, self.grid_size ** 2)
        need_correct = need_correct | is_eos

        out = torch.where(need_correct, gt, pred)
        
        return out

if __name__ == '__main__':
    model = AttConvLSTM(input_shape=[2,128,28,28])
    # print [c for c in model.children()]
    
    feats = torch.ones(2, 128, 28, 28)
    first_vertex = torch.zeros(2,)
    first_vertex[:] = 1
#    print first_vertex

    poly = torch.zeros(8, model.time_steps)
    
    import time
    st = time.time()
    output =  model(feats, first_vertex, poly)
#    print 'Time Taken: ', time.time() - st

#    for k in output.keys():
#        print k, output[k].size()
