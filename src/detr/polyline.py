# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torchvision import ops as torch_ops
from torch import nn
import numpy as np
from src.detr.util import box_ops
from src.detr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from ..nn.resampler import Resampler

from src.detr.deeplab_backbone import build_backbone
from src.detr.matcher import build_matcher, build_polyline_matcher

from baseline.Models.Poly.polyrnnpp import PolyRNNpp
from src import convolutional_rnn

from src.utils import bezier
import logging
from baseline.Utils import utils
from PIL import Image
from scipy.spatial.distance import cdist, directed_hausdorff

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   'traffic_cone', 'barrier']


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 
    
    
class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m
    
class Decoder(nn.Module):
    def __init__(self, indim, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(indim, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        
        self.conv_inter = nn.Conv2d(64, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        
        self.ResMM1 = ResBlock(mdim, mdim)
        
        # self.ResMM2 = ResBlock(mdim, mdim)
        
        # self.Res_side = ResBlock(mdim, mdim)
        
        # self.Res_tot = ResBlock(mdim, mdim)
        # self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(64, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4,  r2):
        m4 = self.ResMM1(self.convFM(r4))
      
        r2 = self.conv_inter(r2)
        
        m2 = self.RF2(r2, m4) # out: 1/4, 256

    
        
        
        p2 = self.pred2(F.relu(m2))
        
        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        # x = self.decoder(x, low_level_feat)
   
        return p2 #, p2, p3, p4
    
    
class Polyline(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone,counter_rnn, transformer,polyrnn, num_classes,args, config,opts):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        
        self.polyrnn = polyrnn
        self.counter_rnn = counter_rnn
        self.transformer = transformer
        self.num_control_points = args.num_spline_points
        self.num_coeffs = self.num_control_points*2
        hidden_dim = 64
        self.hidden_dim = hidden_dim
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        
        self.input_proj = nn.Conv2d(backbone.num_channels + 2, hidden_dim, kernel_size=1)
        
        self.dropout_bev = nn.Dropout(0.1)
        
        self.my_res = ResBlock(backbone.num_channels+2, backbone.num_channels+2)
        
        self.counter_res1 = ResBlock(hidden_dim, hidden_dim)
        # self.dropout1 = nn.Dropout(0.1)
        
        self.counter_res2 = ResBlock(hidden_dim, hidden_dim)
        # self.dropout2 = nn.Dropout(0.1)
        
        self.counter_res3 = ResBlock(hidden_dim, hidden_dim)
        # self.dropout3 = nn.Dropout(0.1)
        
        # self.counter_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,padding = 1)
        # self.counter_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,padding = 1)
        self.counter_conv_dilate = nn.Conv2d(hidden_dim, 8, kernel_size=3,stride=(2,2),padding = 1)
        
        
        self.new_counter_fc = nn.Linear(8*25*25 + 1, 25*25)
        # self.poly_state_conv_dropout = nn.Dropout(0.1)
        
        self.counter_decoder = Decoder(9,hidden_dim)
        
#        self.counter_conv = nn.Conv2d(self.counter_rnn.out_channels, 16, kernel_size=3,padding = 1)
#        self.counter_fc = nn.Linear(16*config.rnn_size[0]*config.rnn_size[1], 2)
#        
#        self.counter_seg_conv1 = nn.Conv2d(self.counter_rnn.out_channels,self.counter_rnn.out_channels,padding = 1, kernel_size=3)
#        self.counter_seg_conv2 = nn.Conv2d(self.counter_rnn.out_channels, 1, padding = 1,kernel_size=3)
#       
        
        self.backbone = backbone
        
        
        self.num_object_queries = args.num_object_queries
        
        self.estimate_objects = args.objects
        self.abs_bev=False
       
        self.poly_state_conv = nn.Conv2d(16, 8, kernel_size=3,stride=(2,2),padding = 1)
        
        self.poly_fc = nn.Linear(8*25*25, hidden_dim)
        
        self.association_embed_maker = MLP(hidden_dim, 128, 64, 2)
        self.fin_embed_maker = MLP(hidden_dim, 128, 64, 2)
        self.start_embed_maker = MLP(hidden_dim, 128, 64, 2)
      
        self.association_classifier = MLP(128, 64, 1, 2)
        self.fin_classifier = MLP(128, 64, 1, 2)
        self.start_classifier = MLP(128, 64, 1, 2)
  
        
    def forward(self, samples,calib,poly_points, left_traffic, training=True,thresh = 0.5, iteration=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        
#        logging.error('CUR POLY ' + str(poly_points.shape))
        
#        if isinstance(samples, (list, torch.Tensor)):
#            samples = nested_tensor_from_tensor_list(samples)
        
        # samples = torch.squeeze(samples,dim=0)
        
#        feed_points = targets['control_points']*50
        
#        logging.error('CALIB SIZE ' + str(calib))
        scale = 16
        calib_downsamp = calib.clone()
        calib_downsamp[:2,:] = calib_downsamp[:2,:] / scale
        # calib_downsamp[:, :2] = calib[:, :2] / scale
        features,low_features, pos, bev_pos = self.backbone(samples,calib, self.abs_bev)
#        logging.error('FEATURES ' + str(features[0].shape))
        # src, mask = features[-1].decompose()
        src = features[-1]
        
        bev_feats = self.transformer(src, torch.unsqueeze(calib_downsamp,0))
        
        
        
        mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1,1,49),torch.linspace(-1,1,50))
        
        my_mesh = torch.unsqueeze(torch.stack([mesh_x,mesh_y],dim=0),0).cuda()
        
        # logging.error('MESH X ' + str(mesh_x.shape))
        
        src = torch.cat([bev_feats,my_mesh],dim=1)
        src = self.my_res(src)
        # assert mask is not None
        bev_feats = self.input_proj(src)
        bev_feats = self.dropout_bev(bev_feats)
        
        
        r1 = self.counter_res1(bev_feats)
        # r1 = self.dropout1(r1)
        
        r2 = self.counter_res2(r1)
        # r1 = self.dropout2(r1)
        
        r3 = self.counter_res3(r2)
        # r1 = self.dropout3(r1)
        
        
        
        # c1 = self.counter_conv1(r3)
        # c2 = self.counter_conv2(c1)
#        logging.error('C2 ' + str(c2.shape))
        c3 = self.counter_conv_dilate(r3)
        
#        logging.error('C3 ' + str(c3.shape))
        
        if left_traffic:
            indi = torch.tensor(np.expand_dims(np.array([1]),axis=0)).cuda().float()
        else:
            indi = torch.tensor(np.expand_dims(np.array([0]),axis=0)).cuda().float()
        
        res_c = self.new_counter_fc(torch.cat([c3.flatten(1),indi],dim=1))
        first_point_loc = res_c.view(res_c.size(0),1,25,25)
        first_point_loc = self.counter_decoder(torch.cat([first_point_loc,c3],dim=1),F.pad(r3,[0,0,1,0]))
        first_point_loc = first_point_loc.squeeze(1)
#        
        out = {         'pred_init_point' : first_point_loc ,'pred_init_point_softmaxed' : first_point_loc.sigmoid()}
        
        
#        first_point_loc = F.interpolate(first_point_loc, scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)
        
        if not training:
            
            np_heatmap = first_point_loc.squeeze(0).sigmoid().detach().cpu().numpy()
            sth_exist = np_heatmap > thresh
            selecteds = np.where(np_heatmap > thresh)
            
            # logging.error('FOUND INIT ' + str(np.sum(sth_exist)))
        #        
        #    else:
        #    selecteds = np.ones((len(poly_locs))) > 0
                
            if np.sum(sth_exist) > 0:
                counter = 1
                while np.sum(sth_exist) > 50:
                # if np.sum(sth_exists) > 50:
                    sth_exist = np_heatmap > (thresh + 0.05*counter)
                    selecteds = np.where(np_heatmap > thresh + 0.05*counter)
                    counter = counter + 1
                    
                init_row, init_col = selecteds
                
                # row_dist = cdist(init_row,init_row)
                # col_dist = cdist(init_col,init_col)
                
                # to_keep = (row_dist > 2) & (col_dist > 2)
                
                
                # row_dist = np.expand_dims(init_row,axis=-1) 
                
                
                init_row = np.int64(init_row/(np_heatmap.shape[0] - 1)*49)
                init_col = np.int64(init_col/(np_heatmap.shape[1] - 1)*49)
                
                to_send = torch.tensor(np.stack([init_col,init_row],axis=-1)).long().cuda().unsqueeze(1)
                
#                logging.error('TO SEND ' + str(to_send.shape))
                
                to_feed = F.pad(bev_feats.expand(to_send.size(0),-1,-1,-1),[0,0,1,0])
                # logging.error('TO FEED  ' + str(to_feed.shape))
                out_dict = self.polyrnn(to_feed, to_send, training=training)
           
            else:
                
                out_dict=dict()
        else:
            cur_poly = poly_points
        
            # logging.error('CUR POLY ' + str(cur_poly))
            to_feed = F.pad(bev_feats.expand(cur_poly.size(0),-1,-1,-1),[0,0,1,0])
            # logging.error('TO FEED  ' + str(to_feed.shape))
            out_dict = self.polyrnn(to_feed, cur_poly, training=training)
   
        
        if 'rnn_state' in out_dict:
   
            states = out_dict['rnn_state'][-1][-1]
            
            # logging.error('STATE LIST ' + str(len(states)))
            # for k in range(len(states)):
            # logging.error('STATES SHAPE ' + str(states.shape))
            
            small_states = self.poly_state_conv(states).view(states.size(0),-1)
            state_vectors = self.poly_fc(small_states)
            '''
            ASSOC
            '''
            
            selected_features = self.association_embed_maker(state_vectors)
            
            reshaped_features1 = torch.unsqueeze(selected_features,dim=1).repeat(1,selected_features.size(0),1)
            reshaped_features2 = torch.unsqueeze(selected_features,dim=0).repeat(selected_features.size(0),1,1)
            
            total_features = torch.cat([reshaped_features1,reshaped_features2],dim=-1)
            
            est = torch.squeeze(self.association_classifier(total_features),dim=-1)
            
            out['pred_assoc'] = torch.unsqueeze(est,dim=0)
            
            out['pred_start'] = None
            
            out['pred_fin'] = None
            
            
            
            
        else:
            out['pred_start'] = None
            
            out['pred_fin'] = None
            out['pred_assoc'] = None

        for k in out_dict.keys():
             out[k] = out_dict[k]

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_object_classes, matcher, weight_dict, eos_coef,object_eos_coef, losses, apply_poly_loss, num_coeffs=3,single_frame=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_object_classes = num_object_classes
        
        self.matcher = matcher

        
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        self.apply_poly_loss = apply_poly_loss
        self.object_eos_coef = object_eos_coef
#        
#        self.cost_obj_center = cost_obj_center
#        self.cost_obj_len = cost_obj_len
#        self.cost_obj_orient = cost_obj_orient
        
        self.losses = losses
        empty_weight = torch.ones(self.num_classes )
        empty_weight[0] = self.eos_coef
        empty_weight_visible = torch.ones(2)
        empty_weight_visible[0] = 0.9
        self.register_buffer('empty_weight', empty_weight)
        
        
 
        
        self.grid_size = 50
        self.dt_threshold = 2
        
        
        
        
        self.register_buffer('empty_weight_visible', empty_weight_visible)

        self.single_frame = single_frame
        self.num_control_points = num_coeffs
        self.bezierA = bezier.bezier_matrix(n_control=self.num_control_points,n_int=50)
        self.bezierA = self.bezierA.cuda()

        self.my_crit = torch.nn.BCEWithLogitsLoss()
 
    
    def loss_assoc(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        
        
        lab = targets[0]['con_matrix']
        lab = lab.float()
        

        est = outputs['pred_assoc']
       
        mask = lab*4 + (1-lab)

        
        loss_ce = torch.mean(F.binary_cross_entropy_with_logits(est.view(-1),lab.view(-1),weight=mask.float().view(-1)))
        losses = {'loss_assoc': loss_ce}
        
      
           
        return losses
     
    def loss_labels(self, outputs, targets, indices,log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        
        # logging.error('LABELS LOSS')
        
        # logging.error('LOGITS ' + str(src_logits.shape))
        
        
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[0,indices[0]] = 1
        
        # logging.error('TARGETS ' + str(target_classes.shape))
        
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        losses = {'loss_ce': loss_ce}

        

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def focal_loss(self,logits, labels, mask, alpha=0.7, gamma=2):
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), 
                                                      reduce=False)
        pt = torch.exp(-bce_loss)
        at = pt.new([alpha, 1 - alpha])[labels.long()]
        focal_loss = at * (1 - pt) ** gamma * bce_loss
    
        return (focal_loss * mask.unsqueeze(1).float()).mean()
        
    def loss_init_points(self,output,target_dict, indices):
        """
        Classification loss for polygon vertex prediction
    
        targets: [batch_size, time_steps, grid_size**2+1]
        Each element is y*grid_size + x, or grid_size**2 for EOS
        mask: [batch_size, time_steps]
        Mask stipulates whether this time step is used for training
        logits: [batch_size, time_steps, grid_size**2 + 1]
        """
        # logging.error('INIT POINTS LOSS')
        logits = output['pred_init_point']
        
        # logging.error('MASK SHAPE ' + str(target_dict[0]['mask'].shape))
        my_mask = target_dict[0]['mask'].unsqueeze(0).unsqueeze(0)
        # logging.error('LOGITS ' + str(logits.shape))
        
        targets = target_dict[0]['init_point_matrix']
        
        # logging.error('TARGETS ' + str(targets.shape))
        
        # Remove the zeroth time step
        # logits = logits.view( -1) # (batch*(time_steps-1), grid_size**2 + 1)
        # targets = targets.view(-1) # (batch*(time_steps-1))
        # targets = targets.sum(0).clamp(0,1).view(-1)
        targets = targets.sum(0).clamp(0,1).unsqueeze(0)
        # F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        # Cross entropy between targets and softmax outputs
#        loss = torch.mean(-targets * F.log_softmax(logits, dim=1), dim=1)
        # my_mask = (targets < 0.3) + 6000*(targets > 0.3)
        
        # loss  = self.my_crit(logits,targets)*my_mask
        
        loss = self.focal_loss(logits,(targets > 0.5).long(),
                               F.interpolate(my_mask[0].float(), size=(100,100), mode='bilinear').squeeze(0).squeeze(0))
       
#        loss = loss.view(batch_size, -1)
#        # Sum across time
#        loss = torch.sum(loss, dim=1)
#        # Mean across batches
        losses = {}
        losses['loss_init_points'] = torch.mean(loss)
            
        return losses
    
    def loss_boxes(self,output,target_dict, indices):
        """
        Classification loss for polygon vertex prediction
    
        targets: [batch_size, time_steps, grid_size**2+1]
        Each element is y*grid_size + x, or grid_size**2 for EOS
        mask: [batch_size, time_steps]
        Mask stipulates whether this time step is used for training
        logits: [batch_size, time_steps, grid_size**2 + 1]
        """
        # logging.error('BOX LOSS')
        
        logits = output['logits']
        
        
        # logging.error('LOGITS ' + str(logits.shape))
        
        dt_targets = utils.dt_targets_from_class(output['poly_class'].cpu().numpy(),
                self.grid_size, self.dt_threshold)
        targets = torch.from_numpy(dt_targets).cuda()
        
        # logging.error('TARGETS ' + str(targets.shape))
        # Remove the zeroth time step
        logits = logits[:, 1:, :].contiguous().view(-1, logits.size(-1)) # (batch*(time_steps-1), grid_size**2 + 1)
        targets = targets[:, 1:, :].contiguous().view(-1, logits.size(-1)) # (batch*(time_steps-1))
        
    
        # Cross entropy between targets and softmax outputs
        loss = torch.mean(-targets * F.log_softmax(logits, dim=1), dim=1)
       
#        loss = loss.view(batch_size, -1)
#        # Sum across time
#        loss = torch.sum(loss, dim=1)
#        # Mean across batches
        
        losses = {}
        losses['loss_bbox'] = torch.mean(loss)
            
        return losses
        
    
        
    
    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
#            'labels': self.loss_labels,
            'loss_init_points': self.loss_init_points,
            'boxes': self.loss_boxes,
            'assoc': self.loss_assoc
#            'loss_polyline': self.loss_polyline,
            
            
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets,indices,   **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
#        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices= self.matcher(outputs, targets)
        
#        outputs = self.get_assoc_estimates(outputs,indices_static)
        
        
        
#        logging.error('MATCHER OUT')
#        logging.error('Len : ' + str(len(indices)))
#        logging.error('Shape : ' + str(indices[0].shape))
#        logging.error(str(indices))
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
#            if 'obj' in loss:
#                
#                losses.update(self.get_loss(loss, outputs, targets, indices_object))
#                
#                
#            else:
            losses.update(self.get_loss(loss, outputs, targets, None))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             if loss == 'masks':
        #                 # Intermediate masks losses are too costly to compute, we ignore them.
        #                 continue
        #             kwargs = {}
        #             if loss == 'labels':
        #                 # Logging is enabled only for the last layer
        #                 kwargs = {'log': False}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes, objects=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_end, out_assoc = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_endpoints'], outputs['pred_assoc']
        # logging.error('OUT END ' + str(out_end.shape))
        # logging.error('OUT LOGITS '+ str(out_logits.shape))
        # logging.error('TARGET SIZES '+ str(target_sizes))
        # logging.error('POST PROCESS ASSOC IN  ' + str(out_assoc))
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        
        
        
        est = torch.reshape(out_bbox,(len(out_bbox),out_bbox.shape[1],-1,2))
        end_est = torch.reshape(out_end,(len(out_end),-1,2,2))
        
#        draw = bezier.interpolate_bezier(est.cpu().numpy())
        

        
                
        results = [{'scores': s, 'labels': l, 'boxes': b,'probs': p,'endpoints': e,'assoc': a} for s, l, b, p, e, a in zip(scores, labels, est,prob,end_est,out_assoc)]
        
#        
#        if objects:
#            
#            logits = outputs['obj_logits']
#            
#            prob = F.softmax(logits, -1).squeeze(0)
#        
#            
#            src_boxes = outputs['obj_boxes'].squeeze(0)
#           
#                    
#            centers = src_boxes[:,:2]
#            angle = src_boxes[:,4]
#            long_len = src_boxes[:,2]
#            short_len = src_boxes[:,3]
#            
#            
#            long_y = torch.abs(torch.sin(angle)*long_len)
#            long_x = torch.cos(angle)*long_len
#            
#            short_x = -torch.sign(torch.cos(angle))*torch.sin(angle)*short_len
#            short_y = torch.abs(torch.cos(angle)*short_len)
#            
#            corner_up = torch.stack([centers[:,0] + long_x/2 + short_x/2, centers[:,1] + long_y/2 + short_y/2],dim=-1)
#            
#            short_corner_up = corner_up - torch.stack([short_x,short_y],dim=-1)
#            
#            long_corner_up = corner_up - torch.stack([long_x,long_y],dim=-1)
#            
#            rest = long_corner_up - torch.stack([short_x,short_y],dim=-1)
#            
#            
#            corners = torch.stack([corner_up, short_corner_up, rest, long_corner_up],dim=1)
#            
#            
#            temp_arx = torch.ones_like(corners[...,0]).cuda()
#            temp_ary = torch.ones_like(corners[...,0]).cuda()
#            
#            temp_arx = 2*temp_arx*corners[...,0]
#            
#            temp_ar = torch.stack([temp_arx, temp_ary],dim=-1)
#                
#            
#            corners = temp_ar - corners
#            
#            
#            
#            
#            # corners = torch.unsqueeze(corners,0)
#            
#            obj_dict={}
##            obj_dict['nms_keep_ind'] = keep_ind
#            
#            obj_dict['corners'] = corners
#            obj_dict['probs'] = prob
#            
#            return (results, obj_dict)
#    
#        else:
        return results
        
#if objects:
#            
#            logits = outputs['obj_logits']
#            
#            prob = F.softmax(logits, -1).squeeze(0)
#        
#            
#            src_boxes = outputs['obj_boxes'].squeeze(0)
#           
#                    
#            centers = src_boxes[:,:2]
#            egim = src_boxes[:,4]
#            long_len = src_boxes[:,2]
#            short_len = src_boxes[:,3]
#            
#            
#            long_y = torch.abs(egim*long_len)
#            long_x = torch.sign(egim)*torch.sqrt(1-torch.square(egim))*long_len
#            
#            short_x = -egim*short_len
#            short_y = torch.sqrt(1-torch.square(egim))*short_len
#            
#            corner_up = torch.stack([centers[:,0] + long_x/2 + short_x/2, centers[:,1] + long_y/2 + short_y/2],dim=-1)
#            
#            short_corner_up = corner_up - torch.stack([short_x,short_y],dim=-1)
#            
#            long_corner_up = corner_up - torch.stack([long_x,long_y],dim=-1)
#            
#            rest = long_corner_up - torch.stack([short_x,short_y],dim=-1)
#            
#            
#            corners = torch.stack([corner_up, short_corner_up, rest, long_corner_up],dim=1)
#            
#            # corners = torch.unsqueeze(corners,0)
#            
#            obj_dict={}
#            obj_dict['corners'] = corners
#            obj_dict['probs'] = prob
#            
#            return (results, obj_dict)
#    
#        else:
#            return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args,config,opts):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = 2
    num_object_classes = args.num_object_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    
    # tfm_resolution = config.map_resolution*2
    # transformer = TransformerPyramid(256, config.tfm_channels, tfm_resolution,
    #                                  config.map_extents, config.ymin, 
    #                                  config.ymax, [config.focal_length*config.patch_size[0]/config.orig_img_size[0],
    #                                                config.focal_length*config.patch_size[1]/config.orig_img_size[1]])
    
    
    in_channels = 64
    resampler = Resampler(4*config.map_resolution, config.map_extents)
    polyrnn =PolyRNNpp(opts,in_channels, config.polyrnn_feat_side)
    
#    counter_rnn = convolutional_rnn.Conv2dLSTM(in_channels=in_channels,  # Corresponds to input size
#                                  out_channels=in_channels,  # Corresponds to hidden size
#                                  kernel_size=(3, 3),  # Int or List[int]
#                                  num_layers=3,
#                                  bidirectional=False)
    
    model = Polyline(
        backbone,
        None,
        resampler,
        polyrnn,
        num_classes=num_classes,
      
        args=args,
        config=config,
        opts=opts
    )
    
    
        
    
    if len(config.gpus) > 1:
        model = nn.DataParallel(model.cuda(), config.gpus)
    elif len(config.gpus) == 1:
        model.cuda()
    
#    
    matcher = build_polyline_matcher(args)
#    
  
    weight_dict = {'loss_ce': args.detection_loss_coef,'loss_bbox': args.bbox_loss_coef,'loss_init_points': args.init_points_loss_coef,
                   'loss_assoc': args.assoc_loss_coef ,
                       'loss_start': args.start_loss_coef ,'loss_fin': args.fin_loss_coef}

  
    losses = [ 'boxes','loss_init_points','assoc']
  
    criterion = SetCriterion(num_classes,num_object_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, object_eos_coef=args.object_eos_coef, losses=losses, 
                             apply_poly_loss = args.apply_poly_loss,
                             num_coeffs=args.num_spline_points)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
   
    return model, criterion, postprocessors
