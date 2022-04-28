# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F

from torch import nn

import numpy as np

import math
from ..nn.resampler import Resampler

from src.detr.deeplab_backbone import build_backbone
from src.detr.matcher import  build_polyline_matcher

from baseline.Models.Poly.polyrnnpp import PolyRNNpp

from src.utils import bezier
import logging
from baseline.Utils import utils

from src.detr.transformer import TransformerDecoderLayer, TransformerDecoder
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
  
        self.RF2 = Refine(64, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4,  r2):
        m4 = self.ResMM1(self.convFM(r4))
      
        r2 = self.conv_inter(r2)
        
        m2 = self.RF2(r2, m4) # out: 1/4, 256

    
        
        
        p2 = self.pred2(F.relu(m2))
        
        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
   
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
        
        self.intersection_mode = args.intersection_mode
        
        self.input_proj = nn.Conv2d(backbone.num_channels + 2, hidden_dim, kernel_size=1)
        
        self.dropout_bev = nn.Dropout(0.1)
        
        self.my_res = ResBlock(backbone.num_channels+2, backbone.num_channels+2)
        
        self.counter_res1 = ResBlock(hidden_dim, hidden_dim)
        
        self.counter_res2 = ResBlock(hidden_dim, hidden_dim)
        
        self.counter_res3 = ResBlock(hidden_dim, hidden_dim)

        self.counter_conv_dilate = nn.Conv2d(hidden_dim, 8, kernel_size=3,stride=(2,2),padding = 1)
        
        
        self.new_counter_fc = nn.Linear(8*25*25 + 1, 25*25)
        # self.poly_state_conv_dropout = nn.Dropout(0.1)
        
        self.counter_decoder = Decoder(9,hidden_dim)
  
        self.backbone = backbone
        
        self.abs_bev=False
       
        self.poly_state_conv = nn.Conv2d(16, 8, kernel_size=3,stride=(2,2),padding = 1)
        
        self.poly_fc = nn.Linear(8*25*25, hidden_dim)
        
        '''
        POLYGON STUFF
        '''
        if self.intersection_mode == 'polygon':
            self.polygon_state_conv = nn.Conv2d(16, 8, kernel_size=3,stride=(2,2),padding = 1)
            
            self.polygon_fc = nn.Linear(8*25*25, hidden_dim)
            self.polygon_embed_maker = MLP(hidden_dim, 128, 128, 2)
        '''
        '''
        self.association_embed_maker = MLP(hidden_dim, 128, 64, 2)
        
        self.association_classifier = MLP(128, 64, 1, 2)
        
        self.d_model = 128
        if self.intersection_mode == 'polygon':
            self.polygon_query_embed = nn.Embedding(100, self.d_model)
            
            decoder_layer = TransformerDecoderLayer(self.d_model, 4, 256,
                                                        args.dropout, activation="relu", normalize_before=False)
            decoder_norm = nn.LayerNorm(self.d_model)
          
            self.polygon_trans_decoder = TransformerDecoder(decoder_layer, 3,None, decoder_norm,
                                          return_intermediate=True,temporal_connections=None)
        
            
            self.polygon_ham_embed = MLP(self.d_model, 128, 55 , 2)
            
            self.polygon_exist_embed = MLP(self.d_model, 128, 2 , 2)
        
            self.polygon_center_embed = MLP(self.d_model, 128, 2 , 2)
    
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
        
        
        scale = 16
        calib_downsamp = calib.clone()
        calib_downsamp[:2,:] = calib_downsamp[:2,:] / scale

        features,low_features, pos, bev_pos = self.backbone(samples.clone(),calib.clone(), self.abs_bev)

        src = features[-1]
        
        bev_feats = self.transformer(src, torch.unsqueeze(calib_downsamp,0))
        
        
        
        mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1,1,49),torch.linspace(-1,1,50))
        
        my_mesh = torch.unsqueeze(torch.stack([mesh_x,mesh_y],dim=0),0).cuda()
        
        src = torch.cat([bev_feats,my_mesh],dim=1)
        src = self.my_res(src)
        # assert mask is not None
        bev_feats = self.input_proj(src)
        bev_feats = self.dropout_bev(bev_feats)
        
        
        r1 = self.counter_res1(bev_feats)
        
        r2 = self.counter_res2(r1)
        
        r3 = self.counter_res3(r2)
       
        c3 = self.counter_conv_dilate(r3)
        
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
#            
#  
                
            if np.sum(sth_exist) > 0:
                counter = 1
                while np.sum(sth_exist) > 50:
                # if np.sum(sth_exists) > 50:
                    sth_exist = np_heatmap > (thresh + 0.05*counter)
                    selecteds = np.where(np_heatmap > thresh + 0.05*counter)
                    counter = counter + 1
                    
                    
                init_row, init_col = selecteds
                
                
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
            cur_poly = poly_points.clone()
        
            # logging.error('CUR POLY ' + str(cur_poly))
            to_feed = F.pad(bev_feats.expand(cur_poly.size(0),-1,-1,-1),[0,0,1,0])
            # logging.error('TO FEED  ' + str(to_feed.shape))
            out_dict = self.polyrnn(to_feed, cur_poly, training=training)
   
        
        if 'rnn_state' in out_dict:
   
            states = out_dict['rnn_state'][-1][-1]
            
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
            
            if self.intersection_mode == 'polygon':
                poly1 = self.polygon_state_conv(states).view(states.size(0),-1)
                
                poly2 = self.polygon_fc(poly1)
                trans_in = self.polygon_embed_maker(poly2)
                
                query_embed = self.polygon_query_embed.weight.unsqueeze(1)
                tgt = torch.zeros_like(query_embed)
                
                # logging.error('TRANS IN '+str(trans_in.shape))
                
                if trans_in.size(0) >= 50:
                    padded_trans_in = trans_in[:50]
                    
                else:
                    to_pad = torch.zeros(50-trans_in.size(0), trans_in.size(1)).cuda()
                
                # logging.error('TO PAD '+str(to_pad.shape))
                
                    padded_trans_in = torch.cat([trans_in,to_pad],dim=0)
                
                padded_trans_in = padded_trans_in.unsqueeze(1)
                
                # logging.error('PADDED TRANS IN '+str(padded_trans_in.shape))
                
                mask=torch.zeros(1, trans_in.size(0)).cuda()
                
                mask=mask > 4
                
                pos_embed = torch.zeros_like(padded_trans_in)
                
                not_mask = torch.ones_like(pos_embed)
                not_mask = not_mask[:,:,0]
                y_embed = not_mask.cumsum(0, dtype=torch.float32)
                  
                eps = 1e-6
                y_embed = y_embed / (y_embed[-1:, :] + eps) * 2 * math.pi
             
        
        
                dim_t = torch.arange(128, dtype=torch.float32, device=padded_trans_in.device)
                dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        
               
                pos_embed = y_embed[:, :, None] / dim_t
                
                pos_embed = torch.stack((pos_embed[:, :,  0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3).flatten(2)
                
                
                
                hs = self.polygon_trans_decoder(tgt, padded_trans_in, None, pos=pos_embed, query_pos = query_embed)
                poly_hs = hs.transpose(1, 2)[-1]
                
                poly_hamming = self.polygon_ham_embed(poly_hs)
                
                poly_exist = self.polygon_exist_embed(poly_hs)
            
                poly_centers = self.polygon_center_embed(poly_hs).sigmoid()
            
                
                out['poly_hamming'] = poly_hamming
                out['poly_prob'] = poly_exist
                out['poly_centers'] = poly_centers
            
        else:
            
#            logging.error('NO RNN STATE')
            
            out['pred_start'] = None
            
            out['pred_fin'] = None
            out['pred_assoc'] = None
            
            
            out['poly_hamming'] = None
            out['poly_prob'] = None
            out['poly_centers'] = None
            
#       
        for k in out_dict.keys():
             out[k] = out_dict[k]
#        'merge_features': merge_features
        
                
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
    def __init__(self, num_classes,  matcher, weight_dict, eos_coef, losses, num_coeffs=3, old=False):
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
        self.old=old
        self.matcher = matcher

        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        self.losses = losses
        empty_weight = torch.ones(self.num_classes )
        empty_weight[0] = self.eos_coef
        empty_weight_visible = torch.ones(2)
        empty_weight_visible[0] = 0.9
        self.register_buffer('empty_weight', empty_weight)
        
        
        poly_empty_weight = torch.ones(2)
        poly_empty_weight[0] = 0.2
      
        self.register_buffer('poly_empty_weight', poly_empty_weight)
        
        self.grid_size = 50
        self.dt_threshold = 2
        
        
        self.num_control_points = num_coeffs
        self.bezierA = bezier.bezier_matrix(n_control=self.num_control_points,n_int=100)
        self.bezierA = self.bezierA.cuda()

        self.my_crit = torch.nn.BCEWithLogitsLoss()
        
        grid_x = np.linspace(0,1,50)
        grid_y = np.linspace(0,1,50)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        stacked_mesh = np.stack([mesh_x.flatten(),mesh_y.flatten()],axis=-1)
        self.my_grid = torch.tensor(stacked_mesh).cuda()
        
        
    def get_arranged_gt_order_labels(self, outputs, targets):
        
        gt = targets[0]['gt_order_labels']
        
        n_lines = len(gt) - 5
        
    
        
        ordered_gt = gt[:-5].cpu()
        
        new_labels=np.zeros((n_lines, int(n_lines+5)))
        for k in range(len(ordered_gt)):
            for m in range(len(ordered_gt[k])):
                if ordered_gt[k,m] < 0:
#                    temp = np.ones((ordered_est[0].shape[2]))*105       
                    break
                else:
#                    temp = np.zeros((ordered_est[0].shape[2]))*105            
                    
                    
                        
                    new_labels[k, m] = 1
                        
        self.naive_matched_labels = np.copy(new_labels)
        
        return new_labels
    
    
    def loss_minimality(self, outputs, targets):
        '''
        MINIMALITY
        '''
        tgt_polys = targets[0]['poly_one_hots']
        target_boxes = targets[0]['poly_centers']
        vor_centers = target_boxes
        
     
        
        probs = outputs['logits'].softmax(-1)
        
        con_points = targets[0]['control_points']
        con_points = con_points.view(-1, 3, 2)
        
        
        
        coeffs = torch.sum(probs[:,:,:-1].unsqueeze(-1)*self.my_grid.unsqueeze(0).unsqueeze(0),dim=2).float()
        
        coeffs = torch.cat([con_points[:,0:1,:], coeffs[:,1:,:]],dim=1)
        
        # self.fuzzy_coeffs = coeffs
        
        inter_points = torch.matmul(self.bezierA.expand(coeffs.size(0),-1,-1),coeffs)
        
        n_points = 100
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
        
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
        
        boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
        
        res_ar = torch.cat([inter_points, torch.tensor(boundaries).cuda()],dim=0)
        
        exp_res_ar = res_ar.unsqueeze(0)
        vor_centers = vor_centers.unsqueeze(1).unsqueeze(1)
        dist = torch.min(torch.sum(torch.abs(exp_res_ar - vor_centers),dim=-1),dim=-1)[0]
        
        
        min_dist = []
        for k in range(len(tgt_polys)):
            subset = dist[k,tgt_polys[k] > 0.5,...]
            if len(subset) > 1:
               
                min_dist.append(torch.min(subset, dim=0)[0])
                
            else:

                min_dist.append(torch.tensor(0).cuda())
                
        dist_thresh = torch.stack(min_dist, dim=0)
        # logging.error('STACKED '+ str(stacked.shape))
        

        my_mask = tgt_polys < 0.5
        my_mask = my_mask.cuda()
        # logging.error('MASK ' + str(my_mask.shape))
        
        # logging.error('DIST '+str(dist.shape))
        
        mini_loss = torch.sum(my_mask*torch.max(dist_thresh.unsqueeze(1)-dist,torch.zeros_like(dist)))/(torch.sum(my_mask) + 0.0001) 
        
        losses = {}
        losses['loss_minimality'] = mini_loss
    
        return losses
    
    def naive_intersection_loss(self, outputs, targets, indices):
       
        probs = outputs['logits'].softmax(-1)
        
        con_points = targets[0]['control_points']
        con_points = con_points.view(-1, 3, 2)
        
        
        
        coeffs = torch.sum(probs[:,:,:-1].unsqueeze(-1)*self.my_grid.unsqueeze(0).unsqueeze(0),dim=2).float()
        
        coeffs = torch.cat([con_points[:,0:1,:], coeffs[:,1:,:]],dim=1)
        self.fuzzy_coeffs = coeffs
        
        
        # logging.error('COEFS ' + str(coeffs.shape))
        
        
        inter_points = torch.matmul(self.bezierA.expand(coeffs.size(0),-1,-1),coeffs)
        
        
        n_points = 100
        # coeffs = np.reshape(coeffs,(coeffs.shape[0],-1,2))
        # res_ar = bezier.batch_interpolate(coeffs, n_points) # N x n_points x 2
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
        
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
        
        boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
        
        res_ar = torch.cat([inter_points, torch.tensor(boundaries).cuda()],dim=0)
        
        # logging.error('RES AR '+ str(res_ar.shape))
        
        exp_res_ar = res_ar.unsqueeze(1).unsqueeze(0)
        other_exp_res_ar = res_ar.unsqueeze(2).unsqueeze(1)
        
        dist_mat = torch.sum(torch.abs(exp_res_ar - other_exp_res_ar),dim=-1)
        
        # logging.error('GT DIST MAT '+ str(dist_mat.shape))
        
        dia_mask = torch.tensor(np.tile(np.expand_dims(np.expand_dims(np.eye(len(dist_mat)),axis=-1),axis=-1),[1,1,dist_mat.shape[2],dist_mat.shape[3]])).cuda()
        
        # logging.error('GT DIA MASK '+ str(dia_mask.shape))
        dist_mat = dist_mat * (1-dia_mask) + dia_mask * (dist_mat + 5)
        
        dist_mat = dist_mat[:-5,:]
        
        min_dist = torch.min(dist_mat.view(dist_mat.shape[0],dist_mat.shape[1],-1),dim=-1)[0]
        
        # threshed_dist = min_dist < 0.01
        
        # logging.error('THRESHED DIST ' + str(min_dist.shape))
        
        
        my_labels = self.get_arranged_gt_order_labels(outputs, targets)
        
        my_labels = torch.tensor(my_labels).cuda()
        
        thresh = 0.04
        
        tot_loss1 = torch.sum(min_dist*my_labels)/(torch.sum(my_labels)  + 0.0001)
        
        tot_loss2 = torch.sum((1-my_labels)*torch.max(thresh-min_dist,torch.zeros_like(min_dist)))/(torch.sum(1-my_labels) + 0.0001) 
        
        tot_loss = tot_loss1 + tot_loss2
        
        losses = {}
        losses['loss_naive'] = tot_loss
        return losses
        
    def loss_poly_hamming(self, outputs, targets, indices):
        
        poly_to_return = indices
        ordered_est = outputs['poly_hamming'][0]
        # logging.error('ORDERED EST ' + str(ordered_est.shape))
        
        ordered_gt = targets[0]['poly_one_hots']
        # logging.error('ORDERED GT ' + str(ordered_gt.shape))
  
        my_idx = self._get_src_permutation_idx(poly_to_return)[1]
        my_tgt_idx = self._get_tgt_permutation_idx(poly_to_return)[1]
      
        
        est = ordered_est[my_idx]
        
        
        
        gt = ordered_gt[my_tgt_idx]
        
        cropped_est = torch.cat([est[:,:gt.size(1)-5], est[:,50:]],dim=-1)
        
        to_pad = torch.zeros(len(ordered_gt),ordered_est.size(1) - ordered_gt.size(1)).cuda()
        
        gt = torch.cat([gt[:,:(gt.size(1)-5)], to_pad, gt[:,-5:]],dim=1)
           
        mask = (gt*4 + (1-gt))
        
        loss = F.binary_cross_entropy_with_logits(est,gt, weight=mask)
        losses = {}
        losses['loss_poly_hamming'] = loss.mean()
            
        
        if not self.old:
            '''
            MINIMALITY
            '''
       
            target_boxes = targets[0]['poly_centers']
            vor_centers = target_boxes
            
         
            
            probs = outputs['logits'].softmax(-1)
            
            con_points = targets[0]['control_points']
            con_points = con_points.view(-1, 3, 2)
            
            
            
            coeffs = torch.sum(probs[:,:,:-1].unsqueeze(-1)*self.my_grid.unsqueeze(0).unsqueeze(0),dim=2).float()
            
            coeffs = torch.cat([con_points[:,0:1,:], coeffs[:,1:,:]],dim=1)
            # self.fuzzy_coeffs = coeffs
            
            inter_points = torch.matmul(self.bezierA.expand(coeffs.size(0),-1,-1),coeffs)
            
            n_points = 100
            
            upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
            
            left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
            right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
            
            bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
            
            bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
            
            boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
            
            res_ar = torch.cat([inter_points, torch.tensor(boundaries).float().cuda()],dim=0)
            
            exp_res_ar = res_ar.unsqueeze(0)
            vor_centers = vor_centers.unsqueeze(1).unsqueeze(1)
            dist = torch.min(torch.sum(torch.abs(exp_res_ar - vor_centers),dim=-1),dim=-1)[0]
            
            min_dist = []
            for k in range(len(cropped_est)):
                subset = dist[k,cropped_est[k] > 0.5,...]
                if len(subset) > 1:
                   
                    min_dist.append(torch.min(subset, dim=0)[0])
                    
                else:
    
                    min_dist.append(torch.tensor(0).float().cuda())
                    
            dist_thresh = torch.stack(min_dist, dim=0)
            # logging.error('STACKED '+ str(stacked.shape))
            
    
            my_mask = cropped_est.sigmoid() < 0.5
            
            # logging.error('MASK ' + str(my_mask.shape))
            
            # logging.error('DIST '+str(dist.shape))
            
            mini_loss = torch.sum(my_mask*torch.max(dist_thresh.unsqueeze(1)-dist,torch.zeros_like(dist)))/(torch.sum(my_mask) + 0.0001) 
            
            
            losses['loss_minimality'] = mini_loss
        return losses
        
        
    def loss_poly_exist(self, outputs, targets, indices):
       
        poly_to_return = indices
        
        src_logits = outputs['poly_prob']
        
        idx = self._get_src_permutation_idx(poly_to_return)
        
        target_classes_o = torch.ones(1,len(idx[1]),dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.poly_empty_weight)
        losses = {'loss_poly_exist': loss_ce.mean()}
        return losses
            
    def loss_poly_center(self, outputs, targets, indices):
       
        poly_to_return = indices
        
        idx = self._get_src_permutation_idx(poly_to_return)
    
        src_boxes = outputs['poly_centers'][idx]
        
        target_boxes = torch.cat([t['poly_centers'][i] for t, (_, i) in zip(targets, poly_to_return)], dim=0)

  
    
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_poly_center'] = loss_bbox.mean()
        
        return losses
    
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
        
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[0,indices[0]] = 1
        
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        losses = {'loss_ce': loss_ce}

        return losses
    
    def focal_loss(self,logits, labels, mask, alpha=0.5, gamma=2):
        
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
        
        logits = output['pred_init_point']
        
        my_mask = target_dict[0]['mask'].unsqueeze(0).unsqueeze(0)
        
        targets = target_dict[0]['init_point_matrix']
        
        targets = targets.sum(0).clamp(0,1).unsqueeze(0)
        
        loss = self.focal_loss(logits,(targets > 0.5).long(),
                               F.interpolate(my_mask[0].float(), size=(100,100), mode='bilinear').squeeze(0).squeeze(0))
       

        losses = {}
        losses['loss_init_points'] = torch.mean(loss)
            
        return losses
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def loss_boxes(self,output,target_dict, indices):
        """
        Classification loss for polygon vertex prediction
    
        targets: [batch_size, time_steps, grid_size**2+1]
        Each element is y*grid_size + x, or grid_size**2 for EOS
        mask: [batch_size, time_steps]
        Mask stipulates whether this time step is used for training
        logits: [batch_size, time_steps, grid_size**2 + 1]
        """
        
        logits = output['logits']
        
        
        dt_targets = utils.dt_targets_from_class(output['poly_class'].cpu().numpy(),
                self.grid_size, self.dt_threshold)
        targets = torch.from_numpy(dt_targets).cuda()
        
        # logging.error('TARGETS ' + str(targets.shape))
        # Remove the zeroth time step
        logits = logits[:, 1:, :].contiguous().view(-1, logits.size(-1)) # (batch*(time_steps-1), grid_size**2 + 1)
        targets = targets[:, 1:, :].contiguous().view(-1, logits.size(-1)) # (batch*(time_steps-1))
        
    
        # Cross entropy between targets and softmax outputs
        loss = torch.mean(-targets * F.log_softmax(logits, dim=1), dim=1)

        losses = {}
        losses['loss_bbox'] = torch.mean(loss)
            
        return losses
        
    
        
    
    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {

            'loss_init_points': self.loss_init_points,
            'boxes': self.loss_boxes,
            'assoc': self.loss_assoc,
            'naive_intersection_loss': self.naive_intersection_loss
#            'loss_polyline': self.loss_polyline,
            
     
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

        losses = {}
        for loss in self.losses:
#            :
            losses.update(self.get_loss(loss, outputs, targets, None))


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
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        
        
        
        est = torch.reshape(out_bbox,(len(out_bbox),out_bbox.shape[1],-1,2))
        end_est = torch.reshape(out_end,(len(out_end),-1,2,2))

        results = [{'scores': s, 'labels': l, 'boxes': b,'probs': p,'endpoints': e,'assoc': a} for s, l, b, p, e, a in zip(scores, labels, est,prob,end_est,out_assoc)]
        
#        
        return results
     
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


def build(args,config,opts, old=False):
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

    device = torch.device(args.device)

    backbone = build_backbone(args)
    
    
    in_channels = 64
    resampler = Resampler(4*config.map_resolution, config.map_extents)
    polyrnn =PolyRNNpp(opts,in_channels, config.polyrnn_feat_side)
    
    
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
                      
                       'loss_poly_hamming': args.loss_poly_hamming_coef,
                   'loss_poly_exist': args.loss_poly_exist_coef,
                   'loss_poly_center': args.loss_poly_center_coef,
                   'loss_naive': args.loss_naive_coef,
                   'minimality': args.minimality_coef}

    
    losses = [ 'boxes','loss_init_points','assoc']
  
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, 
                           
                             num_coeffs=args.num_spline_points, old=old)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
   
    return model, criterion, postprocessors
