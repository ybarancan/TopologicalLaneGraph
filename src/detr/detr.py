# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F

from torch import nn
import numpy as np


from src.detr.deeplab_backbone import build_backbone
from src.detr.matcher import build_matcher

from src.detr.transformer import build_transformer

from ..nn.resampler import Resampler
from src.utils import bezier
import logging


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, warpers, num_classes, num_queries,args, aux_loss=False,parameter_setting=None):
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
        self.num_queries = num_queries
        self.transformer = transformer
        
        self.warper1, self.warper2 = warpers
        
        self.intersection_mode = args.intersection_mode
        
        self.num_control_points = args.num_spline_points
        self.num_coeffs = self.num_control_points*2
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.left_query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.split_pe = args.split_pe
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
            
    
        self.bev_pe = args.bev_pe
        self.abs_bev = args.abs_bev
        
        self.only_bev = args.only_bev_pe
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        self.class_embed = MLP(hidden_dim, parameter_setting['class_embed_dim'], num_classes + 1, parameter_setting['class_embed_num'])
        
        self.spline_embed = MLP(hidden_dim, parameter_setting['box_embed_dim'], self.num_coeffs, parameter_setting['box_embed_num'])
    
        self.endpoint_embed = MLP(hidden_dim, parameter_setting['endpoint_embed_dim'], 4, parameter_setting['endpoint_embed_num'])
        self.association_embed_maker = MLP(hidden_dim, parameter_setting['assoc_embed_dim'], parameter_setting['assoc_embed_last_dim'], parameter_setting['assoc_embed_num'])
        self.association_classifier = MLP(2*parameter_setting['assoc_embed_last_dim'], parameter_setting['assoc_classifier_dim'], 1, parameter_setting['assoc_classifier_num'])
        
        if self.intersection_mode == 'polygon':
            
            self.num_poly_queries = args.num_poly_queries
            
            self.poly_query_embed = nn.Embedding(self.num_poly_queries, hidden_dim)
            self.poly_left_query_embed = nn.Embedding(self.num_poly_queries, hidden_dim)
            
            self.poly_ham_embed = MLP(hidden_dim, parameter_setting['poly_embed_dim'], self.num_queries + 5 , parameter_setting['poly_embed_num'])
            
            self.poly_exist_embed = MLP(hidden_dim, parameter_setting['poly_exist_dim'], 2, parameter_setting['poly_exist_num'])
        
            self.poly_center_embed = MLP(hidden_dim, parameter_setting['poly_center_dim'], 2, parameter_setting['poly_center_num'])
    
    def thresh_and_assoc_estimates(self,  outputs,thresh=0.5):

        assoc_features = torch.squeeze(outputs['assoc_features'])
       
        out_logits = torch.squeeze(outputs['pred_logits'])
#        
        prob = F.softmax(out_logits, -1)
        
        selected_features = assoc_features[prob[:,1] > thresh]
        
        reshaped_features1 = torch.unsqueeze(selected_features,dim=1).repeat(1,selected_features.size(0),1)
        reshaped_features2 = torch.unsqueeze(selected_features,dim=0).repeat(selected_features.size(0),1,1)
        
        total_features = torch.cat([reshaped_features1,reshaped_features2],dim=-1)
        
        est = torch.squeeze(self.association_classifier(total_features).sigmoid(),dim=-1)
        
        outputs['pred_assoc'] = torch.unsqueeze(est,dim=0)

        return outputs
    
    
    def forward(self, samples,calib=None,left_traffic=False):
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
#       
        calib_smallest = calib.clone()
        calib_smallest[:2,:] = calib_smallest[:2,:] / 16
        
        calib_big = calib.clone()
        calib_big[:2,:] = calib_big[:2,:] / 4
        
        
        features, low_features, pos, bev_pos = self.backbone(samples,calib_smallest, self.abs_bev)
        
        src = features[-1]
        
        # assert mask is not None
        mask=torch.zeros_like(src)
        mask = mask[:,0,...]
        mask=mask > 4
        
        if left_traffic:

            selected_embed = self.left_query_embed.weight
            
            if self.intersection_mode == 'polygon':
        
                selected_poly_embed = self.poly_left_query_embed.weight
        else:

            selected_embed = self.query_embed.weight
            
            if self.intersection_mode == 'polygon':
        
                selected_poly_embed = self.poly_query_embed.weight
            
            
        if self.intersection_mode == 'polygon':
        
            selected_embed = torch.cat([selected_embed, selected_poly_embed],dim=0)
            
        
        if self.split_pe:
            hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed, torch.cat([pos[-1], bev_pos[-1]],dim=1))
       
        
        elif self.bev_pe:
            if self.only_bev:
                hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed,  bev_pos[-1])
            else:
                hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed, pos[-1] + bev_pos[-1])
        else:
            hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed, pos[-1] )
        

        
        static_hs = hs[:,:,:self.num_queries]
        
        outputs_class = self.class_embed(static_hs)
        outputs_coord = self.spline_embed(static_hs).sigmoid()
        outputs_endpoints = self.endpoint_embed(static_hs).sigmoid()
        
        assoc_features = self.association_embed_maker(static_hs[-1])
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_endpoints': outputs_endpoints[-1], 'assoc_features': assoc_features,
              
               }
        '''
        POLY STUFF
        '''
        if self.intersection_mode == 'polygon':
        
            poly_hs = hs[:,:,-self.num_poly_queries:]
            
            poly_hamming = self.poly_ham_embed(poly_hs)
            
            poly_exist = self.poly_exist_embed(poly_hs)
        
            poly_centers = self.poly_center_embed(poly_hs).sigmoid()
        
            out['poly_hamming'] = poly_hamming[-1]
            out['poly_prob'] = poly_exist[-1]
            out['poly_centers'] = poly_centers[-1]
            
        
      
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
    def __init__(self, num_classes,  matcher, weight_dict, eos_coef, poly_negative_coef,
                 losses,assoc_net,  apply_poly_loss, match_polygons = True, intersection_mode='naive', num_coeffs=3, single_frame=True, old=False):
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
        self.match_polygons = match_polygons
        self.intersection_mode = intersection_mode
        self.matcher = matcher
        
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        self.poly_negative_coef = poly_negative_coef
        
        self.apply_poly_loss = apply_poly_loss

        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        empty_weight_visible = torch.ones(2)
        empty_weight_visible[0] = 0.9
        self.register_buffer('empty_weight', empty_weight)
        
        poly_empty_weight = torch.ones(2)
        poly_empty_weight[0] = self.poly_negative_coef
      
        self.register_buffer('poly_empty_weight', poly_empty_weight)
        
        
        self.naive_matched_labels = None
        
        
        self.register_buffer('empty_weight_visible', empty_weight_visible)

        self.single_frame = single_frame
        self.num_control_points = num_coeffs
        self.bezierA = bezier.bezier_matrix(n_control=self.num_control_points,n_int=100)
        self.bezierA = self.bezierA.cuda()
        self.assoc_net = assoc_net
        
        self.poly_voronoi =None

    def get_arranged_gt_order_labels(self, outputs, targets, indices):
        
        gt = targets[0]['gt_order_labels']
        
        n_lines = len(gt) - 5
        
        idx = self._get_src_permutation_idx(indices)[1]
        tgt_idx = self._get_tgt_permutation_idx(indices)[1]
        
        ordered_gt = gt[tgt_idx].cpu()
        
        new_labels=np.zeros((len(idx), len(idx)+5))
        for k in range(len(ordered_gt)):
            for m in range(len(ordered_gt[k])):
                if ordered_gt[k,m] < 0:
#                    temp = np.ones((ordered_est[0].shape[2]))*105       
                    break
                else:
#                    temp = np.zeros((ordered_est[0].shape[2]))*105            
                    
                    if ordered_gt[k,m] >= n_lines:
                        
                        new_labels[k, len(idx) + ordered_gt[k,m] - n_lines] = 1
                        
                    else:
                        new_labels[k,np.where(tgt_idx==ordered_gt[k,m])[0]] = 1
                        
        self.naive_matched_labels = np.copy(new_labels)
        
        return new_labels
    
    
    def naive_intersection_loss(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx].contiguous().view(-1,self.num_control_points,2)
        
       
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
        
        n_points = 100
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
        
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
        
        boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
        
        res_ar = torch.cat([inter_points, torch.tensor(boundaries).float().cuda()],dim=0)
        
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
        
        
        my_labels = self.get_arranged_gt_order_labels(outputs, targets, indices)
        
        my_labels = torch.tensor(my_labels).cuda()
        
        thresh = 0.04
        
        tot_loss1 = torch.sum(min_dist*my_labels)/(torch.sum(my_labels)  + 0.0001)
        
        tot_loss2 = torch.sum((1-my_labels)*torch.max(thresh-min_dist,torch.zeros_like(min_dist)))/(torch.sum(1-my_labels) + 0.0001) 
        
        tot_loss = tot_loss1 + tot_loss2
        
        losses = {}
        losses['loss_naive'] = tot_loss
        return losses
        
    # def loss_minimal(self, outputs, targets, indices, poly_indices):
        
    @torch.no_grad()
    def voronoi_finder(self, outputs, targets, indices):
    
        poly_to_return = indices
        ordered_est = outputs['poly_hamming'][0]
        # logging.error('ORDERED EST ' + str(ordered_est.shape))
        
        
        my_idx = self._get_src_permutation_idx(poly_to_return)[1]
        my_tgt_idx = self._get_tgt_permutation_idx(poly_to_return)[1]
    
        
        est = ordered_est[my_idx].sigmoid()
        
        grid_x = np.linspace(0,1,100)
        grid_y = np.linspace(0,1,98)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        # logging.error('MESH X '+ str(mesh_x.shape))
        
        stacked_mesh = np.stack([mesh_x.flatten(),mesh_y.flatten()],axis=-1)
        
        orig_mesh = torch.tensor(stacked_mesh).cuda()


        mesh = torch.tensor(stacked_mesh).cuda().unsqueeze(1).unsqueeze(0)

        
        src_boxes = outputs['pred_boxes'].view(-1,self.num_control_points,2)
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
        n_points = 100
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
        
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
        
        boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
        
        res_ar = torch.cat([inter_points, torch.tensor(boundaries).float().cuda()],dim=0)
        
        exp_res_ar = res_ar.unsqueeze(1)
        
        dist = torch.min(torch.sum(torch.abs(exp_res_ar - mesh),dim=-1),dim=-1)[0]
        
        thresh = 0.5
        
        # logging.error('ZERO SHAPE ' + str(torch.tensor(0).cuda()))
        centers = []
        min_dist = []
        for k in range(len(est)):
            subset = dist[est[k] > 0.5]
            if len(subset) > 1:
                temp_mean = torch.mean(subset,dim=0,keepdim=True)
                var = torch.mean(torch.square(temp_mean - subset),dim=0)
                
                # logging.error('SUBSET ' + str(len(subset)) + ' VORONOI VAR ' + str(len(var)))
                
                _, sel = torch.min(var, dim=0)
                # logging.error('NOP '+str(sel))
                centers.append(orig_mesh[sel])
                min_dist.append(torch.min(subset[:,sel], dim=0)[0])
                
                # logging.error('MIN DIST '+str(min_dist[-1]))
                
            else:
                centers.append(torch.zeros(2).cuda())
                min_dist.append(torch.tensor(0).cuda())
                
        stacked = torch.stack(min_dist, dim=0)
        # logging.error('STACKED '+ str(stacked.shape))
        
        return torch.stack(centers,dim=0), stacked
        
    def loss_minimality(self, outputs, targets, indices, comp_ar):
        '''
        MINIMALITY
        '''
        
        target_boxes = targets[0]['poly_centers']
        vor_centers = target_boxes
        
     
        
        src_boxes = outputs['pred_boxes'].view(-1,self.num_control_points,2)
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
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
        for k in range(len(comp_ar)):
            subset = dist[k,comp_ar[k] > 0.5,...]
            if len(subset) > 1:
               
                min_dist.append(torch.min(subset, dim=0)[0])
                
            else:

                min_dist.append(torch.tensor(0).cuda())
                
        dist_thresh = torch.stack(min_dist, dim=0)
        # logging.error('STACKED '+ str(stacked.shape))
        

        my_mask = comp_ar < 0.5
        my_mask = my_mask.cuda()
        # logging.error('MASK ' + str(my_mask.shape))
        
        # logging.error('DIST '+str(dist.shape))
        
        mini_loss = torch.sum(my_mask*torch.max(dist_thresh.unsqueeze(1)-dist,torch.zeros_like(dist)))/(torch.sum(my_mask) + 0.0001) 
        
        losses = {}
        losses['loss_minimality'] = mini_loss
    
        return losses
    
    def loss_poly_hamming(self, outputs, targets, indices):
        
   
        poly_to_return = indices
        ordered_est = outputs['poly_hamming'][0]
        # logging.error('ORDERED EST ' + str(ordered_est.shape))
        
        ordered_gt = targets[0]['modified_gt_poly_one_hots']
        # logging.error('ORDERED GT ' + str(ordered_gt.shape))
        target_boxes = torch.cat([t['poly_centers'][i] for t, (_, i) in zip(targets, poly_to_return)], dim=0)

      
        my_idx = self._get_src_permutation_idx(poly_to_return)[1]
        my_tgt_idx = self._get_tgt_permutation_idx(poly_to_return)[1]

        
        est = ordered_est[my_idx]
        gt = ordered_gt[my_tgt_idx]
        
        mask = (gt*3 + (1-gt))
        
        loss = F.binary_cross_entropy_with_logits(est,gt, weight=mask)
        losses = {}
        losses['loss_poly_hamming'] = loss.mean()
            
        if not self.old:
            '''
            MINIMALITY
            '''
            # vor_centers, dist_thresh = self.voronoi_finder(outputs, targets, indices)
            
            # self.poly_voronoi = vor_centers
            vor_centers = target_boxes
            
            src_boxes = outputs['pred_boxes'].view(-1,self.num_control_points,2)
            
            inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
            
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
            for k in range(len(est)):
                subset = dist[k,est[k] > 0.5,...]
                if len(subset) > 1:
                   
                    min_dist.append(torch.min(subset, dim=0)[0])
                    
                else:
    
                    min_dist.append(torch.tensor(0).float().cuda())
                    
            dist_thresh = torch.stack(min_dist, dim=0)
            # logging.error('STACKED '+ str(stacked.shape))
            
    
            my_mask = est.sigmoid() < 0.5
            
       
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
          
    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
#        logging.error('ROAD IDX0 ' + str(idx[0].shape))
#        logging.error('ROAD IDX ' + str(idx[1].shape))
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
#        logging.error('ROAD TARGET CLASSES O ' + str(target_classes_o.shape))
#        logging.error('RoAD TARGET CLASSES ' + str(target_classes.shape))

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    
    
    def loss_assoc(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        
        _, idx = self._get_src_permutation_idx(indices)
        
        _, target_ids = self._get_tgt_permutation_idx(indices)
        
        lab = targets[0]['con_matrix']
        lab = lab.float()
        lab = lab[target_ids,:]
        lab = lab[:,target_ids]


        est = outputs['pred_assoc']
        
        # mask = torch.eye(idx.size(0)).cuda()
        # mask = 1-mask
        # mask = mask*lab*4 + mask*(1-lab)
        mask = lab*3 + (1-lab)
        
        # logging.error('EST ' + str(est.dtype))
        # logging.error('LAB ' + str(lab.dtype))
        # logging.error('MASK ' + str(mask.dtype))
        
        loss_ce = torch.mean(F.binary_cross_entropy_with_logits(est.view(-1),lab.view(-1),weight=mask.float().view(-1)))
        losses = {'loss_assoc': loss_ce}
   
        

        src_boxes = outputs['pred_boxes'][0][idx]
        src_boxes = src_boxes.view(-1, int(src_boxes.shape[-1]/2), 2)
        
        start_points = src_boxes[:,0,:].contiguous()
        end_points = src_boxes[:,-1,:].contiguous()
        
        my_dist = torch.cdist(end_points, start_points, p=1)
        
        cost_end = 2*my_dist*lab - 3*torch.min(my_dist - 0.05,torch.zeros_like(my_dist).cuda())*(1-lab)
#        losses = {'loss_end_match': cost_end.sum()/(lab.sum() + 0.0001)}
        losses['loss_end_match']= cost_end.mean()

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses



    def loss_polyline(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx].contiguous().view(-1,self.num_control_points,2)
        target_boxes = torch.cat([t['control_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
       
        target_boxes = target_boxes.contiguous().view(-1,self.num_control_points,2)
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
        target_points = torch.matmul(self.bezierA.expand(target_boxes.size(0),-1,-1),target_boxes)
        
        cost_bbox = torch.cdist(inter_points, target_points, p=1)
        
        min0 = torch.mean(torch.min(cost_bbox,dim=1)[0],dim=-1)
        min1 = torch.mean(torch.min(cost_bbox,dim=2)[0],dim=-1)
        
        
        losses = {}
        losses['loss_polyline'] = torch.mean(min0 + min1)

      
        return losses
        
        
    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
        
        if self.single_frame:
            
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['control_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    
            losses = {}
            losses['loss_bbox'] = loss_bbox.mean()
    
          
            return losses
        
    
    
    def loss_endpoints(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
        
        if self.single_frame:
            
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_endpoints'][idx]
            target_boxes = torch.cat([t['endpoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    
            losses = {}
            losses['loss_endpoints'] = loss_bbox.mean()
    
          
            return losses
        

    
    def get_interpolated(self,outputs, targets,indices):
#        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)
        
        target_ids = self._get_tgt_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes'][idx].view(-1,self.num_control_points,2)
        
        src_endpoints = outputs['pred_endpoints'][idx].view(-1,2,2)
#        src_logits = torch.squeeze(outputs['pred_assoc'])

        '''
        ASSOC
        '''        

        
        lab = targets[0]['con_matrix']
        lab = lab.long()
        lab = lab[target_ids[1],:]
        lab = lab[:,target_ids[1]]
        
#        est = src_logits[idx[1]][:,idx[1]]        
        est = outputs['pred_assoc']

        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        my_dict = dict()
        my_dict['interpolated'] = inter_points
        my_dict['endpoints'] = src_endpoints
        my_dict['src_boxes'] = src_boxes
        my_dict['assoc_est'] = torch.squeeze(est,dim=0)

        my_dict['assoc_gt'] = lab

        
        return my_dict, idx, target_ids
        
    
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
    
    
    def get_assoc_estimates(self,  outputs, indices):
        _, idx = self._get_src_permutation_idx(indices)
        
#        _, target_ids = self._get_tgt_permutation_idx(indices)
        
        assoc_features = torch.squeeze(outputs['assoc_features'])
        
        
            
        selected_features = assoc_features[idx]
        
        reshaped_features1 = torch.unsqueeze(selected_features,dim=1).repeat(1,selected_features.size(0),1)
        reshaped_features2 = torch.unsqueeze(selected_features,dim=0).repeat(selected_features.size(0),1,1)
        
        total_features = torch.cat([reshaped_features1,reshaped_features2],dim=-1)
        
        est = torch.squeeze(self.assoc_net(total_features),dim=-1)
        
        outputs['pred_assoc'] = torch.unsqueeze(est,dim=0)
        
  
        
        return outputs
    
    
    
    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'assoc': self.loss_assoc,

            'boxes': self.loss_boxes,
            'loss_polyline': self.loss_polyline,
            
            'endpoints': self.loss_endpoints,
            
            'poly_hamming': self.loss_poly_hamming, 
            'poly_exist': self.loss_poly_exist,
            'poly_center': self.loss_poly_center,
            
            'naive_intersection_loss': self.naive_intersection_loss
          
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices,  **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_static, indices_poly = self.matcher(outputs_without_aux, targets,  do_polygons=self.match_polygons)
        
        outputs = self.get_assoc_estimates(outputs,indices_static)
       
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss[:4] == 'poly' :
                losses.update(self.get_loss(loss, outputs, targets, indices_poly))
                
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices_static))
      
        losses['object_indices'] = (0,0)
        
        losses['static_indices'] = (self._get_src_permutation_idx(indices_static)[1], self._get_tgt_permutation_idx(indices_static)[1])
        
      
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes, objects=True):
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
        
        
#        out_assoc = out_assoc.sigmoid()
        est = torch.reshape(out_bbox,(len(out_bbox),out_bbox.shape[1],-1,2))
        end_est = torch.reshape(out_end,(len(out_end),-1,2,2))

        results = [{'scores': s, 'labels': l, 'boxes': b,'probs': p,'endpoints': e,'assoc': a} for s, l, b, p, e, a in zip(scores, labels, est,prob,end_est,out_assoc)]
        
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


def build(args,config,params, old=False):
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
    resampler1 = Resampler(4*config.map_resolution, config.map_extents)
    
    resampler2 = Resampler(2*config.map_resolution, config.map_extents)
    
    transformer = build_transformer(args)
    
    model = DETR(
        backbone,
        transformer,
        (resampler1, resampler2),
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        parameter_setting=params,
        args=args
    )
    
    
        
    if len(config.gpus) > 1:
        model = nn.DataParallel(model.cuda(), config.gpus)
    elif len(config.gpus) == 1:
        model.cuda()
    
    
    matcher = build_matcher(args)
    

    weight_dict = {'loss_ce': args.detection_loss_coef,'loss_polyline': args.polyline_loss_coef,
                   'loss_endpoints': args.endpoints_loss_coef,'loss_assoc': args.assoc_loss_coef ,
                
                   'loss_end_match': args.loss_end_match_coef,
                   'loss_poly_hamming': args.loss_poly_hamming_coef,
                   'loss_poly_exist': args.loss_poly_exist_coef,
                   'loss_poly_center': args.loss_poly_center_coef,
                   'loss_naive': args.loss_naive_coef,
                   'loss_minimality': args.minimality_coef
                   }

    losses = ['labels', 'boxes','loss_polyline', 'endpoints','assoc']
#    else:
#        if args.intersection_mode == 'naive':
#            losses = ['labels', 'boxes','loss_polyline', 'endpoints','assoc', 'naive_intersection_loss']
#        
#        else:
#            losses = ['labels', 'boxes','loss_polyline', 'endpoints','assoc', 'naive_intersection_loss']
#        # losses = ['poly_hamming', 'poly_exist', 'poly_center']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, poly_negative_coef=args.poly_negative_coef, losses=losses, assoc_net=model.association_classifier,
        apply_poly_loss = args.apply_poly_loss, match_polygons = False, intersection_mode=args.intersection_mode,
                             num_coeffs=args.num_spline_points, old=old)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
   
    return model, criterion, postprocessors
