# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from src.detr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import logging
from src.utils import bezier
import os
import sys
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,cost_visible: float = 1,cost_end: float = 1,
                 cost_poly_hamming=1, cost_poly_exist=1, cost_poly_center=1,  polyline=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        
       
        self.cost_poly_hamming = cost_poly_hamming
        self.cost_poly_exist = cost_poly_exist
        self.cost_poly_center = cost_poly_center
        
        self.cost_end = cost_end
        self.cost_giou = cost_giou
        self.cost_visible = cost_visible
       
        self.polyline = polyline
        self.bezierA = bezier.bezier_matrix(n_control=3,n_int=50)
        self.bezierA = self.bezierA.cuda()
        
    @torch.no_grad()
    def forward(self, outputs, targets,val=False, thresh=0.5, pinet=False,  do_polygons =True, gt_train=False, min_max=False, dist_thresh=0.05):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        

        try:
        
            '''
            FIRST DEAL WITH ROADS
            '''
            if self.polyline:
                
                if do_polygons:
                    
                    if gt_train:
                        num_poly_queries = outputs["poly_hamming"].shape[1]
                        out_poly_ham = outputs["poly_hamming"].flatten(0, 1)
                        out_poly_prob = outputs["poly_prob"].flatten(0, 1).softmax(-1)
                        out_poly_centers = outputs["poly_centers"].flatten(0, 1)
                        
                        tgt_polys = torch.cat([v["poly_one_hots"] for v in targets])
                        tgt_poly_centers = torch.cat([v["poly_centers"] for v in targets])
    
    
                        out_poly_ham = torch.cat([out_poly_ham[:,:(tgt_polys.size(1)-5)],out_poly_ham[:,-5:]],dim=1)
                        
                        
                        to_use_ham = out_poly_ham.sigmoid().unsqueeze(1)
                        
                        
            
                        # poly_ham_loss = torch.sum(-(3*ordered_gt.unsqueeze(0)*torch.log(to_use_ham + 0.0001) +\
                        #                   (1-ordered_gt.unsqueeze(0))*torch.log(1-to_use_ham + 0.0001)), dim=-1)
                        pos_weight = 2
                        poly_ham_loss = torch.sum((pos_weight*tgt_polys.unsqueeze(0)*to_use_ham) +\
                                          (1-tgt_polys.unsqueeze(0))*(1-to_use_ham ), dim=-1)
                        
                        poly_ham_loss = 1 - poly_ham_loss/torch.sum(pos_weight*tgt_polys.unsqueeze(0) + (1-tgt_polys.unsqueeze(0)),dim=-1)
                            
                        # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                            
                        poly_prob_loss = -out_poly_prob[:,-1:]
                        
                        # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                        
                        poly_center_loss = torch.cdist(out_poly_centers, tgt_poly_centers, p=1)
                  
                        # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                        
                        C = self.cost_poly_hamming * poly_ham_loss + self.cost_poly_exist * poly_prob_loss + self.cost_poly_center * poly_center_loss
                        C = C.view(1, num_poly_queries, -1).cpu()
                
                        sizes = [len(v["poly_one_hots"]) for v in targets]
                        poly_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                        poly_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in poly_indices]
                         
                    else:
                        poly_to_return = None
                        
                    return poly_to_return
                
                
                bs, num_queries = outputs["init_point_detection_softmaxed"].shape[:2]
    
            # We flatten to compute the cost matrices in a batch
                out_prob = outputs["init_point_detection_softmaxed"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
          
                est_loc = outputs['pred_init_point_softmaxed'].flatten(1)
            
                # logging.error('LOGITS ' + str(logits.shape))
                
                gt_centers = targets[0]['init_point_matrix']
                if gt_centers.size(0) > 30:
                    gt_centers = gt_centers[:30]
                
                gt_centers = gt_centers.flatten(1)
                gt_center_locs = torch.argmax(gt_centers,dim=-1)
                
                est_dist = est_loc[:,gt_center_locs]
                
                
          
                # Final cost matrix
                # C = -self.cost_bbox * est_dist - self.cost_class * prob_dist
                C = -self.cost_bbox * est_dist 
                C =C.cpu()
        
                static_to_return = linear_sum_assignment(C)
   
                return static_to_return
            
            if pinet:
                tgt_bbox = torch.cat([v["control_points"] for v in targets])
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(outputs, tgt_bbox, p=1)
           
                
           
                # Final cost matrix
                C = cost_bbox
                C = C.view(1, outputs.shape[0], -1).cpu()
        
                sizes = [len(v["control_points"]) for v in targets]
                static_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                static_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in static_indices]
            
                return static_to_return
            bs, num_queries = outputs["pred_logits"].shape[:2]
            
            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["control_points"] for v in targets])
#            tgt_end = torch.cat([v["endpoints"] for v in targets])
        
            if min_max:
                
                problem = False
                tgt_polys = torch.cat([v["poly_one_hots"] for v in targets])
                tgt_poly_centers = torch.cat([v["poly_centers"] for v in targets])
    
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
                out_bbox = outputs["pred_boxes"].flatten(0, 1)  
                
                all_ind = np.arange(len(out_prob))
                selecteds = out_prob[:,1].cpu() > thresh
                
                if torch.sum(selecteds) == 0:
                    problem=True
                    return None, problem
                sel_ind = all_ind[selecteds]
                
                src_boxes = out_bbox.view(out_bbox.size(0), -1, 2)[selecteds]
                tgt_boxes = tgt_bbox.view(tgt_bbox.size(0), -1, 2)
                
                inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
                
                tgt_inter_points = torch.matmul(self.bezierA.expand(tgt_boxes.size(0),-1,-1),tgt_boxes)
                
                
                exp_tgt = tgt_inter_points.unsqueeze(1).unsqueeze(0)
                exp_src = inter_points.unsqueeze(2).unsqueeze(1)
                
                dist_mat = torch.mean(torch.min(torch.sum(torch.abs(exp_tgt - exp_src),dim=-1),dim=-1)[0],dim=-1).cpu().numpy()
                
              
                ind = np.argmin(dist_mat, axis=-1)
                min_vals = np.min(dist_mat,axis=-1)            
    
                fin_sel = sel_ind[min_vals < dist_thresh]
                
                ind = ind[min_vals < dist_thresh]
                
                ara = []
                for k in range(tgt_boxes.size(0)):
                    ara.append(fin_sel[np.where(ind==k)[0]])
    
                ara.append(100)
                ara.append(101)
                ara.append(102)
                ara.append(103)
                ara.append(104)
                
                
                comp_ar = np.zeros((len(tgt_polys),105),np.float32)
                
                
                for k in range(len(tgt_polys)):
                    cu = tgt_polys[k]
                    for m in range(len(cu)):
                        if cu[m]>0:
                            comp_ar[k,ara[m]] = 1
                
                comp_ar = torch.tensor(comp_ar)
                return comp_ar, False
    
            if val:
                cost_class = 5*(out_prob[:, tgt_ids] < thresh)
        
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_bbox2 = torch.cdist(out_bbox, torch.flipud(tgt_bbox), p=1)
           
                cost_bbox = torch.min(torch.cat([cost_bbox,cost_bbox2],dim=0),dim=0)[0] 
           
#                cost_end = torch.cdist(out_endpoints, tgt_end, p=1)
           
                # Final cost matrix
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class 
                C = C.view(bs, num_queries, -1).cpu()
        
                sizes = [len(v["control_points"]) for v in targets]
                static_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                static_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in static_indices]
            
            elif gt_train:
                problem = False
                num_poly_queries = outputs["poly_hamming"].shape[1]
                out_poly_ham = outputs["poly_hamming"].flatten(0, 1)
                out_poly_prob = outputs["poly_prob"].flatten(0, 1).softmax(-1)
                out_poly_centers = outputs["poly_centers"].flatten(0, 1)
                
                to_use_ham = out_poly_ham.sigmoid().unsqueeze(1).cpu()
                
                tgt_polys = torch.cat([v["poly_one_hots"] for v in targets])
                tgt_poly_centers = torch.cat([v["poly_centers"] for v in targets])
    
                
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
                out_bbox = outputs["pred_boxes"].flatten(0, 1)  
                
                all_ind = np.arange(len(out_prob))
                selecteds = out_prob[:,1].cpu() > thresh
                
                if torch.sum(selecteds) == 0:
                    problem=True
                    return None, None, problem
                sel_ind = all_ind[selecteds]
                
                src_boxes = out_bbox.view(out_bbox.size(0), -1, 2)[selecteds]
                tgt_boxes = tgt_bbox.view(tgt_bbox.size(0), -1, 2)
                
                inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
                
                tgt_inter_points = torch.matmul(self.bezierA.expand(tgt_boxes.size(0),-1,-1),tgt_boxes)
                
                
                exp_tgt = tgt_inter_points.unsqueeze(1).unsqueeze(0)
                exp_src = inter_points.unsqueeze(2).unsqueeze(1)
                
                dist_mat = torch.mean(torch.min(torch.sum(torch.abs(exp_tgt - exp_src),dim=-1),dim=-1)[0],dim=-1).cpu().numpy()
                
              
                ind = np.argmin(dist_mat, axis=-1)
                min_vals = np.min(dist_mat,axis=-1)            
    
                fin_sel = sel_ind[min_vals < dist_thresh]
                
                ind = ind[min_vals < dist_thresh]
                
                ara = []
                for k in range(tgt_boxes.size(0)):
                    ara.append(fin_sel[np.where(ind==k)[0]])
    
                ara.append(100)
                ara.append(101)
                ara.append(102)
                ara.append(103)
                ara.append(104)
                
                
                # logging.error('LEN OF ARA  '+ str(len(ara)))
                # logging.error('LEN TGT POLYS  '+ str(len(tgt_polys)))
                # logging.error('LEN CU  '+ str(len(tgt_polys[0])))
    
                comp_ar = np.zeros((len(tgt_polys),105),np.float32)
                
                
                for k in range(len(tgt_polys)):
                    cu = tgt_polys[k]
                    for m in range(len(cu)):
                        if cu[m]>0:
                            comp_ar[k,ara[m]] = 1
                
                comp_ar = torch.tensor(comp_ar).cpu()
                
                pos_weight = 3
                poly_ham_loss = torch.sum((pos_weight*comp_ar.unsqueeze(0)*to_use_ham) +\
                                  (1-comp_ar.unsqueeze(0))*(1-to_use_ham ), dim=-1)
                
                poly_ham_loss = 1 - poly_ham_loss/torch.sum(pos_weight*comp_ar.unsqueeze(0) + (1-comp_ar.unsqueeze(0)),dim=-1)
                    
                # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                    
                poly_prob_loss = -out_poly_prob[:,-1:].cpu()
                
                # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                
                poly_center_loss = torch.cdist(out_poly_centers, tgt_poly_centers, p=1).cpu()
          
                # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                
                C = self.cost_poly_hamming * poly_ham_loss + self.cost_poly_exist * poly_prob_loss + self.cost_poly_center * poly_center_loss
                C = C.view(bs, num_poly_queries, -1).cpu()
        
                sizes = [len(v["poly_one_hots"]) for v in targets]
                poly_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                poly_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in poly_indices]
                 
                return comp_ar, poly_to_return, problem
            

                
            else:
                
                cost_class = -out_prob[:, tgt_ids]
        
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
           
    #            cost_end = torch.cdist(out_endpoints, tgt_end, p=1)
           
                # Final cost matrix
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
                C = C.view(bs, num_queries, -1).cpu()
        
                sizes = [len(v["control_points"]) for v in targets]
                static_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                static_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in static_indices]
                    
                
                # logging.error('STATIC INDICES ')
                # logging.error(str(static_indices))
                if do_polygons:
                    num_poly_queries = outputs["poly_hamming"].shape[1]
                    out_poly_ham = outputs["poly_hamming"].flatten(0, 1)
                    out_poly_prob = outputs["poly_prob"].flatten(0, 1).softmax(-1)
                    out_poly_centers = outputs["poly_centers"].flatten(0, 1)
                
                    tgt_polys = torch.cat([v["poly_one_hots"] for v in targets])
                    tgt_poly_centers = torch.cat([v["poly_centers"] for v in targets])
    
                    src_idx = static_indices[0][0]
                    
                    # logging.error('SRC IDX ')
                    # logging.error(str(src_idx))
                    
                    unused_idx = np.setdiff1d(np.arange(out_prob.shape[0]),src_idx)
                    tgt_idx = static_indices[0][1]
                    
                    # logging.error('TGT IDX ')
                    # logging.error(str(tgt_idx))
                    
                    
                    first_part = out_poly_ham[:,src_idx]
                    # logging.error('FIRST EST ' + str(first_part.shape))
                    second_part = out_poly_ham[:,out_prob.shape[0]:]
                    # logging.error('SECOND EST ' + str(second_part.shape))
                    
                    ordered_est = torch.cat([first_part, second_part ],dim=-1)
        
                    
                    unused_est = out_poly_ham[:,unused_idx]
                    
                    
                    first_gt = tgt_polys[:,tgt_idx]
                    # logging.error('FIRST GT ' + str(first_gt.shape))
                    second_gt = tgt_polys[:,len(tgt_idx):]
                    # second_gt = tgt_polys[:,-5:]
                    # logging.error('SECOND GT ' + str(second_gt.shape))
                    
                    
                    ordered_gt = torch.cat([first_gt, second_gt],dim=-1)
                    
                    to_use_ham = ordered_est.sigmoid().unsqueeze(1)
                    
                    # poly_ham_loss = torch.sum(-(3*ordered_gt.unsqueeze(0)*torch.log(to_use_ham + 0.0001) +\
                    #                   (1-ordered_gt.unsqueeze(0))*torch.log(1-to_use_ham + 0.0001)), dim=-1)
                    pos_weight = 3
                    poly_ham_loss = torch.sum((pos_weight*ordered_gt.unsqueeze(0)*to_use_ham) +\
                                      (1-ordered_gt.unsqueeze(0))*(1-to_use_ham ), dim=-1)
                    
                    poly_ham_loss = 1 - poly_ham_loss/torch.sum(pos_weight*ordered_gt.unsqueeze(0) + (1-ordered_gt.unsqueeze(0)),dim=-1)
                        
                    # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                        
                    poly_prob_loss = -out_poly_prob[:,-1:]
                    
                    # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                    
                    poly_center_loss = torch.cdist(out_poly_centers, tgt_poly_centers, p=1)
              
                    # logging.error('POLY MATCH HAM LOSS ' + str(poly_ham_loss))
                    
                    C = self.cost_poly_hamming * poly_ham_loss + self.cost_poly_exist * poly_prob_loss + self.cost_poly_center * poly_center_loss
                    C = C.view(bs, num_poly_queries, -1).cpu()
            
                    sizes = [len(v["poly_one_hots"]) for v in targets]
                    poly_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                    poly_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in poly_indices]
                     
                    poly_total_return = (src_idx, unused_idx, tgt_idx, ordered_est, unused_est, ordered_gt, poly_to_return)
                
                else:
                    poly_total_return = None
                
            
                
       
            return static_to_return, poly_total_return
    

        except Exception as e:
            logging.error('MATCHER ' + str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))

            
            return (None, None)
        
def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,cost_end=args.set_cost_end, cost_giou=args.set_cost_giou,
                          
                            cost_poly_hamming=args.set_cost_poly_hamming, cost_poly_exist=args.set_cost_poly_exist, cost_poly_center=args.set_cost_poly_center)
    
def build_polyline_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,cost_end=args.set_cost_end, cost_giou=args.set_cost_giou,
                          polyline=True)
