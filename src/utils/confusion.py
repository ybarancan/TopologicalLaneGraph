import torch
import src.utils.visualise as vis_tools
import logging
import numpy as np
import cv2
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import edit_distance
from src.utils import bezier
import sys
import os
import math
import time
#from mean_average_precision import MeanAveragePrecision
def render_polygon(mask, polygon, shape, value=1):
    
#    logging.error('POLYGON ' + str(polygon.coords))
#    logging.error('EXTENTS ' + str(np.array(extents[:2])))
    to_mult = np.expand_dims(np.array([shape[1],shape[0]]),axis=0)
    polygon = polygon*to_mult
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)


class BinaryConfusionMatrix(object):

    def __init__(self):
        num_class = 1
        num_object_class = 8
        self.num_class = num_class
        
        self.num_object_class = num_object_class
        
        static_metrics_list = ['precision' ,'recall', 'mse', 'miou_lane', 'miou_line',
                               'miou_drivable', 'miou_line_drivable']
        
        self.static_steps = [ 0.01, 0.02, 0.03 , 0.04, 0.05, 0.06, 0.07, 0.08 ,0.09 ,0.1]
        
        self.refine_thresholds = np.arange(0.1,1,0.1)
        
        self.object_dil_sizes = [0,1,2,3]
        self.struct_steps = [1,2,3]
        
        
        self.structs = []
        for k in self.struct_steps:
            self.structs.append(np.ones((int(2*k+1),int(2*k+1))) > 0) 
        
        
        self.matched_gt = 0
        self.unmatched_gt = 0
        
        self.merged_matched_gt = 0
        self.merged_unmatched_gt = 0
        
        self.obj_intersect_limits = [0.1, 0.25, 0.5]
        
        self.object_tp_dict = dict()
        self.object_fp_dict = dict()
        self.object_fn_dict = dict()
        for n in range(len(self.object_dil_sizes)):
            self.object_tp_dict[str(self.object_dil_sizes[n])] = []
            self.object_fp_dict[str(self.object_dil_sizes[n])] = []
            self.object_fn_dict[str(self.object_dil_sizes[n])] = []
            for k in range(len(self.obj_intersect_limits)):
                
            
                self.object_tp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.object_fp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.object_fn_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                
                
                
        self.nms_object_tp_dict = dict()
        self.nms_object_fp_dict = dict()
        self.nms_object_fn_dict = dict()
        for n in range(len(self.object_dil_sizes)):
            self.nms_object_tp_dict[str(self.object_dil_sizes[n])] = []
            self.nms_object_fp_dict[str(self.object_dil_sizes[n])] = []
            self.nms_object_fn_dict[str(self.object_dil_sizes[n])] = []
            for k in range(len(self.obj_intersect_limits)):
                
            
                self.nms_object_tp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.nms_object_fp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.nms_object_fn_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
       
        
        self.seg_object_tp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.seg_object_fp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.seg_object_fn = np.zeros((num_object_class, len(self.object_dil_sizes)))
        
        self.refine_object_tp = np.zeros((num_object_class, len(self.refine_thresholds)))
        self.refine_object_fp = np.zeros((num_object_class, len(self.refine_thresholds)))
        self.refine_object_fn = np.zeros((num_object_class, len(self.refine_thresholds)))
        
        self.argmax_refine_object_tp = np.zeros((num_object_class))
        self.argmax_refine_object_fp = np.zeros((num_object_class))
        self.argmax_refine_object_fn = np.zeros((num_object_class))
        
        self.nms_seg_object_tp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.nms_seg_object_fp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.nms_seg_object_fn = np.zeros((num_object_class, len(self.object_dil_sizes)))
        
        # self.seg_object_tn = np.zeros((num_object_class, len(self.object_dil_sizes)))
        
        self.object_cm = np.zeros((num_object_class+1, num_object_class+1))
        
        self.object_mAP_list = np.zeros((num_object_class))
#        self.object_mAP_list = []
        
#        self.metric_fn = MeanAveragePrecision(num_classes=num_object_class)
        
        self.static_pr_total_est = 0
        self.static_pr_total_gt = 0
        self.static_pr_tp = []
        self.static_pr_fn = []
        self.static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.static_pr_tp.append(0)
            self.static_pr_fn.append(0)
            self.static_pr_fp.append(0)
            
            
        self.merged_static_pr_total_est = 0
        self.merged_static_pr_total_gt = 0
        self.merged_static_pr_tp = []
        self.merged_static_pr_fn = []
        self.merged_static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.merged_static_pr_tp.append(0)
            self.merged_static_pr_fn.append(0)
            self.merged_static_pr_fp.append(0)
        
        self.assoc_tp = 0
        self.assoc_fn = 0
        self.assoc_fp = 0
        
        self.static_lane_tp = 0
        self.static_lane_fn = 0
        self.static_lane_fp = 0
        # self.static_lane_tn = 0
        
        self.static_line_tp = 0
        self.static_line_fn = 0
        self.static_line_fp = 0
        # self.static_tn = 0
        
        self.static_drivable_tp = 0
        self.static_drivable_fn = 0
        self.static_drivable_fp = 0
        
        self.static_line_drivable_tp = 0
        self.static_line_drivable_fn = 0
        self.static_line_drivable_fp = 0
        
        self.static_mse_list=[]
        
        self.order_dist=[]
        
        self.rnn_order_dist=[]
        
        self.poly_tp = 0
        self.poly_fn = 0
        self.poly_fp = 0
        
        self.per_poly_tp = 0
        self.per_poly_fn = 0
        self.per_poly_fp = 0
        
        self.common_poly_tp = 0
        self.common_poly_fn = 0
        self.common_poly_fp = 0
        
        self.common_per_poly_tp = 0
        self.common_per_poly_fn = 0
        self.common_per_poly_fp = 0
        
        
        
        self.static_metrics = dict()
        self.object_metrics = dict()
        
        self.intermediate_metrics = dict()
        
        
        
        self.scene_name = ''
        
        self.to_store_dist=[]
        self.last_sel_ar=None
        self.last_comp_ar=None
        self.last_gt_res_ar=None
        
        self.last_calc_gt_res_ar=None
        
        self.last_est_intersection_list = []
        self.last_gt_intersection_list = []
        
        
        self.rnn_order_dist = []
                    
        self.rnn_to_store_dist = []
        
        self.all_rnn_to_compare_gt = None
        self.all_rnn_to_compare_est = None
        
        self.last_matched_gt_polys = None
        self.last_matched_est_polys = None
        
        self.last_poly_match_gt_indices = None
        self.last_poly_match_est_indices = None
        
        self.cross_per_poly_tp = 0
        self.cross_per_poly_fn = 0
        self.cross_per_poly_fp = 0
        
        '''
        FRAMEWISE
        '''
        self.per_poly_tp_list = []
        self.per_poly_fn_list = []
        self.per_poly_fp_list = []
     
        self.cross_per_poly_tp_list = []
        self.cross_per_poly_fn_list = []
        self.cross_per_poly_fp_list = []
        
        self.common_per_poly_tp_list = []
        self.common_per_poly_fn_list = []
        self.common_per_poly_fp_list = []
        
        
        self.intersection_nums=[]
        self.polygon_nums=[]
        self.common_polygon_nums=[]
        self.head_polygon_nums=[]
        self.line_nums=[]
        
        self.occ_nums=[]
        
        # self.last_

#    @property
#    def num_class(self):
#        return len(self.tp)
    
    def update(self, out, inter_dict, haus_gt, haus_idx, merged_haus_idx, hung_idx, target_ids,targets, poly_stuff=None, do_common_poly=False, pinet=False, polyline=False, dist_thresh=0.05, rnn=False, do_polys=False, assoc_thresh=0.5):
        try:
       
            res_interpolated_list = out['interpolated_points']
            

            poly_ara = []
            for k in range(len(haus_gt)):
                # logging.error('ARA MATCH ' + str(fin_sel[np.where(ind==k)[0]].shape))
                poly_ara.append([])
                
            for m in range(5):
                
                poly_ara.append(np.int64(np.array([len(res_interpolated_list)+m])))
            
            if len(res_interpolated_list) > 0:
                
                
                for k in range(len(res_interpolated_list)):

                    poly_ara[haus_idx[k]].append(k)
            '''
            POLY
            '''
            ara = []
            for k in range(len(haus_gt)):
                # logging.error('ARA MATCH ' + str(fin_sel[np.where(ind==k)[0]].shape))
                ara.append([])
                
            for m in range(5):
                
                ara.append(np.int64(np.array([len(res_interpolated_list)+m])))
            
            if len(res_interpolated_list) > 0:
                
                
                for k in range(len(res_interpolated_list)):

                    ara[haus_idx[k]].append(k)
                    
            # logging.error('ARA ' + str(ara))
            
            non_matched_gts = []
            for k in range(len(haus_gt)):
                if len(ara[k]) == 0:
                    non_matched_gts.append(k)
                    
                    
            poly_non_matched_gts = []
            for k in range(len(haus_gt)):
                if len(poly_ara[k]) == 0:
                    poly_non_matched_gts.append(k)
                    
            # logging.error('NON MATCHED ' + str(non_matched_gts))
            '''
            HAPPENED TO BE POLY
            '''
            
            time1 = time.time()
            
            tgt_polys = targets["poly_one_hots"].cpu().numpy()
                        
            
            self.polygon_nums.append(len(tgt_polys))
            
            if do_common_poly:
                
                poly_centers, poly_one_hots, blob_mat, blob_ids, real_hots = poly_stuff
                
                if np.any(poly_centers == None):
               
                    self.common_per_poly_fn = self.common_per_poly_fn + np.sum(tgt_polys)
                
                else:
                    
                    self.common_polygon_nums.append(len(poly_one_hots))
                    
                    comp_ar = np.zeros((len(tgt_polys),poly_one_hots.shape[-1]),np.float32)
                        
                        
                    for k in range(len(tgt_polys)):
                        cu = tgt_polys[k]
                        for m in range(len(cu)):
                            if cu[m]>0:
                                
                                comp_ar[k,poly_ara[m]] = 1
                    
                    poly_loss_matrix = np.zeros((len(tgt_polys),poly_one_hots.shape[0]))
                    poly_tp_matrix = np.zeros((len(tgt_polys),poly_one_hots.shape[0]))
                    poly_fn_matrix = np.zeros((len(tgt_polys),poly_one_hots.shape[0]))
                    poly_fp_matrix = np.zeros((len(tgt_polys),poly_one_hots.shape[0]))
                    for k in range(len(tgt_polys)):
                        cu = tgt_polys[k]
                        for m in range(len(cu)):
                            if cu[m]>0:
                                
                                if len(poly_ara[m]) > 0:
                                
                                    poly_loss_matrix[k, :] = poly_loss_matrix[k, :] + np.logical_not(np.any(poly_one_hots[:,poly_ara[m]] > 0.5,axis=-1))
                                    
                                    
                                    poly_tp_matrix[k, :] = poly_tp_matrix[k, :] + np.any(poly_one_hots[:,poly_ara[m]] > 0.5,axis=-1)
                                
                                    
                                    poly_fn_matrix[k, :] = poly_fn_matrix[k, :] + np.logical_not(np.any(poly_one_hots[:,poly_ara[m]] > 0.5,axis=-1))
                                
                                
                            else:
                                
                                if len(ara[m]) > 0:
                                
                                    poly_loss_matrix[k, :] = poly_loss_matrix[k, :] + np.any(poly_one_hots[:,poly_ara[m]] > 0.5,axis=-1)
                                    
                                    
                                    poly_fp_matrix[k, :] = poly_fp_matrix[k, :] + np.any(poly_one_hots[:,poly_ara[m]] > 0.5,axis=-1)
                                
                        
                    
                    i,j=linear_sum_assignment(poly_loss_matrix)   
                    
                    matched_gt_polys = comp_ar[i]
                    
                    matched_est_polys = poly_one_hots[j]
                    
                    self.common_last_matched_gt_polys = matched_gt_polys
                    self.common_last_matched_est_polys = matched_est_polys
                    
                    self.common_last_poly_match_gt_indices = i
                    self.common_last_poly_match_est_indices = j
                    
                    matched_gt_polys = matched_gt_polys > 0.5
                    matched_est_polys = matched_est_polys > 0.5
                    
           
                    
                    
                    self.common_per_poly_tp = self.common_per_poly_tp + np.sum(poly_tp_matrix[i,j])
                    self.common_per_poly_fn = self.common_per_poly_fn + np.sum(poly_fn_matrix[i,j])
                    self.common_per_poly_fp = self.common_per_poly_fp + np.sum(poly_fp_matrix[i,j])
    
                    self.common_per_poly_tp_list.append(np.sum(poly_tp_matrix[i,j]))
                    self.common_per_poly_fn_list.append(np.sum(poly_fn_matrix[i,j]))
                    self.common_per_poly_fp_list.append(np.sum(poly_fp_matrix[i,j]))
                    
                    non_matched_penalty = 0
                    
                    for k in range(len(poly_non_matched_gts)):
                        temp_sum = np.sum(tgt_polys[:,poly_non_matched_gts[k]])
                        non_matched_penalty = non_matched_penalty + temp_sum
                        
                    self.common_per_poly_fn = self.common_per_poly_fn + non_matched_penalty
                    self.common_per_poly_fn_list.append(non_matched_penalty)
                    
                    # if len(i) < len(tgt_polys):
                    #     non_detected_tgts = np.setdiff1d(np.arange(len(tgt_polys)),i)
                    #     self.common_per_poly_fn = self.common_per_poly_fn + np.sum(tgt_polys[non_detected_tgts])
                      
                    # if len(j) < len(poly_one_hots):
                    #     non_detected_tgts = np.setdiff1d(np.arange(len(poly_one_hots)),j)
                    #     self.common_per_poly_fp = self.common_per_poly_fp + np.sum(np.logical_or(np.any(poly_fp_matrix[:,non_detected_tgts],axis=0), np.any(poly_tp_matrix[:,non_detected_tgts],axis=0)))
                        
       
                    if do_polys:
                        if np.any(poly_centers == None):
               
                            self.cross_per_poly_fn = self.cross_per_poly_fn + np.sum(poly_one_hots)
                            
                        else:
                            if len(out['boxes']) > 0:
                                if len(out['poly_hamming']) > 0:
                                    unique_hamming = out['poly_hamming'] 
                                    
                                    inv_cross_loss_matrix = np.sum(np.expand_dims(poly_one_hots,axis=1)*np.expand_dims(unique_hamming,axis=0),axis=-1)
                                    
                                    i,j=linear_sum_assignment(poly_one_hots.shape[-1] - inv_cross_loss_matrix)   
                        
                                    cross_gt_polys = poly_one_hots[i]
                                    
                                    cross_est_polys = unique_hamming[j]
                                    
                                    
                                    self.cross_per_poly_tp = self.cross_per_poly_tp + np.sum(cross_gt_polys*cross_est_polys)
                                    self.cross_per_poly_fn = self.cross_per_poly_fn + np.sum(cross_gt_polys*(1-cross_est_polys))
                                    self.cross_per_poly_fp = self.cross_per_poly_fp + np.sum((1-cross_gt_polys)*cross_est_polys)
                          
                                    self.cross_per_poly_tp_list.append(np.sum(cross_gt_polys*cross_est_polys))
                                    self.cross_per_poly_fn_list.append(np.sum(cross_gt_polys*(1-cross_est_polys)))
                                    self.cross_per_poly_fp_list.append(np.sum((1-cross_gt_polys)*cross_est_polys))
                      
                        
            
            time2 = time.time()
#            logging.error('COMMON POLY ' + str(time2 - time1))
            
            if do_polys:   
                
                # logging.error('KEYS')
                # logging.error(str(out.keys()))
                if len(out['boxes']) > 0:
                    detected_poly_probs = out['poly_probs']  
                    detected_poly_centers = out['poly_centers']  
                    unique_hamming = out['poly_hamming'] 
                    self.head_polygon_nums.append(len(unique_hamming))
                    if len(unique_hamming) > 0:
                    
                        comp_ar = np.zeros((len(tgt_polys),unique_hamming.shape[-1]),np.float32)
                        
                        
                        for k in range(len(tgt_polys)):
                            cu = tgt_polys[k]
                            for m in range(len(cu)):
                                if cu[m]>0:
                                    comp_ar[k,poly_ara[m]] = 1
                                    
                        
                        poly_loss_matrix = np.zeros((len(tgt_polys),unique_hamming.shape[0]))
                        poly_tp_matrix = np.zeros((len(tgt_polys),unique_hamming.shape[0]))
                        poly_fn_matrix = np.zeros((len(tgt_polys),unique_hamming.shape[0]))
                        poly_fp_matrix = np.zeros((len(tgt_polys),unique_hamming.shape[0]))
                        for k in range(len(tgt_polys)):
                            cu = tgt_polys[k]
                            for m in range(len(cu)):
                                if cu[m]>0:
                                    
                                    if len(poly_ara[m]) > 0:
                                    
                                        poly_loss_matrix[k, :] = poly_loss_matrix[k, :] + np.logical_not(np.any(unique_hamming[:,poly_ara[m]] > 0.5,axis=-1))
                                        
                                        
                                        poly_tp_matrix[k, :] = poly_tp_matrix[k, :] + np.any(unique_hamming[:,poly_ara[m]] > 0.5,axis=-1)
                                    
                                        
                                        poly_fn_matrix[k, :] = poly_fn_matrix[k, :] + np.logical_not(np.any(unique_hamming[:,poly_ara[m]] > 0.5,axis=-1))
                                    
             
                                else:
                                    
                                    if len(ara[m]) > 0:
                                    
                                        poly_loss_matrix[k, :] = poly_loss_matrix[k, :] + np.any(unique_hamming[:,poly_ara[m]] > 0.5,axis=-1)
                                        
                                        
                                        poly_fp_matrix[k, :] = poly_fp_matrix[k, :] + np.any(unique_hamming[:,poly_ara[m]] > 0.5,axis=-1)
                                    
                            
                        
                        i,j=linear_sum_assignment(poly_loss_matrix)   
                        
                        matched_gt_polys = comp_ar[i]
                        
                        matched_est_polys = unique_hamming[j]
                        
                        self.last_matched_gt_polys = matched_gt_polys
                        self.last_matched_est_polys = matched_est_polys
                        
                        self.last_poly_match_gt_indices = i
                        self.last_poly_match_est_indices = j
                        
                        matched_gt_polys = matched_gt_polys > 0.5
                        matched_est_polys = matched_est_polys > 0.5
                        
               
                        
                        
                        self.per_poly_tp = self.per_poly_tp + np.sum(poly_tp_matrix[i,j])
                        self.per_poly_fn = self.per_poly_fn + np.sum(poly_fn_matrix[i,j])
                        self.per_poly_fp = self.per_poly_fp + np.sum(poly_fp_matrix[i,j])
                                    
                        self.per_poly_tp_list.append(np.sum(poly_tp_matrix[i,j]))
                        self.per_poly_fn_list.append(np.sum(poly_fn_matrix[i,j]))
                        self.per_poly_fp_list.append(np.sum(poly_fp_matrix[i,j]))
                                    
                        non_matched_penalty = 0
                        
                        for k in range(len(poly_non_matched_gts)):
                            temp_sum = np.sum(tgt_polys[:,poly_non_matched_gts[k]])
                            non_matched_penalty = non_matched_penalty + temp_sum
                            
                        self.per_poly_fn = self.per_poly_fn + non_matched_penalty
                        self.per_poly_fn_list.append(non_matched_penalty)
                 
                    else:
             
                        self.per_poly_fn = self.per_poly_fn + np.sum(tgt_polys)
                        self.poly_fn = self.poly_fn + len(tgt_polys)
            
                    
                
                else:
             
                    self.per_poly_fn = self.per_poly_fn + np.sum(tgt_polys)
                    self.poly_fn = self.poly_fn + len(tgt_polys)
               
                
            time3 = time.time()
            logging.error('PER POLY ' + str(time3 - time2))
            '''
            ORDER 
            '''
            
            '''
            FIRST GET ESTIMATED
            '''
            
            
            
            
            # logging.error('RES AR ORDER ' + str(res_ar.shape))
#                non_matched_list = []
            
            final_matched_list=[]
            
            if len(out['boxes']) > 0:
                est_coeffs_ar = np.reshape(out['boxes'],(out['boxes'].shape[0],-1))
                orig_coeffs = targets['control_points'].cpu().numpy()
                if not pinet:
                    res_ar = np.array(out['merged_interpolated_points'])
                    
                    
                    
                    # if len(out['assoc']) > 0:
                    if len(out['assoc']) > 0:
                        assoc_est = out['assoc'] > 0.5
                        
                        sel_candids=[]
                        
                        
                        
                        for m in range(len(haus_gt)):
                            all_matched = ara[m]
                            if len(all_matched) == 0:
                                
#                                non_matched_list.append(m)
                                continue
                            
                            cur_asso = assoc_est[all_matched, all_matched]
                            diag_mask = np.eye(len(cur_asso))
                
                            
                            diag_mask = 1 - diag_mask
                            cur_asso = cur_asso*diag_mask
                            ins, outs = vis_tools.get_vertices(cur_asso)
                            
                            # candid_in = np.zeros((len(all_matched)))
                            # candid_out = np.zeros((len(all_matched)))
                            
                            all_candids=[]
                            for k in range(len(ins)):
                                
                                if k == 0:
                                    # temp_tuple = (np.array(ins[k]), np.array(outs[k]), [ins[k] + outs[k]])
                                    temp_tuple = (ins[k], outs[k], [])
                                    
                                    all_candids.append(temp_tuple)
                                
                                else:
                                    for t in range(len(all_candids)):
                                        to_comp_in, to_comp_out, all_lin = all_candids[t]
                                        
                                        for m in range(len(ins[k])):
                                            
                                            if ins[k][m] in to_comp_out:
                                                new_all_lin = []
                                                for qwe in all_lin:
                                                    new_all_lin.append(qwe)
                                                new_all_lin.append(ins[k][m])
                                                new_to_comp_in = to_comp_in
                                                new_to_comp_out = outs[k]
                                                
                                                all_candids.append((new_to_comp_in, new_to_comp_out, new_all_lin))
                                        
                                        for m in range(len(outs[k])):
                                            
                                            if outs[k][m] in to_comp_in:
                                                new_all_lin = []
                                                for qwe in all_lin:
                                                    new_all_lin.append(qwe)
                                                new_all_lin.append(outs[k][m])
                                              
                                                
                                                
                                                new_to_comp_out = to_comp_out
                                                new_to_comp_in = ins[k]
                                                
                                                all_candids.append((new_to_comp_in, new_to_comp_out, new_all_lin))
                            
                            if len(all_candids) == 0:
                                # logging.error('ENTERED ALL CANDIDS 0')
                                
                                temp_res = res_ar[all_matched]    
                                # logging.error('TEMP RES '+str(temp_res.shape))
                                mind = np.argmin(np.sum(np.abs(est_coeffs_ar[all_matched] - orig_coeffs[m:m+1]),axis=-1))
                                final_matched_list.append([all_matched[mind]])
                                res = temp_res[mind]
         
                                # logging.error('0 ALL CANDID RES '+str(res.shape))
                            else:
                                cand_lens=[]                
                                for t in range(len(all_candids)):
                                    cand_lens.append(len(all_candids[t][2]))
                                    
                                longest = np.argmax(np.array(cand_lens))
                                    
                                sel_curves = all_candids[longest][2]
                                
                                sel_curves.append(all_candids[longest][0][0])
                                sel_curves.append(all_candids[longest][1][0])
                                    
                                sel_coefs = res_ar[sel_curves]
                                
                                start_points = sel_coefs[:,0]
                                end_points = sel_coefs[:,-1]
                                
                                exp_start = np.expand_dims(start_points,axis=1)
                                exp_end = np.expand_dims(end_points,axis=0)
                                
                                start_end = np.sum(np.abs(exp_start - exp_end),axis=-1)
                                
                                
                                ordered_set = []
                                min_start = np.min(start_end,axis=-1)
                                max_start_id = np.argmax(min_start)
                                cur_ind = max_start_id
                                ordered_set.append(max_start_id)
                                for line in range(len(start_end)-1):
                                    next_one = np.argmin(start_end[:,ordered_set[-1]])
                                    if next_one not in ordered_set:
                                        ordered_set.append(next_one)
                                    else:
                                        break
                                    
                                
#                                    logging.error('ORDERED SET ' + str(ordered_set))
                                picked_merged = []
                                for k in range(len(ordered_set)):
                                    
                                    picked_merged.append(out['merged_interpolated_points'][ordered_set[k]])
                                
                                final_matched_list.append(ordered_set)
                                
                                
                                concatted = np.concatenate(picked_merged, axis=0)
                                
#                               
                                fitted_coeffs = bezier.fit_bezier(concatted, 3)[0]
                                
                                                              
                                res = bezier.interpolate_bezier(fitted_coeffs, n_int=res_ar.shape[1])
                                
                            sel_candids.append(np.copy(res))
                            
                        sel_ar = np.array(sel_candids)
                        
                    else:
                        res_ar = np.array(out['interpolated_points'])
                        sel_candids=[]
                        for m in range(len(haus_gt)):
                            
                            all_matched = ara[m]
                            if len(all_matched) == 0:
                                
                                # logging.error('NOTHING MATCHED')
                                
#                                res= -10*np.ones_like(haus_gt[0])   
#                                sel_candids.append(np.copy(res))
#                                non_matched_list.append(m)
                                continue
                            temp_res = res_ar[all_matched]    
                                # logging.error('TEMP RES '+str(temp_res.shape))
                            mind = np.argmin(np.sum(np.abs(est_coeffs_ar[all_matched] - orig_coeffs[m:m+1]),axis=-1))
                            final_matched_list.append([all_matched[mind]])
                            res = temp_res[mind]
     
                            sel_candids.append(np.copy(res))
                        sel_ar = np.array(sel_candids)
                else:
                    res_ar = np.array(out['interpolated_points'])
                    sel_candids=[]
                    for m in range(len(haus_gt)):
                        
                        all_matched = ara[m]
                        if len(all_matched) == 0:
                            
                            # logging.error('NOTHING MATCHED')
                            
#                                res= -10*np.ones_like(haus_gt[0])   
#                                sel_candids.append(np.copy(res))
#                            non_matched_list.append(m)
                            continue
                        temp_res = res_ar[all_matched]    
                            # logging.error('TEMP RES '+str(temp_res.shape))
                        mind = np.argmin(np.sum(np.abs(est_coeffs_ar[all_matched] - orig_coeffs[m:m+1]),axis=-1))
                        final_matched_list.append([all_matched[mind]])
                        res = temp_res[mind]
 
                        sel_candids.append(np.copy(res))
                    sel_ar = np.array(sel_candids)
                    
                    
                    
                # logging.error('SEL AR ' + str(sel_ar.shape))
                
                if len(sel_ar) > 0:
                
                    n_points = res_ar.shape[1]
                    upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
                    
                    left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
                    right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
                    
                    bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
                    
                    bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
                    
                    boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
                    
                    sel_ar = np.concatenate([sel_ar, boundaries],axis=0)
                    
                    self.last_sel_ar = np.copy(sel_ar)
                    
                    # logging.error('FINAL MATCHED LIST ' + str(final_matched_list))
                    
                    fin_list = []
                    for k in range(len(final_matched_list)):
                        for m in range(len(final_matched_list[k])):
                            fin_list.append(final_matched_list[k][m])
                    
                    # logging.error('FIN LIST ' + str(fin_list))
                    
                    non_used = np.setdiff1d(np.arange(len(res_ar)), np.array(fin_list))
                    
                    base_sel_ar = np.copy(sel_ar)
                    
                    exp_res_ar = np.expand_dims(np.expand_dims(sel_ar, axis=1), axis=0)
                    other_exp_res_ar = np.expand_dims(np.expand_dims(sel_ar, axis=2), axis=1)
                    
                    dist_mat = np.sum(np.abs(exp_res_ar - other_exp_res_ar),axis=-1)
                
                    
                    dia_mask = np.tile(np.expand_dims(np.expand_dims(np.eye(len(dist_mat)),axis=-1),axis=-1),[1,1,dist_mat.shape[2],dist_mat.shape[3]])
                    dist_mat = dist_mat * (1-dia_mask) + dia_mask * (dist_mat + 5)
                    
                    min_dist = np.min(dist_mat,axis=-1)
                    
                    threshed_dist = min_dist < 0.01
                    
            
                    
                    est_intersection_list = []
                    
                    for k in range(len(threshed_dist)):
                        cur_list = []
                        inter_so_far = []
                        # logging.error('EST REF CURVE ' + str(k))
                        for m in range(threshed_dist.shape[-1]):
                            if np.any(threshed_dist[k,:,m]):
                                # logging.error('GT REF CURVE ' + str(m))
                                temp = np.where(threshed_dist[k,:,m])[0]
                                
                                temp_temp=[]
                                for n in range(len(temp)):
                                    
                                    # if temp[n] >= len(base_sel_ar):
                                    #     cur_to = temp[n] + 200
                                    # else:
                                    cur_to = temp[n]
                                    if not (cur_to in inter_so_far):
                                        temp_temp.append(cur_to)
                                        inter_so_far.append(cur_to)
                                if len(temp_temp)>0:
                                    cur_list.append(temp_temp)
                           
                        est_intersection_list.append(cur_list)
                        
                    self.last_est_intersection_list = est_intersection_list
                
                else:
                    est_intersection_list = []
                    
                    for k in range(len(haus_gt)+5):
                        
                        est_intersection_list.append([])
                        
                    self.last_est_intersection_list = est_intersection_list
                    
                    
                # logging.error('ORDER EST INTER SIZE ' + str(len(est_intersection_list)))
                
                '''
                GT INTERSECTIONS
                '''
                
                # n_points = res_ar.shape[1]
                
                
                    
                res_ar = np.array(haus_gt)
                n_points = res_ar.shape[1]
                upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
                
                left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
                right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
                
                bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
                
                bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
                
                boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
                
                
                res_ar = np.concatenate([res_ar, boundaries],axis=0)
                self.last_gt_res_ar = np.copy(res_ar)
                
                # logging.error('GT RES AR '+ str(self.last_gt_res_ar.shape))
                
                exp_res_ar = np.expand_dims(np.expand_dims(res_ar, axis=1), axis=0)
                other_exp_res_ar = np.expand_dims(np.expand_dims(res_ar, axis=2), axis=1)
                
                dist_mat = np.sum(np.abs(exp_res_ar - other_exp_res_ar),axis=-1)
                
                # logging.error('GT DIST MAT '+ str(dist_mat.shape))
                
                dia_mask = np.tile(np.expand_dims(np.expand_dims(np.eye(len(dist_mat)),axis=-1),axis=-1),[1,1,dist_mat.shape[2],dist_mat.shape[3]])
                
                # logging.error('GT DIA MASK '+ str(dia_mask.shape))
                dist_mat = dist_mat * (1-dia_mask) + dia_mask * (dist_mat + 5)
                
                min_dist = np.min(dist_mat,axis=-1)
                
                threshed_dist = min_dist < 0.01
                
                # trans_thresh_dist = np.transpose(threshed_dist,(0,2,1))
                
                # logging.error('TRANS GT THRESH DIST ' + str(trans_thresh_dist.shape))
                
                # wher_res = np.where(trans_thresh_dist)
                total_intersection_count = 0
                gt_intersection_list = []
                
                for k in range(len(threshed_dist)):
                    cur_list = []
                    inter_so_far = []
                    # logging.error('GT REF CURVE ' + str(k))
                    for m in range(threshed_dist.shape[-1]):
                        if np.any(threshed_dist[k,:,m]):
                            # logging.error('GT CANDIDATE POINT ' + str(m))
                            temp = np.where(threshed_dist[k,:,m])[0]
                            
                            # logging.error('TEMP ' + str(temp))
                            
                            temp_temp=[]
                            for n in range(len(temp)):
                                if not (temp[n] in inter_so_far):
                                    temp_temp.append(temp[n])
                                    inter_so_far.append(temp[n])
                                    
                                    total_intersection_count = total_intersection_count+1
                            if len(temp_temp)>0:
                                cur_list.append(temp_temp)
                            # logging.error('CUR LIST ' + str(cur_list))
                    gt_intersection_list.append(cur_list)
                            
                    
                    # gt_intersection_list.append(wher_res[2][wher_res[0] == k])
                    
                    # logging.error('GT INTERSEC ' + str(gt_intersection_list[-1]))
                self.last_gt_intersection_list = gt_intersection_list
                
                
                
                
                '''
                GT WITH ONLY MATCHED ONES
                '''
                indeed_matched_list = np.setdiff1d(np.arange(len(gt_intersection_list)-5),non_matched_gts)
                # logging.error('INDEED MATCHED ' + str(indeed_matched_list))
                
                self.indeed_matched_list = indeed_matched_list
                
                res_ar = np.array(haus_gt)[indeed_matched_list]
                n_points = res_ar.shape[1]
                upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
                
                left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
                right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
                
                bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
                
                bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
                
                boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
                
                
                res_ar = np.concatenate([res_ar, boundaries],axis=0)
                self.last_calc_gt_res_ar = np.copy(res_ar)
                
                # logging.error('GT RES AR '+ str(self.last_gt_res_ar.shape))
                
                exp_res_ar = np.expand_dims(np.expand_dims(res_ar, axis=1), axis=0)
                other_exp_res_ar = np.expand_dims(np.expand_dims(res_ar, axis=2), axis=1)
                
                dist_mat = np.sum(np.abs(exp_res_ar - other_exp_res_ar),axis=-1)
                
                # logging.error('GT DIST MAT '+ str(dist_mat.shape))
                
                dia_mask = np.tile(np.expand_dims(np.expand_dims(np.eye(len(dist_mat)),axis=-1),axis=-1),[1,1,dist_mat.shape[2],dist_mat.shape[3]])
                
                # logging.error('GT DIA MASK '+ str(dia_mask.shape))
                dist_mat = dist_mat * (1-dia_mask) + dia_mask * (dist_mat + 5)
                
                min_dist = np.min(dist_mat,axis=-1)
                
                threshed_dist = min_dist < 0.01
                
                # trans_thresh_dist = np.transpose(threshed_dist,(0,2,1))
                
                # logging.error('TRANS GT THRESH DIST ' + str(trans_thresh_dist.shape))
                
                # wher_res = np.where(trans_thresh_dist)
                
                calc_gt_intersection_list = []
                
                for k in range(len(threshed_dist)):
                    cur_list = []
                    inter_so_far = []
                    # logging.error('GT REF CURVE ' + str(k))
                    for m in range(threshed_dist.shape[-1]):
                        if np.any(threshed_dist[k,:,m]):
                            # logging.error('GT CANDIDATE POINT ' + str(m))
                            temp = np.where(threshed_dist[k,:,m])[0]
                            
                            # logging.error('TEMP ' + str(temp))
                            
                            temp_temp=[]
                            for n in range(len(temp)):
                                if not (temp[n] in inter_so_far):
                                    temp_temp.append(temp[n])
                                    inter_so_far.append(temp[n])
                            if len(temp_temp)>0:
                                cur_list.append(temp_temp)
                            # logging.error('CUR LIST ' + str(cur_list))
                    calc_gt_intersection_list.append(cur_list)
                            
                    
                    # gt_intersection_list.append(wher_res[2][wher_res[0] == k])
                    
                    # logging.error('GT INTERSEC ' + str(gt_intersection_list[-1]))
                self.last_calc_gt_intersection_list = calc_gt_intersection_list
                
                
                    # logging.error('ORDER GT INTER SIZE ' + str(len(gt_intersection_list[])))
            '''
            CALC PRE_RECALL
            '''
            if len(out['interpolated_points']) > 0:
                
                
                to_store_dist=[]
                edit_distances = []
                all_compare_gts = []
                # indeed_matched_list = np.setdiff1d(np.arange(len(gt_intersection_list)-5),non_matched_gts)
                
                
                
                for k in range(len(calc_gt_intersection_list)):
                    
                    # if k in non_matched_gts:
                    #     continue
                    
                    if len(calc_gt_intersection_list[k]) == 0:
                        to_store_dist.append(0)
                        edit_distances.append(0)
                    
                    else:
                        all_specials = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
                        cur_special_count = 0
                        cur_specials_gt = []
                        cur_specials_gt_pos = []
                        to_compare_gt = []
                        to_compare_est = []
                        # logging.error('ORIG GT ' + str(gt_intersection_list[k]))
                        for m in range(len(calc_gt_intersection_list[k])):
                            if len(calc_gt_intersection_list[k][m]) > 1:
                                for n in range(len(calc_gt_intersection_list[k][m])):
                                    
                                    cur_specials_gt.append(calc_gt_intersection_list[k][m][n])
                                
                                    cur_specials_gt_pos.append(cur_special_count)
                                
                                to_compare_gt.append(all_specials[cur_special_count])
                                cur_special_count = cur_special_count + 1
                            
                            elif len(calc_gt_intersection_list[k][m]) > 0:
                                to_compare_gt.append(calc_gt_intersection_list[k][m][0])
                            
                            else:
                                
                                continue
                            
                        all_compare_gts.append(to_compare_gt)
                        # logging.error('TO COMPARE GT ' + str(to_compare_gt))
                        
                        # logging.error('ORIG EST ' + str(est_intersection_list[k]))
                        # logging.error('CUR SPECIALS GT ' + str(cur_specials_gt))
                        # logging.error('CUR SPECIALS GT POS ' + str(cur_specials_gt_pos))
                        for m in range(len(est_intersection_list[k])):
                            for n in range(len(est_intersection_list[k][m])):
                                # any_special=False
                                if est_intersection_list[k][m][n] in cur_specials_gt:
                                    
                                    sele_ind = np.array(cur_specials_gt) == est_intersection_list[k][m][n]
                                    
                                    # logging.error('SELE IND ' + str(sele_ind))
                                    
                                    to_add = int(np.array(cur_specials_gt_pos)[sele_ind][0])
                                    
                                    
                                    # logging.error('TO ADD BEFORE ' + str(to_add))
                                    
                                    # if len(to_add) > 1:
                                    #     to_add = to_add[0]
                                    
                                    # logging.error('TO ADD AFTER ONE ' + str(to_add))
                                    # to_add = int(to_add[0])
                                    # any_special=True
                                    if all_specials[to_add] not in to_compare_est:
                                        
                                        to_compare_est.append(all_specials[to_add])
                                    
                                
                                else:
                                    
                                    to_compare_est.append(est_intersection_list[k][m][n])
                        
                        
                        # logging.error('TO COMPARE EST ' + str(to_compare_est))
                        
                    
                        sm = edit_distance.SequenceMatcher(a=to_compare_gt, b=to_compare_est)
                        dist = sm.distance()
                        # logging.error('DIST ' + str(dist))
                        to_store_dist.append(np.copy(dist))
                        edit_distances.append(np.copy(dist)/len(to_compare_gt))
                    
                # logging.error('EDIT DISTANCES ' + str(edit_distances))
                    
                for k in range(len(non_matched_gts)):
                    edit_distances.append(2)
                
                self.order_dist.append(np.mean(edit_distances))
                
                # logging.error('EDIT ' + str(np.mean(edit_distances)))
                
                self.to_store_dist = to_store_dist
                
                self.line_nums.append(len(haus_gt))
                
#                    vis_region = (1-targets['occ_mask'][-1].cpu().numpy())*targets['occ_mask'][0].cpu().numpy()
                # logging.error('VIS REGION ' + str(vis_region.shape))
#                    vis_pixels = np.sum(vis_region)
#                    self.occ_nums.append(vis_pixels)
                
                
                
                self.intersection_nums.append(total_intersection_count)
            else:

                
                self.order_dist.append(2)
                self.to_store_dist = None
            
                self.line_nums.append(len(haus_gt))
                
                
                self.intersection_nums.append(50)
                
                
            time4 = time.time()
            logging.error('ORDER ' + str(time4 - time3))
            '''
            RNN STUFF
            '''
            if rnn:
                selected_order_estimates = out['selected_order_estimates']
                double_selected_order_estimates = out['double_selected_order_estimates']
                 
                est_coefs = out['boxes']
                
#                    logging.error('OUT BOXES ' + str(est_coefs.shape))
                
                orig_coefs = targets['control_points'].cpu().numpy()
                orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
                
                rnn_loss = np.mean(np.sum(np.abs(np.expand_dims(orig_coefs,axis=1) - np.expand_dims(est_coefs,axis=0)),axis=-1),axis=-1)
                
                
                rnn_gt_indices, rnn_est_indices = linear_sum_assignment(rnn_loss)   
                
                self.rnn_gt_indices = rnn_gt_indices
                self.rnn_est_indices = rnn_est_indices
                
                
                # logging.error('RNN GT INDICES ' + str(rnn_gt_indices.shape))
                
#                    logging.error('RNN EST INDICES ' + str(rnn_est_indices))
                
                
                # logging.error('DOUBLE SELECTED ORDER ESTIMATES ' + str(double_selected_order_estimates.shape))
                
                arranged_est = double_selected_order_estimates[:,rnn_est_indices]
                
                arranged_est = np.concatenate([arranged_est[:,:, :-5][:,:,rnn_est_indices],arranged_est[:,:, -5:]],axis=-1)
                
                # logging.error('CONFUSION ARRANGED EST ' + str(arranged_est.shape))
                
                # arranged_gt = double_selected_order_estimates[:,rnn_est_indices]
                
                # arranged_gt = np.concatenate([arranged_est[:,:, :100][:,:,rnn_est_indices],arranged_est[:,:, 100:]],axis=-1)
                
                
                rnn_to_store_dist=[]
                rnn_edit_distances = []
                
                all_rnn_to_compare_gt = []
                all_rnn_to_compare_est = []
                
                to_run_index = np.min([len(rnn_gt_indices), len(rnn_est_indices)])
                
                if len(rnn_gt_indices) > len(rnn_est_indices):
                    not_matched = np.setdiff1d(np.arange(len(orig_coefs)), rnn_gt_indices)
                    for k in range(len(not_matched)):
                        
                        rnn_to_store_dist.append(len(gt_intersection_list[not_matched[k]]))
                        rnn_edit_distances.append(1)
                        
                
                for k in range(to_run_index):
                    
                    # logging.error('RNN EST ' + str(est_intersection_list[k]))
                    # logging.error('CUR SPECIALS GT ' + str(cur_specials_gt))
                    # logging.error('CUR SPECIALS GT POS ' + str(cur_specials_gt_pos))
                    if len(gt_intersection_list[rnn_gt_indices[k]]) == 0:
                        rnn_to_store_dist.append(0)
                        rnn_edit_distances.append(0)
                    
                    else:
                        cur_gt_inter = gt_intersection_list[rnn_gt_indices[k]]
                        rnn_to_compare_gt = []
                        rnn_to_compare_est = []
#                            logging.error('ORIG GT ' + str(gt_intersection_list[rnn_gt_indices[k]]))
                        for m in range(len(cur_gt_inter)):
                            if len(cur_gt_inter[m]) > 0:
                                for n in range(len(cur_gt_inter[m])):
                                    cora = np.where(cur_gt_inter[m][n] == rnn_gt_indices)[0]
                                    if len(cora) > 0:
                                        
                                        rnn_to_compare_gt.append(cora[0])
                                    else:
                                        rnn_to_compare_gt.append(cur_gt_inter[m][n] + len(orig_coefs))
                                        # continue
                            else:
                                
                                continue
#                            logging.error('RNN TO COMPARE GT ' + str(rnn_to_compare_gt))
                        
                        # cur_est_inter = arranged_est[:,rnn_est_indices[k],:]
                        cur_est_inter = arranged_est[:,k,:]
                        for m in range(len(cur_est_inter)):
                            
                            sel_cur = np.argmax(cur_est_inter[m])
                            
                            if sel_cur == (cur_est_inter.shape[-1] - 1):
                                break
                            elif sel_cur in rnn_to_compare_est:
                                continue
                            else:
                                rnn_to_compare_est.append(sel_cur)
                            
#                            logging.error('RNN TO COMPARE EST ' + str(rnn_to_compare_est))
#                            
                    
                        sm = edit_distance.SequenceMatcher(a=rnn_to_compare_gt, b=rnn_to_compare_est)
                        dist = sm.distance()
                        # logging.error('RNN DIST ' + str(dist))
                        rnn_to_store_dist.append(np.copy(dist))
                        rnn_edit_distances.append(np.copy(dist)/len(rnn_to_compare_gt))
                        
                        all_rnn_to_compare_gt.append(rnn_to_compare_gt)
                        all_rnn_to_compare_est.append(rnn_to_compare_est)
                        
                    # logging.error('RNN EDIT DISTANCES ' + str(rnn_edit_distances))
                    
                self.rnn_order_dist.append(np.mean(rnn_edit_distances))
                
                self.rnn_to_store_dist = rnn_to_store_dist
                
                self.all_rnn_to_compare_gt = all_rnn_to_compare_gt
                self.all_rnn_to_compare_est = all_rnn_to_compare_est
                
            '''
            PRECISION-RECALL
            '''
            res_interpolated_list = out['interpolated_points']
            num_estimates = len(res_interpolated_list)
            num_gt = len(haus_gt)
            
            if num_estimates == 0:
                for k in range(len(self.static_steps)):
                
                    self.static_pr_fn[k] = np.copy(self.static_pr_fn[k]) + len(haus_gt)*len(haus_gt[0]) 
                    
                self.unmatched_gt += len(haus_gt)
                
                # self.order_fn = self.order_fn + 
                
            
            else:
                
                '''
                PRE ASSOC
                '''
                
                m_g = len(np.unique(np.array(haus_idx)))
                self.matched_gt += m_g
                self.unmatched_gt += len(haus_gt) - m_g
                
                for est_id in range(num_estimates):
                    cur_gt = haus_gt[haus_idx[est_id]]
                    
                    dis = cdist(res_interpolated_list[est_id],cur_gt,'euclidean')
                    
                    res_dis = np.min(dis,axis=-1)
                    gt_dis = np.min(dis,axis=0)
                    
                    self.static_pr_total_gt += len(cur_gt)
                    self.static_pr_total_est += len(res_interpolated_list[est_id])
                    
                    for k in range(len(self.static_steps)):
                    
                        self.static_pr_tp[k] = np.copy(self.static_pr_tp[k]) + np.sum(res_dis < self.static_steps[k]) 
                        self.static_pr_fn[k] = np.copy(self.static_pr_fn[k]) + np.sum(gt_dis > self.static_steps[k]) 
                        self.static_pr_fp[k] = np.copy(self.static_pr_fp[k]) + np.sum(res_dis > self.static_steps[k]) 
                
                
                
            time5 = time.time()
            logging.error('F score ' + str(time5 - time4))
                
            '''
            MSE
            '''
            orig_coeffs = targets['control_points'].cpu().numpy()
            
            
            
            
            time6 = time.time()
            
            if not pinet:
                '''
                ASSOC LOSS
                '''
                # if polyline:
                #     assoc_est = out['pred_assoc']
                # else:
                assoc_est = out['assoc']
                
                # logging.error('ASSOC ' + str((np.min(assoc_est),np.max(assoc_est))))
#                    logging.error('ASSOC NEW ' + str(np.max(assoc_est)))
                
                gt_con_matrix = targets['con_matrix'].cpu().numpy()
                for est_id in range(num_estimates):
                    matched_gt = haus_idx[est_id]
                    cur_gt_assoc = gt_con_matrix[matched_gt]
                    cur_est_assoc = assoc_est[est_id]
                    
                    
                    
                    for m in range(len(cur_est_assoc)):
                        if cur_est_assoc[m] > assoc_thresh:
                            temp_id = haus_idx[m]
                            if temp_id == matched_gt:
                                self.assoc_tp += 1
                            elif cur_gt_assoc[temp_id] > 0.5: 
                                self.assoc_tp += 1
                            else:
                                self.assoc_fp += 1
                    
                for gt_id in range(len(gt_con_matrix)):
                    cur_gt_assoc = gt_con_matrix[gt_id]
                    
                    temp_mat = np.copy(cur_gt_assoc)
                    temp_mat = -temp_mat
                    
                    if not np.any(haus_idx == None):
                        
                    
                        if gt_id in haus_idx:
                            matched_ests = np.where(np.array(haus_idx)==gt_id)[0]
                            
                            
                            for m in range(len(cur_gt_assoc)):
                            
                                if cur_gt_assoc[m] > 0.5:
                                    
                                    if temp_mat[m] == -1:
                                        
                                        
                                        if m in haus_idx:
                                            other_ests = np.where(np.array(haus_idx)==m)[0]
                                             
                                            cur_est_assoc = assoc_est[matched_ests]
                                            
    #                                        temp_found = False
                                            for my_est in range(len(cur_est_assoc)):
                                                if np.any(cur_est_assoc[my_est][other_ests] > assoc_thresh):
    #                                                temp_found=True
                                                    temp_mat[m] = 1
                                                    break
                                                
                                            
                            self.assoc_fn += np.sum(temp_mat == -1)
                                            
                        else:
                            self.assoc_fn += np.sum(cur_gt_assoc)
                    else:
                        self.assoc_fn += np.sum(cur_gt_assoc)
                        
                
                # logging.error('ASSOC ' + str((self.assoc_tp, self.assoc_fp, self.assoc_fn)))
                        
            time7 = time.time()
            logging.error('ASSOC ' + str(time7 - time6))
            
          
        except Exception as e:
            logging.error('EXCEPTION IN REAL CONFUSION ')
            logging.error(str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def get_dilated_estimates(self, ar, dil_size):
        
        if dil_size == 0:
            return ar
        
        else:
            res = []
            for k in range(len(ar)):           
                dilated = ndimage.binary_dilation(ar[k], structure=self.structs[dil_size-1])  
                res.append(np.copy(dilated))
            return np.stack(res, axis=0)
    @property        
    def get_intermediate_dict(self):
        
        
        self.intermediate_metrics['order'] = np.array(self.order_dist)
        
        
        self.intermediate_metrics['per_poly_tp'] = self.per_poly_tp
        self.intermediate_metrics['per_poly_fp'] =  self.per_poly_fp 
        self.intermediate_metrics['per_poly_fn'] = self.per_poly_fn 
        
        self.intermediate_metrics['cross_per_poly_tp'] = self.cross_per_poly_tp
        self.intermediate_metrics['cross_per_poly_fp'] =  self.cross_per_poly_fp 
        self.intermediate_metrics['cross_per_poly_fn'] = self.cross_per_poly_fn 

        
        self.intermediate_metrics['common_per_poly_tp'] = self.common_per_poly_tp
        self.intermediate_metrics['common_per_poly_fp'] = self.common_per_poly_fp 
        self.intermediate_metrics['common_per_poly_fn'] = self.common_per_poly_fn

        
        
        for k in range(len(self.static_steps)):
            
            self.intermediate_metrics['static_pr_tp_'+str(self.static_steps[k])] = self.static_pr_tp[k]
            self.intermediate_metrics['static_pr_fp_'+str(self.static_steps[k])] = self.static_pr_fp[k] 
            self.intermediate_metrics['static_pr_fn_'+str(self.static_steps[k])] = self.static_pr_fn[k]
            
        
        
        self.intermediate_metrics['assoc_tp'] = self.assoc_tp
        self.intermediate_metrics['assoc_fp'] = self.assoc_fp 
        self.intermediate_metrics['assoc_fn'] = self.assoc_fn 
        
        self.intermediate_metrics['matched_gt'] = self.matched_gt
        self.intermediate_metrics['unmatched_gt'] = self.unmatched_gt
        
        self.intermediate_metrics['per_poly_tp_list'] = np.array(self.per_poly_tp_list)
        self.intermediate_metrics['per_poly_fn_list'] = np.array(self.per_poly_fn_list)
        self.intermediate_metrics['per_poly_fp_list'] = np.array(self.per_poly_fp_list)
        
        
        self.intermediate_metrics['cross_per_poly_tp_list'] = np.array(self.cross_per_poly_tp_list)
        self.intermediate_metrics['cross_per_poly_fn_list'] = np.array(self.cross_per_poly_fn_list)
        self.intermediate_metrics['cross_per_poly_fp_list'] = np.array(self.cross_per_poly_fp_list)
        
        
        self.intermediate_metrics['common_per_poly_tp_list'] = np.array(self.common_per_poly_tp_list)
        self.intermediate_metrics['common_per_poly_fn_list'] = np.array(self.common_per_poly_fn_list)
        self.intermediate_metrics['common_per_poly_fp_list'] = np.array(self.common_per_poly_fp_list)
        
        self.intermediate_metrics['intersection_nums'] = np.array(self.intersection_nums)
        self.intermediate_metrics['polygon_nums'] = np.array(self.polygon_nums)
        self.intermediate_metrics['common_polygon_nums'] = np.array(self.common_polygon_nums)
        self.intermediate_metrics['head_polygon_nums'] = np.array(self.head_polygon_nums)
        self.intermediate_metrics['line_nums'] = np.array(self.line_nums)
        self.intermediate_metrics['occ_nums'] = np.array(self.occ_nums)
        
        return self.intermediate_metrics
        
        
        
        
    @property
    def get_res_dict(self):


        self.static_metrics['cross_per_poly_tp'] = self.cross_per_poly_tp
        self.static_metrics['cross_per_poly_fp'] =  self.cross_per_poly_fp 
        self.static_metrics['cross_per_poly_fn'] = self.cross_per_poly_fn 
        
        self.static_metrics['cross_f'] = self.cross_per_poly_tp/(self.cross_per_poly_tp + (self.cross_per_poly_fp + self.cross_per_poly_fn + 0.001)/2 + 0.1) 
        
        
        self.static_metrics['order'] = np.mean(self.order_dist)
        
        self.static_metrics['rnn_order'] = np.mean(self.rnn_order_dist)
        
        self.static_metrics['per_miou_poly'] = self.per_poly_tp/(self.per_poly_tp + self.per_poly_fn + self.per_poly_fp + 0.01)
        self.static_metrics['per_precision_poly'] = self.per_poly_tp/(self.per_poly_tp  + self.per_poly_fp + 0.01)
        self.static_metrics['per_recall_poly'] = self.per_poly_tp/(self.per_poly_tp + self.per_poly_fn + 0.01)

        self.static_metrics['per_f'] = self.per_poly_tp/(self.per_poly_tp + (self.per_poly_fn + self.per_poly_fp +0.001)/2 + 0.01)

        self.static_metrics['common_per_miou_poly'] = self.common_per_poly_tp/(self.common_per_poly_tp + self.common_per_poly_fn + self.common_per_poly_fp + 0.001)
        self.static_metrics['common_per_precision_poly'] = self.common_per_poly_tp/(self.common_per_poly_tp  + self.common_per_poly_fp + 0.001)
        self.static_metrics['common_per_recall_poly'] = self.common_per_poly_tp/(self.common_per_poly_tp + self.common_per_poly_fn + 0.001)
        
        self.static_metrics['common_f'] = self.common_per_poly_tp/(self.common_per_poly_tp + (self.common_per_poly_fn + self.common_per_poly_fp + 0.01)/2 + 0.01)
        
        
        means_pre = []
        means_rec = []
        for k in range(len(self.static_steps)):
            
            self.static_metrics['precision_'+str(self.static_steps[k])] = self.static_pr_tp[k]/(self.static_pr_fp[k] + self.static_pr_tp[k] + 0.0001)
            self.static_metrics['recall_'+str(self.static_steps[k])] = self.static_pr_tp[k]/(self.static_pr_fn[k] + self.static_pr_tp[k] + 0.0001)
            
            means_pre.append(self.static_pr_tp[k]/(self.static_pr_fp[k] + self.static_pr_tp[k] + 0.0001))
            means_rec.append(self.static_pr_tp[k]/(self.static_pr_fn[k] + self.static_pr_tp[k] + 0.0001))
            # self.static_metrics['tp_'+str(self.static_steps[k])] = self.static_pr_tp[k]
            # self.static_metrics['fp_'+str(self.static_steps[k])] = self.static_pr_fp[k]
            # self.static_metrics['fn_'+str(self.static_steps[k])] = self.static_pr_fn[k]
        
        self.static_metrics['mean_pre'] = np.mean(means_pre)
        self.static_metrics['mean_rec'] = np.mean(means_rec)
        self.static_metrics['mean_f_score'] = np.mean(means_pre)*np.mean(means_rec)*2/(np.mean(means_pre)+np.mean(means_rec))


        self.static_metrics['assoc_iou'] = self.assoc_tp/(self.assoc_tp + self.assoc_fn + self.assoc_fp + 0.0001)
        
        self.static_metrics['assoc_precision'] = self.assoc_tp/(self.assoc_tp +  self.assoc_fp + 0.0001)
        self.static_metrics['assoc_recall'] = self.assoc_tp/(self.assoc_tp + self.assoc_fn +  0.0001)
        
        
        self.static_metrics['assoc_f'] = self.static_metrics['assoc_precision']*self.static_metrics['assoc_recall']*2/(self.static_metrics['assoc_precision']+self.static_metrics['assoc_recall'] + 0.001)
        
        
        self.static_metrics['matched_gt'] = self.matched_gt
        self.static_metrics['unmatched_gt'] = self.unmatched_gt
        self.static_metrics['detection_ratio'] = self.matched_gt/(self.matched_gt+self.unmatched_gt)
        # self.static_metrics['merged_matched_gt'] = self.merged_matched_gt
        # self.static_metrics['merged_unmatched_gt'] = self.merged_unmatched_gt
        
        '''
        OBJECT 
        '''
        self.object_metrics['refined_miou'] = self.refine_object_tp/(self.refine_object_tp +
                                                        self.refine_object_fp +self.refine_object_fn+0.0001 )
        
        self.object_metrics['refined_precision'] = self.refine_object_tp/(self.refine_object_tp +
                                                        self.refine_object_fp +0.0001 )
        self.object_metrics['refined_recall'] = self.refine_object_tp/(self.refine_object_tp +
                                                        self.refine_object_fn+0.0001 )
        
        
        self.object_metrics['argmax_refined_miou'] = self.argmax_refine_object_tp/(self.argmax_refine_object_tp +
                                                        self.argmax_refine_object_fp +self.argmax_refine_object_fn+0.0001 )
        
        self.object_metrics['argmax_refined_precision'] = self.argmax_refine_object_tp/(self.argmax_refine_object_tp +
                                                        self.argmax_refine_object_fp +0.0001 )
        self.object_metrics['argmax_refined_recall'] = self.argmax_refine_object_tp/(self.argmax_refine_object_tp +
                                                        self.argmax_refine_object_fn+0.0001 )
        
        
        return self.static_metrics
        
                
    @property
    def static_mse(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return np.mean(self.static_mse_list)
           

    @property
    def static_iou(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return self.static_line_tp / (self.static_line_tp + self.static_line_fn + self.static_line_fp + 0.0001)
    
    @property
    def poly_iou(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return self.poly_tp / (self.poly_tp + self.poly_fn + self.poly_fp + 0.0001)
    
    @property
    def per_poly_iou(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return self.per_poly_tp / (self.per_poly_tp + self.per_poly_fn + self.per_poly_fp + 0.0001)
    
    @property
    def order_error(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return np.mean(self.order_dist)
    
    @property
    def rnn_order_error(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return np.mean(self.rnn_order_dist)
    
    
    @property
    def object_seg_iou(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return self.object_tp / (self.object_tp + self.object_fn + self.object_fp + 0.0001)
    
    
    
    @property
    def object_class_iou(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        ious=[]
        for k in range(self.num_object_class):
            
            tp = self.object_cm[k,k]
            fp = np.sum(self.object_cm[:,k]) - tp
            fn = np.sum(self.object_cm[k,:]) - tp
            ious.append(tp/(tp + fp + fn))
            
        return np.array(ious)
    
    
    def reset(self):
#        self.tp = torch.zeros(self.num_class, dtype=torch.long)
#        self.fp = torch.zeros(self.num_class, dtype=torch.long)
#        self.fn = torch.zeros(self.num_class, dtype=torch.long)
#        self.tn = torch.zeros(self.num_class, dtype=torch.long)

        self.poly_tp=0
        self.poly_fp=0
        self.poly_fn=0
        
        self.order_dist=[]
        
        
        
        self.object_tp = np.zeros(self.num_object_class)
        self.object_fp = np.zeros(self.num_object_class)
        self.object_fn = np.zeros(self.num_object_class)
        self.object_tn = np.zeros(self.num_object_class)
        
        self.object_cm = np.zeros((self.num_object_class+1,self. num_object_class+1))
        
        
        self.static_tp = 0
        self.static_fn = 0
        self.static_fp = 0
        self.static_tn = 0
        
        self.static_mse_list = []
    
    @property
    def mean_iou(self):
        # Only compute mean over classes with at least one ground truth
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.iou[valid].mean())

    @property
    def dice(self):
        return 2 * self.tp.float() / (2 * self.tp + self.fp + self.fn).float()
    
    @property
    def macro_dice(self):
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.dice[valid].mean())
    
    @property
    def precision(self):
        return self.tp.float() / (self.tp + self.fp).float()
    
    @property
    def recall(self):
        return self.tp.float() / (self.tp + self.fn).float()