import matplotlib
matplotlib.use('Agg') 
from matplotlib.cm import get_cmap
import numpy as np

import logging
import sys
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

import scipy.ndimage as ndimage
from src.utils import bezier
import cv2 
#import networkx as nx

from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.optimize import linear_sum_assignment

from skimage import measure
image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]


def render_polygon(mask, polygon, shape, value=1):
    
#    logging.error('POLYGON ' + str(polygon.coords))
#    logging.error('EXTENTS ' + str(np.array(extents[:2])))
    to_mult = np.expand_dims(np.array([shape[1],shape[0]]),axis=0)
    polygon = polygon*to_mult
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)



def colorise(tensor, cmap, vmin=None, vmax=None):

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    
    tensor = tensor.detach().cpu().float()

    vmin = float(tensor.min()) if vmin is None else vmin
    vmax = float(tensor.max()) if vmax is None else vmax

    tensor = (tensor - vmin) / (vmax - vmin)
    return cmap(tensor.numpy())[..., :3]

def my_line_maker(points,size=(196,200)):
    
    res = np.zeros(size)
    
    if np.any(points < 0):
        return np.uint8(255*res)
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    return np.uint8(255*res)


def my_color_line_maker(points,endpoints,size=(196,200)):
    if len(endpoints) == 4:
        endpoints = np.reshape(endpoints,[2,2])
    res = np.zeros((size[0],size[1],3))
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    
    base_start = np.zeros((res.shape[0],res.shape[1]))
    base_start[np.min([int(endpoints[0,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[0,0]*size[1]),int(size[1]-1)])] = 1
    struct = ndimage.generate_binary_structure(2, 2)
    # struct = ndimage.generate_binary_structure(5, 2)
    
    # logging.error('STRUCT ' + str(struct))
    # logging.error('BASE START ' + str(base_start.shape))
    
    dilated = ndimage.binary_dilation(base_start>0, structure=struct)
    
    res[dilated,0] = 0
    res[dilated,1] = 1
    res[dilated,2] = 0
    
    base_end = np.zeros((res.shape[0],res.shape[1]))
    base_end[np.min([int(endpoints[1,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[1,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(base_end>0, structure=struct)
    
    res[dilated,0] = 1
    res[dilated,1] = 0
    res[dilated,2] = 0
    
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),0] = 1
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),1] = 0
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),2] = 0
    
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),0] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),1] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),2] = 1
    
    return np.uint8(255*res)


def my_float_line_maker(points,size=(196,200)):
    
    res = np.zeros(size)
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    return res

def hausdorff_match(out, target,pinet=False):
    
    # res_coef_list = out['interpolated_points']
    est_coefs = out['boxes']
    
    orig_coefs = target['control_points'].cpu().numpy()
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
    interpolated_origs = []
    
    for k in range(len(orig_coefs)):
        inter = bezier.interpolate_bezier(orig_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
    
    if len(est_coefs) == 0:
        return None,None, interpolated_origs
    
    
    
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    
    if pinet:
        second_dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs[:,::-1,:],axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
        dist_mat = np.min(np.stack([dist_mat,second_dist_mat],axis=0),axis=0)
    
    ind = np.argmin(dist_mat, axis=-1)
    min_vals = np.min(dist_mat,axis=-1)
    
  
        
        
    return min_vals, ind, interpolated_origs 

    
def merged_hausdorff_match(out, target):
    
    # res_coef_list = out['interpolated_points']
    est_coefs = out['merged_coeffs']
    
    # est_coefs = np.reshape(est_coefs,(est_coefs.shape[0],-1))
    
    orig_coefs = target['control_points'].cpu().numpy()
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
    interpolated_origs = []
    if len(est_coefs) == 0:
        return None,None, interpolated_origs
    for k in range(len(est_coefs)):
        inter = bezier.interpolate_bezier(est_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
    
    
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    
    ind = np.argmin(dist_mat, axis=-1)
    min_vals = np.min(dist_mat,axis=-1)
    
  
        
        
    return min_vals, ind, interpolated_origs 


def get_selected_estimates(targets, outputs, thresh = 0.5, do_polygons=False, poly_thresh=0.5, poly_hamming_thresh=0.5):
    
    res = []
    for b in range(len(targets)):
        
        temp_dict = dict()
        
        scores = targets[b]['scores'].detach().cpu().numpy()
        probs = targets[b]['probs'].detach().cpu().numpy()
        labels = targets[b]['labels'].detach().cpu().numpy()
        coeffs = targets[b]['boxes'].detach().cpu().numpy()
        endpoints = targets[b]['endpoints'].detach().cpu().numpy()
        assoc = targets[b]['assoc'].detach().cpu().numpy()
        
        
        selecteds = probs[:,1] > thresh
        
        
        # logging.error('ASSOC IN GET SELECTED ' + str(assoc.shape))
        if 'poly_prob' in outputs:
        
            poly_hamming = np.squeeze(outputs['poly_hamming'].detach().cpu().numpy()) 
            poly_prob = np.squeeze(outputs['poly_prob'].softmax(-1).detach().cpu().numpy())
            poly_centers = np.squeeze(outputs['poly_centers'].detach().cpu().numpy())
            temp_ham = poly_hamming[:,:-5]
        
            inter_poly_hamming = np.concatenate([temp_ham[:,selecteds], poly_hamming[:,-5:]],axis=-1)
            
            
            detected_polys = poly_prob[:,1] > poly_thresh
            
        
            
            
        detected_scores = probs[selecteds,1]
        detected_coeffs = coeffs[selecteds,...]
        detected_endpoints = endpoints[selecteds,...]
        
#        detected_con_matrix = assoc[selecteds]
#        detected_con_matrix = detected_con_matrix[:,selecteds]
        
        
        all_roads = np.zeros((196,200,3),np.float32)
        coef_all_roads = np.zeros((196,200,3),np.float32)
        if len(detected_scores) > 0:
            if 'seq_estimates' in outputs:
            
                seq_estimates = outputs['seq_estimates'].detach().cpu().numpy() 
                
                selected_order_estimates = seq_estimates[:,selecteds,:]
                
                double_selected_order_estimates = np.concatenate([selected_order_estimates[:,:, :100][:,:,selecteds],selected_order_estimates[:,:, 100:]],axis=-1)
                temp_dict['selected_order_estimates'] = selected_order_estimates
                temp_dict['double_selected_order_estimates'] = double_selected_order_estimates
             
            else:
                temp_dict['selected_order_estimates'] = []
                temp_dict['double_selected_order_estimates'] = []
                
                
            if 'poly_prob' in outputs:
                detected_poly_probs = poly_prob[detected_polys,1]
                if len(detected_poly_probs) > 0:
                
                    detected_poly_centers = poly_centers[detected_polys]
                    
                    temp_dict['half_selected_polys'] = inter_poly_hamming
                    
                    detected_poly_hamming = inter_poly_hamming[detected_polys] > poly_hamming_thresh
                    
                    not_all_zeros = np.logical_not(np.all(detected_poly_hamming, axis=-1))
                    if np.sum(not_all_zeros)>0:
                        detected_poly_hamming = detected_poly_hamming[not_all_zeros]
                        detected_poly_probs = detected_poly_probs[not_all_zeros]
                        detected_poly_centers = detected_poly_centers[not_all_zeros]
                        unique_hamming = np.unique(detected_poly_hamming,axis=0)

                        temp_dict['poly_probs'] = detected_poly_probs
                        temp_dict['poly_centers'] = detected_poly_centers
                        temp_dict['poly_hamming'] = unique_hamming
                        
                    else:
                        temp_dict['poly_probs'] = []
                        temp_dict['poly_centers'] = []
                        temp_dict['poly_hamming'] = []
                        
                        temp_dict['half_selected_polys'] = []
                else:
                    temp_dict['poly_probs'] = []
                    temp_dict['poly_centers'] = []
                    temp_dict['poly_hamming'] = []
                    
                    temp_dict['half_selected_polys'] = []
            temp_dict['scores'] = detected_scores
            temp_dict['boxes'] = detected_coeffs
            temp_dict['endpoints'] = detected_endpoints
            temp_dict['assoc'] = assoc
           
            to_merge = {'assoc': assoc, 'start':None,'fin':None, 'boxes':detected_coeffs}
            merged = get_merged_coeffs(to_merge)
            
            
            
            temp_dict['merged_coeffs'] = merged
        
            res_list = []
            res_coef_list=[]
            
            res_interpolated_list=[]
            # res_assoc_list = []
            
            for k in range(len(detected_scores)):
                

                control = detected_coeffs[k]
                
                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                
                interpolated = bezier.interpolate_bezier(control,100)
                
                res_interpolated_list.append(np.copy(interpolated))
                
                line = my_color_line_maker(interpolated,detected_endpoints[k],size=(196,200))
                line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
                res_list.append(line)
                res_coef_list.append(line2)
                all_roads = all_roads + np.float32(line)
                coef_all_roads = coef_all_roads + np.float32(line2)
            
            temp_dict['lines'] = res_list
            temp_dict['coef_lines'] = res_coef_list
            
            temp_dict['interpolated_points'] = res_interpolated_list
            
            temp_dict['all_roads'] = all_roads
            temp_dict['coef_all_roads'] = coef_all_roads
            temp_dict['labels'] = labels[selecteds]
            
            merged_interpolated_list=[]
            for k in range(len(merged)):
                

                control = merged[k]
                
                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                
                merged_interpolated = bezier.interpolate_bezier(control,100)
                
                merged_interpolated_list.append(np.copy(merged_interpolated))
            
            
            temp_dict['merged_interpolated_points'] = merged_interpolated_list
            
            
            
            
            
        else:
            
            logging.error('DETECTED NOTHING')
            temp_dict['scores'] = []
            temp_dict['interpolated_points'] = []
            temp_dict['scores'] = []
            temp_dict['boxes'] = []
            temp_dict['lines'] = []
            temp_dict['coef_lines'] = []
            temp_dict['all_roads'] = []
            temp_dict['coef_all_roads'] = []
            temp_dict['labels'] = []
            temp_dict['assoc'] = []
            temp_dict['start'] = []
            temp_dict['fin'] = []
            temp_dict['merged_interpolated_points'] = []
            temp_dict['merged_coeffs'] = []
            
            temp_dict['poly_probs'] = []
            temp_dict['poly_centers'] = []
            temp_dict['poly_hamming'] = []
        res.append(temp_dict)
        
    return res
            

def get_vertices(adj):

    ins = []
    outs = []
    
    for k in range(len(adj)):
    #for k in range(7):
        for m in range(len(adj)):
        
            if adj[k,m] > 0.5:
                if len(ins) > 0:
                    ins_exists = False
                    out_exists = False
    
                    for temin in range(len(ins)):
                        if k in ins[temin]:
                            if not (m in outs[temin]):
                                outs[temin].append(m)
                            ins_exists=True
                            break
                    
                    if not ins_exists:
                        for temin in range(len(outs)):
                            if m in outs[temin]:
                                if not (k in ins[temin]):
                                    ins[temin].append(k)
                                out_exists=True
                                break
                        
                        if not out_exists:
                            ins.append([k])
                            outs.append([m])
                            
                else:
                    ins.append([k])
                    outs.append([m])
                            
    
    return ins, outs                    
   

def gather_all_ends(adj):

    clusters = []
    
    for k in range(len(adj)):
    #for k in range(7):
        for m in range(len(adj)):
        
            if adj[k,m] > 0.5:
                if len(clusters) > 0:
                    ins_exists = False
                    out_exists = False
    
                    for temin in range(len(clusters)):
                        if k in clusters[temin]:
                            if not (m in clusters[temin]):
                                clusters[temin].append(m)
                            ins_exists=True
                            break
                    
                    if not ins_exists:
                        for temin in range(len(clusters)):
                            if m in clusters[temin]:
                                if not (k in clusters[temin]):
                                    clusters[temin].append(k)
                                out_exists=True
                                break
                        
                        if not out_exists:
                            clusters.append([k,m])
                            
                else:
                    clusters.append([k,m])
                            
    
    return clusters             
#

def get_merged_coeffs(targets):
    

    coeffs = targets['boxes']

    assoc = targets['assoc'] 

           
    diag_mask = np.eye(len(assoc))

    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask
    
    corrected_coeffs = np.copy(coeffs)
    
    
    ins, outs = get_vertices(assoc)
    
    
    
    for k in range(len(ins)):
        all_points=[]
        for m in ins[k]:
            all_points.append(corrected_coeffs[m,-1])
            
        for m in outs[k]:
            all_points.append(corrected_coeffs[m,0])
            
        
        av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
        for m in ins[k]:
            corrected_coeffs[m,-1] = av_p
            
        for m in outs[k]:
            corrected_coeffs[m,0] = av_p
    
    
    return corrected_coeffs


def get_merged_network(targets):
    
    try:

        coeffs = targets['boxes']
        

        assoc = targets['assoc'] 

        diag_mask = np.eye(len(assoc))
        
  
        
        diag_mask = 1 - diag_mask
        assoc = assoc*diag_mask
        
        corrected_coeffs = np.copy(coeffs)
        
    
            
        
        ins, outs = get_vertices(assoc)
        
        
        
        for k in range(len(ins)):
            all_points=[]
            for m in ins[k]:
                all_points.append(corrected_coeffs[m,-1])
                
            for m in outs[k]:
                all_points.append(corrected_coeffs[m,0])
                
            
            av_p = np.mean(np.stack(all_points,axis=0),axis=0)
            
            for m in ins[k]:
                corrected_coeffs[m,-1] = av_p
                
            for m in outs[k]:
                corrected_coeffs[m,0] = av_p
        
        lines=[]
                
        for k in range(len(corrected_coeffs)):        
            control = corrected_coeffs[k]
            coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
            interpolated = bezier.interpolate_bezier(control)
            line = np.float32(my_color_line_maker(interpolated,coef_endpoints,size=(196,200)))/255
            lines.append(line)    
            
        return lines
    
    except Exception as e:
        logging.error('GET MERGED NETWORK ')
        logging.error(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
        
        

def visual_est(images,targets, matched_poly, raw_out, save_path,name=None):
 
    try:
    
        b=0
        
        
        merged = get_merged_network(targets[0])
        
        
        
        if len(merged) > 0:
            
            merged = np.sum(np.stack(merged,axis=0),axis=0)
            merged = np.uint8(np.clip(merged, 0, 1)*255)
            res = Image.fromarray(merged)
            
            if name==None:
                res.save(os.path.join(save_path,'batch_'+str(b) + '_merged_road.jpg'))
                
            else:
                res.save(os.path.join(save_path,name + '_merged_road.jpg'))
        else:
            logging.error('EMPTY MERGED')
        
        scores = targets[b]['scores']
        labels = targets[b]['labels']
        coeffs = targets[b]['boxes']
        
        res_list = targets[b]['lines'] 
        res_coef_list = targets[b]['coef_lines'] 
        all_roads = targets[b]['all_roads'] 
        coef_all_roads = targets[b]['coef_all_roads'] 
        assoc = targets[b]['assoc'] 
            
        
        # logging.error('VIS EST '+ str(assoc.shape))
        if len(res_list) > 0:
            
            if 'my_blob_mat' in targets[b]:
                gt_blob_mat = np.squeeze(targets[b]['my_blob_mat'])
              
                
                cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(gt_blob_mat) + 1))
                
                colored_image = cm(np.int64(gt_blob_mat))
                
                Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'estimate_common.png' ))
            
            
            all_lanes = np.zeros((196,200))
#            for k in range(len(res_list)):
                
                # res = Image.fromarray(res_list[k])
                # res_coef = Image.fromarray(res_coef_list[k])
                # if name==None:
                #     res.save(os.path.join(save_path,'batch_'+str(b) + '_est_interp_road_'+str(k)+'.jpg'))
                #     res_coef.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_interp_road_'+str(k)+'.jpg'))
     
                
                # else:
                #     res.save(os.path.join(save_path,name + '_est_interp_road_'+str(k)+'.jpg'))
                #     res_coef.save(os.path.join(save_path,name + '_est_coef_interp_road_'+str(k)+'.jpg'))
      

                # merged, merged_coeffs = get_merged_lines(coeffs,assoc,k)
                # for m in range(len(assoc[k])):
                #     if assoc[k][m] > 0:
                #         first_one = np.float32(res_coef_list[k])/255
                #         second_one = np.float32(res_coef_list[m])/255
                        
                #         tot = np.clip(first_one + second_one,0,1)
                #         temp_img = Image.fromarray(np.uint8( tot*255))
                        
                #         if name==None:
                #             temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                        
                #         else:
                #             temp_img.save(os.path.join(save_path,name + '_est_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                     
            all_lanes = np.uint8(np.clip(all_lanes,0,1)*255)
            if name==None:
                
                
                
                all_roads = np.uint8(np.clip(all_roads,0,1)*255)
                temp_img = Image.fromarray(all_roads)
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_all_roads.jpg' ))       
                
                coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
                temp_img = Image.fromarray(coef_all_roads)
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_all_roads.jpg' ))       
            else:
             
                
                all_roads = np.uint8(np.clip(all_roads,0,1)*255)
                temp_img = Image.fromarray(all_roads)
                temp_img.save(os.path.join(save_path,name + '_est_all_roads.jpg' ))    
                
                coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
                temp_img = Image.fromarray(coef_all_roads)
                temp_img.save(os.path.join(save_path,name + '_est_coef_all_roads.jpg' ))    

    except Exception as e:
        logging.error('VISUAL EST ')
        logging.error(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
    
def visual_masks_gt(images,targets,save_path,name=None):
 
    
    for b in range(len(targets)):
        img_centers = targets[b]['center_img']
        
        img_centers = img_centers.cpu().numpy()
        
        orig_img_centers = targets[b]['orig_center_img']
        
        orig_img_centers = orig_img_centers.cpu().numpy()
        
        roads = targets[b]['roads'].cpu().numpy()
        
        all_endpoints = targets[b]['endpoints'].cpu().numpy()
        
        true_assoc = targets[b]['con_matrix'].cpu().numpy()
#        

        
        all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        coef_all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        
    
        
        orig_coefs = targets[b]['control_points'].cpu().numpy()
    
        coef_endpoints = get_endpoints_from_coeffs(orig_coefs)
    
        all_masks = targets[b]['mask'].cpu().numpy()
        occ_img = Image.fromarray(np.uint8(255*all_masks[1]))
        vis_img = Image.fromarray(np.uint8(255*all_masks[0]))
        
        all_lanes = np.zeros((196,200))
        
        gt_blob_mat = np.squeeze(targets[b]['blob_mat'].cpu().numpy())
      
        
        cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(gt_blob_mat) + 1))
        
        colored_image = cm(np.int64(gt_blob_mat))
        
        Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'poly_all_gt.png' ))
        
     
            
        for k in range(len(roads)):
            cur_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),all_endpoints[k])
            cur_coef_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),coef_endpoints[k])
            
#            all_roads[img_centers == roads[k]] = 1
            temp_img = Image.fromarray(np.uint8(cur_full*255))
            
            temp_coef_img = Image.fromarray(np.uint8(cur_coef_full*255))
            
            all_roads = all_roads + cur_full
            coef_all_roads = coef_all_roads + cur_coef_full
            
           
            
            # for m in range(len(true_assoc[k])):
            #     if true_assoc[k][m] > 0.5:
            #         first_one = add_endpoints_to_line(np.float32(img_centers == roads[k]),coef_endpoints[k])
            
            #         second_one = add_endpoints_to_line(np.float32(img_centers == roads[m]),coef_endpoints[m])
            
            #         tot = np.clip(first_one + second_one,0,1)
            #         temp_img = Image.fromarray(np.uint8( tot*255))
                    
            #         if name==None:
            #             temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_true_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                    
            #         else:
            #             temp_img.save(os.path.join(save_path,name + '_true_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                 
                    
         

            
#             if name==None:

#                 temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_road_'+str(k)+'.jpg' ))
#                 temp_coef_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_road_'+str(k)+'.jpg' ))
                

#             else:

#                 temp_img.save(os.path.join(save_path,name + '_visible_road_'+str(k)+'.jpg' ))
#                 temp_coef_img.save(os.path.join(save_path,name + '_coef_visible_road_'+str(k)+'.jpg' ))

                
        all_roads = np.clip(all_roads,0,1)
        coef_all_roads = np.clip(coef_all_roads,0,1)
        
        all_lanes = np.clip(all_lanes,0,1)
        
        
        if name==None:
        
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_all_roads.jpg' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_all_roads.jpg' ))
            
        else:
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_visible_all_roads.jpg' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_coef_visible_all_roads.jpg' ))

def process_image(image):
    
    image = np.transpose(image,(0,2,3,1))
    
    image = (image + 1)/2*255
    return image
 
    
def add_endpoints_to_line(ar,endpoints):
    if len(endpoints) == 4:
        endpoints = np.reshape(endpoints,[2,2])
    size = ar.shape
    res = np.zeros((ar.shape[0],ar.shape[1],3))
    res[ar > 0] = 1
    
    # logging.error('AR SHAPE ' + str(ar.shape))
    # logging.error('ENDPOINTS SHAPE ' + str(endpoints.shape))
    # logging.error('ENDPOINTS ' + str(endpoints))
    
    base_start = np.zeros((ar.shape[0],ar.shape[1]))
    base_start[np.min([int(endpoints[0,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[0,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(5, 2)
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(base_start>0, structure=struct)
    
    res[dilated,0] = 0
    res[dilated,1] = 1
    res[dilated,2] = 0
    
    base_end = np.zeros((ar.shape[0],ar.shape[1]))
    base_end[np.min([int(endpoints[1,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[1,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(base_end>0, structure=struct)
    
    res[dilated,0] = 1
    res[dilated,1] = 0
    res[dilated,2] = 0
    
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),0] = 1
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),1] = 0
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),2] = 0
    
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),0] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),1] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),2] = 1
    
    return res
    

def get_endpoints_from_coeffs(coeffs):
    
    start = coeffs[:,:2]
    end = coeffs[:,-2:]
    
    return np.concatenate([start,end],axis=-1)
    

def get_poly_pretrain_targets(targets, out, thresh):
    
    
    b=0
    probs = out[b]['probs'].detach().cpu().numpy()
    
    all_ind = np.arange(len(probs))
    selecteds = probs[:,1] > thresh
    
    sel_ind = all_ind[selecteds]
    
    
    sel_ind = np.concatenate([sel_ind, np.array([100,101,102,103,104])],axis=0)
    
    poly_centers = targets['pre_poly_centers'].numpy()
    poly_one_hots = targets['pre_poly_one_hots'].numpy()
    real_one_hot = np.zeros((len(poly_one_hots),len(probs) + 5))
    
    for k in range(len(poly_one_hots)):
        real_one_hot[k][sel_ind[poly_one_hots[k] > 0]] = 1
        
    return real_one_hot, poly_centers
  

def pinet_get_polygons(targets):
    
        
    coefs = targets['boxes'] 
        
    
                
    
    res = []
   
    
    temp_dict = dict()
    
    
    if len(coefs) > 0:
        orig_inter = np.stack(targets['interpolated_points'],axis=0)
    
        n_points = orig_inter[0].shape[0]
        
        '''
        BUILD BOUNDARY LINES
        '''
        upper = np.stack([np.linspace(0,1,orig_inter.shape[1]), np.zeros((n_points))],axis=-1)
        lower = np.stack([np.linspace(0,1,orig_inter.shape[1]), np.ones((n_points))],axis=-1)
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,orig_inter.shape[1])],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,orig_inter.shape[1])],axis=-1)
        
        
        # bev_mask_path = 'C:\\winpy\\WPy64-3850\\codes\\simplice-net\\real_labels_4_batch_0_class_6.png'
        # bev_mask = np.array(Image.open(bev_mask_path), np.uint8)
        
        bev_left = np.stack([np.linspace(0,0.54,orig_inter.shape[1]), np.linspace(30/200,1,orig_inter.shape[1])],axis=-1)
        
        bev_right = np.stack([np.linspace(0.46,1,orig_inter.shape[1]), np.linspace(1,30/200,orig_inter.shape[1])],axis=-1)
        
        boundaries = np.stack([upper,  left, right, bev_left, bev_right, lower],axis=0)
        inter = np.float32(np.concatenate([orig_inter,boundaries],axis=0))
        
       
        
        grid_x = np.linspace(0,1,200)
        grid_y = np.linspace(0,1,196)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        
        all_lines = []
        for k in range(len(inter)):
            all_lines.append(np.float32(my_line_maker(inter[k]) > 0 ))
        
        road_mat = np.clip(np.sum(np.stack(all_lines,axis=0),axis=0),0,1)
        
        road_mat = np.uint8(1-road_mat)
        
        dist_trans = cv2.distanceTransform(road_mat,cv2.DIST_L2,5)
        
        # logging.error('DISTANCE TRANSFORM OKAY')
        
        thresh_dist = dist_trans > 2
        
        blob_mat, num_blobs = measure.label(thresh_dist, connectivity=2, return_num=True)
        
        one_hots = []
        real_hots=[]
        all_polys = []
        all_centers = []
        blob_ids=[]
        # to_removes = [np.array([len(orig_inter)+1, len(orig_inter)+2,len(orig_inter)+4]),
        #               np.array([len(orig_inter)+1, len(orig_inter)+3,len(orig_inter)+5])]
        for k in range(num_blobs):
            cur_x = mesh_x[(blob_mat == (k+1)) & thresh_dist]
            cur_y = mesh_y[(blob_mat == (k+1)) & thresh_dist]
            
            flat_mesh = np.stack([cur_x, cur_y],axis=-1)
            
            cur_dist = np.expand_dims(inter,axis=0) - np.expand_dims(np.expand_dims(flat_mesh,axis=1),axis=1)
        
            uniques = np.unique(np.argmin(np.min(np.sum(np.abs(cur_dist),axis=-1),axis=-1),axis=-1))    
       
                
            if (len(inter)-1) not in uniques:
                blob_ids.append(np.copy(k+1))
                all_polys.append(np.copy(uniques))
                o_h = np.zeros((len(inter)-1))
                o_h[uniques] = 1
                
                real_one_hot = o_h
                # print(str(o_h))
                # plt.imshow((blob_mat == (k+1)) & thresh_dist)
                # o_h = np.concatenate([o_h[:len(orig_inter)+1], o_h[len(orig_inter)+2:])
                
                real_hots.append(np.copy(real_one_hot))
                
                one_hots.append(np.copy(o_h))
                m_d = dist_trans * ((blob_mat == (k+1)) & thresh_dist)
                m_loc = np.argmax(m_d)
                x_center = mesh_x.flatten()[m_loc]
                y_center = mesh_y.flatten()[m_loc]
                my_blob_center = np.array([x_center, y_center])
                all_centers.append(np.copy(my_blob_center))
                
        
        
                
        return np.array(all_centers), np.array(one_hots), blob_mat, np.array(blob_ids), np.array(real_hots)
        
    else:
        return None, None, None, None, None
def get_polygons(targets, thresh):
        
        res = []
        b=0
        
        temp_dict = dict()
        
        # scores = targets[b]['scores'].detach().cpu().numpy()
        probs = targets[b]['probs'].detach().cpu().numpy()
        
        # labels = targets[b]['labels'].detach().cpu().numpy()
        coeffs = targets[b]['boxes'].detach().cpu().numpy()
        # endpoints = targets[b]['endpoints'].detach().cpu().numpy()
        
        assoc = targets[b]['assoc'].detach().cpu().numpy()

        # logging.error('ASSOC IN GET SELECTED ' + str(assoc.shape))
        
        all_ind = np.arange(len(probs))
        selecteds = probs[:,1] > thresh
        
        sel_ind = all_ind[selecteds]
    
    
        sel_ind = np.concatenate([sel_ind, np.array([100,101,102,103,104])],axis=0)
        detected_scores = probs[selecteds,1]
        detected_coeffs = coeffs[selecteds,...]
        # detected_endpoints = endpoints[selecteds,...]
        
#        detected_con_matrix = assoc[selecteds]
#        detected_con_matrix = detected_con_matrix[:,selecteds]
        
        
        # all_roads = np.zeros((196,200,3),np.float32)
        # coef_all_roads = np.zeros((196,200,3),np.float32)
        if len(detected_scores) > 0:
            
            
            to_merge = {'assoc': assoc, 'start':None,'fin':None, 'boxes':detected_coeffs}
            detected_coeffs = get_merged_coeffs(to_merge)
        
            n_points = 200
    
            orig_inter = bezier.batch_interpolate(detected_coeffs, n_points) # N x n_points x 2
         
            
            '''
            BUILD BOUNDARY LINES
            '''
            upper = np.stack([np.linspace(0,1,orig_inter.shape[1]), np.zeros((n_points))],axis=-1)
            lower = np.stack([np.linspace(0,1,orig_inter.shape[1]), np.ones((n_points))],axis=-1)
            left = np.stack([ np.zeros((n_points)),np.linspace(0,1,orig_inter.shape[1])],axis=-1)
            right = np.stack([np.ones((n_points)),np.linspace(0,1,orig_inter.shape[1])],axis=-1)
            
            
            # bev_mask_path = 'C:\\winpy\\WPy64-3850\\codes\\simplice-net\\real_labels_4_batch_0_class_6.png'
            # bev_mask = np.array(Image.open(bev_mask_path), np.uint8)
            
            bev_left = np.stack([np.linspace(0,0.54,orig_inter.shape[1]), np.linspace(30/200,1,orig_inter.shape[1])],axis=-1)
            
            bev_right = np.stack([np.linspace(0.46,1,orig_inter.shape[1]), np.linspace(1,30/200,orig_inter.shape[1])],axis=-1)
            
            boundaries = np.stack([upper,  left, right, bev_left, bev_right, lower],axis=0)
            inter = np.float32(np.concatenate([orig_inter,boundaries],axis=0))
            
           
            
            grid_x = np.linspace(0,1,200)
            grid_y = np.linspace(0,1,196)
            mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
            
            all_lines = []
            for k in range(len(inter)):
                all_lines.append(np.float32(my_line_maker(inter[k]) > 0 ))
            
            road_mat = np.clip(np.sum(np.stack(all_lines,axis=0),axis=0),0,1)
            
            road_mat = np.uint8(1-road_mat)
            
            dist_trans = cv2.distanceTransform(road_mat,cv2.DIST_L2,5)
            
            # logging.error('DISTANCE TRANSFORM OKAY')
            
            thresh_dist = dist_trans > 2
            
            blob_mat, num_blobs = measure.label(thresh_dist, connectivity=2, return_num=True)
            
            one_hots = []
            real_hots=[]
            all_polys = []
            all_centers = []
            blob_ids=[]
            # to_removes = [np.array([len(orig_inter)+1, len(orig_inter)+2,len(orig_inter)+4]),
            #               np.array([len(orig_inter)+1, len(orig_inter)+3,len(orig_inter)+5])]
            for k in range(num_blobs):
                cur_x = mesh_x[(blob_mat == (k+1)) & thresh_dist]
                cur_y = mesh_y[(blob_mat == (k+1)) & thresh_dist]
                
                flat_mesh = np.stack([cur_x, cur_y],axis=-1)
                
                cur_dist = np.expand_dims(inter,axis=0) - np.expand_dims(np.expand_dims(flat_mesh,axis=1),axis=1)
            
                uniques = np.unique(np.argmin(np.min(np.sum(np.abs(cur_dist),axis=-1),axis=-1),axis=-1))    
           
                    
                if (len(inter)-1) not in uniques:
                    
                    blob_ids.append(np.copy(k+1))
                    all_polys.append(np.copy(uniques))
                    o_h = np.zeros((len(inter)-1))
                    o_h[uniques] = 1
                    
                    real_one_hot = np.zeros((len(probs) + 5))
                    real_one_hot[sel_ind[o_h > 0]] = 1
                    # print(str(o_h))
                    # plt.imshow((blob_mat == (k+1)) & thresh_dist)
                    # o_h = np.concatenate([o_h[:len(orig_inter)+1], o_h[len(orig_inter)+2:])
                    
                    real_hots.append(np.copy(real_one_hot))
                    
                    one_hots.append(np.copy(o_h))
                    m_d = dist_trans * ((blob_mat == (k+1)) & thresh_dist)
                    m_loc = np.argmax(m_d)
                    x_center = mesh_x.flatten()[m_loc]
                    y_center = mesh_y.flatten()[m_loc]
                    my_blob_center = np.array([x_center, y_center])
                    all_centers.append(np.copy(my_blob_center))
                    
            
            
                    
            return np.array(all_centers), np.array(one_hots), blob_mat, np.array(blob_ids), np.array(real_hots)
            
        else:
            return None, None, None, None, None
        
        
def save_train_order_rnn( outputs, targets,  train_order , match_static_indices, config,save_path, train_rnn_matched_labels=None, gt=True):
    inter_points = bezier.batch_interpolate(outputs['pred_boxes'].view(-1, 3, 2).detach().cpu().numpy(), 100)
    
    
    all_made_lines = []
    for k in range(len(inter_points)):
            
        cur_est = inter_points[k,...]
        cur_est = np.float32(my_line_maker(cur_est))/255
        all_made_lines.append(np.copy(cur_est)) 
        
        
    n_points = 200
    
    upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
    
    left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
    right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
    
    bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
        
    bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
        
    boundary_lines = []
    
    boundary_lines.append(np.float32(my_line_maker(upper))/255)
    boundary_lines.append(np.float32(my_line_maker(left))/255)
    boundary_lines.append(np.float32(my_line_maker(right))/255)
    boundary_lines.append(np.float32(my_line_maker(bev_left))/255)
    boundary_lines.append(np.float32(my_line_maker(bev_right))/255)
    
    all_made_lines = np.concatenate([np.stack(all_made_lines,axis=0),np.stack(boundary_lines,axis=0)],axis=0)
  
    if not gt:
        if not np.any(train_order == None):
            train_order, train_order_sel = train_order
            for k in range(len(train_order)):
                gt_ar = np.copy(all_made_lines[int(train_order_sel[k])])
                gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
                base = np.copy(gt_ar)
                for m in range(len(train_order[k])):
                    if train_order[k,m] == 105:
                        break
                
                    temp = all_made_lines[int(train_order[k][m])]
             
                    temp_img = np.copy(base)
                    temp_img[temp > 0.5,0] = 1
                    temp_img[temp > 0.5,1:] = 0
                    
                    temp_img = Image.fromarray(np.uint8(temp_img*255))  
                    temp_img.save(os.path.join(save_path,'rnn_train_orders_'+str(k) +'_' + str(m)+'.jpg' ))
                    
                    
    else:
        if not np.any(train_rnn_matched_labels == None):
            if not np.any( match_static_indices == None):
                
                
                my_src_idx = np.int64(match_static_indices[0][0].cpu().numpy())
                my_tgt_idx = np.int64(match_static_indices[0][1].cpu().numpy())
                
                gt_orders = targets['gt_order_labels']
                
                
                seq_estimates = outputs['seq_estimates'].detach().cpu().numpy() 
                
                seq_estimates = seq_estimates[:,my_src_idx]
                for k in range(len(my_src_idx)):
                    
                    '''
                    GT
                    '''
                    gt_ar = np.copy(all_made_lines[my_src_idx[k]])
                    gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
                    base = np.copy(gt_ar)
                    for m in range(len(train_rnn_matched_labels[k])):
                        if train_rnn_matched_labels[k,m] == 105:
                            continue
                    
                        temp = all_made_lines[int(train_rnn_matched_labels[k][m])]
                 
                        temp_img = np.copy(base)
                        temp_img[temp > 0.5,0] = 1
                        temp_img[temp > 0.5,1:] = 0
                        
                        temp_img = Image.fromarray(np.uint8(temp_img*255))  
                        temp_img.save(os.path.join(save_path,'rnn_gt_matched_train_orders_'+str(k) +'_' + str(m)+'.jpg' ))
                        
                    '''
                    EST
                    '''
                
                    
                    gt_ar = np.copy(all_made_lines[my_src_idx[k]])
                    gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
                    base = np.copy(gt_ar)
                    for m in range(len(seq_estimates)):
                        
                        cur_li = np.argmax(seq_estimates[m,k])
                        if cur_li == 105:
                            continue
                    
                        temp = all_made_lines[cur_li]
                 
                        temp_img = np.copy(base)
                        temp_img[temp > 0.5,0] = 1
                        temp_img[temp > 0.5,1:] = 0
                        
                        temp_img = Image.fromarray(np.uint8(temp_img*255))  
                        temp_img.save(os.path.join(save_path,'rnn_estimated_matched_train_orders_'+str(k) +'_' + str(m)+'.jpg' ))
                        
                    
def save_matched_results(inter_dict, out, outputs, naive_matched_labels, mod_ham,poly_voronoi, matched_poly, targets,target_ids,config,save_path, val=False, common_poly = None):
    
    _, target_ids = target_ids
    
    # inter_points = inter_dict['interpolated'].detach().cpu().numpy()
    img_centers = targets['center_img']
    
    try:
    
        origs = targets['origs'].cpu().numpy()
        main_origs = np.copy(origs)
        img_centers = img_centers.cpu().numpy()
            
        '''
        POLYGON STUFF
        '''
        
        # poly_src_idx, poly_unused_idx, poly_tgt_idx, poly_ordered_est, poly_unused_est, poly_ordered_gt, poly_to_return = matched_poly
        if not val:
            poly_to_return = matched_poly
            
            
            
            my_src_poly = np.int64(poly_to_return[0][0].cpu().numpy())
            my_tgt_poly = np.int64(poly_to_return[0][1].cpu().numpy())
            
            est_poly_centers = np.squeeze(outputs['poly_centers'].detach().cpu().numpy())
            est_poly_one_hots = np.squeeze(outputs['poly_hamming'].sigmoid().detach().cpu().numpy()) > 0.5
            
            inter_poly_hamming = out['half_selected_polys'] 
        
       
        est_poly_map =np.zeros((196,200,3))
        
        inter_points = bezier.batch_interpolate(outputs['pred_boxes'].view(-1, 3, 2).detach().cpu().numpy(), 100)
        
        
        all_made_lines = []
        for k in range(len(inter_points)):
                
            cur_est = inter_points[k,...]
            cur_est = np.float32(my_line_maker(cur_est))/255
            all_made_lines.append(np.copy(cur_est)) 
            
            
        n_points = 200
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
            
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
            
        boundary_lines = []
        
        boundary_lines.append(np.float32(my_line_maker(upper))/255)
        boundary_lines.append(np.float32(my_line_maker(left))/255)
        boundary_lines.append(np.float32(my_line_maker(right))/255)
        boundary_lines.append(np.float32(my_line_maker(bev_left))/255)
        boundary_lines.append(np.float32(my_line_maker(bev_right))/255)
        
        all_made_lines = np.concatenate([np.stack(all_made_lines,axis=0),np.stack(boundary_lines,axis=0)],axis=0)
      
        origs = np.float32(main_origs > 0)
        # boundaries = np.stack([upper, left, right, bev_left, bev_right],axis=0)
        origs = np.concatenate([origs, np.stack(boundary_lines,axis=0)],axis=0)
            
        
        # gt_polys = np.squeeze(targets['poly_list'].cpu().numpy())
        gt_blob_mat = np.squeeze(targets['blob_mat'].cpu().numpy())
        gt_blob_ids = np.squeeze(targets['blob_ids'].cpu().numpy())
        gt_poly_centers = np.squeeze(targets['poly_centers'].cpu().numpy())
        gt_one_hot_polys = np.squeeze(targets['poly_one_hots'].cpu().numpy()) > 0.5
        
        gt_poly_map = np.zeros((196,200,3))
        
        cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(gt_blob_mat) + 1))
        
        colored_image = cm(np.int64(gt_blob_mat))
        
        Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'poly_all_gt.png' ))
        

        if not np.any(common_poly==None):
            common_poly_centers, common_poly_one_hots, common_blob_mat, common_blob_ids, common_real_hots = common_poly
            cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(common_blob_mat) + 1))
        
            colored_image = cm(np.int64(common_blob_mat))
            
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'common_polys.png' ))
 
                
        
        # logging.error('LEN ORIGS ' + str(len(origs)))
        
        # logging.error('GT POLY ' + str(gt_one_hot_polys.shape))
        # logging.error('POLY ORDERED GT ' + str(poly_ordered_gt.shape))
        
        # logging.error('LOOP START')
        # logging.error('MOD HAM ' + str(mod_ham.shape))
        if mod_ham != None:
            
            mod_ham = mod_ham.cpu().numpy()
            
            for k in range(len(mod_ham)):
                my_cur_est = mod_ham[my_tgt_poly[k]] > 0
                
                # logging.error('CUR EST')
                
                # logging.error('MY CUR EST ' + str(my_cur_est.shape))
                # logging.error('ALL MADE LINES ' + str(all_made_lines.shape))
                my_gt = np.clip(np.sum(all_made_lines[my_cur_est],axis=0),0,1)
                
                cur_gt = np.stack([my_gt, my_gt, my_gt],axis=-1)
                
                
                center_ar = np.zeros((196,200,3))
            
                center_ar[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),-1] = 1
                struct = ndimage.generate_binary_structure(2, 2)
           
                dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
                # center_ar[...,-1] = dilated
                
                
                # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
                cur_gt[dilated,-1] = 1
                cur_gt[dilated,:-1] = 0
                
                my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
                my_gt_img.save(os.path.join(save_path,'poly_mod_gt_'+str(k) +'.jpg' ))
            
                
        res_interpolated_list = out['interpolated_points']
        
       
            
        all_made_lines = []
        for k in range(len(res_interpolated_list)):
                
            cur_est = res_interpolated_list[k]
            cur_est = np.float32(my_line_maker(cur_est))/255
            all_made_lines.append(np.copy(cur_est)) 
        
        all_made_lines = np.stack(all_made_lines,axis=0)
      
        printed_estimateds = np.clip(np.sum(all_made_lines,axis=0),0,1)
        my_base_est = np.float32(np.copy(printed_estimateds) > 0)
        my_base_est = np.stack([my_base_est, my_base_est, my_base_est],axis=-1)
        
        all_made_lines = np.concatenate([all_made_lines,np.stack(boundary_lines,axis=0)],axis=0)
      
        
        # if not np.any(naive_matched_labels == None):
        #     for k in range(len(naive_matched_labels)):
        #         gt_ar = np.copy(all_made_lines[k])
        #         gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
        #         base = np.copy(gt_ar)

        #         for m in range(len(naive_matched_labels[0])):
        #             if naive_matched_labels[k,m] > 0:
        #                 temp = all_made_lines[m]
                         
        #                 # gt_ar[temp > 0.5,0] = 0.5 + 0.5*(m+1)/(len(last_gt_intersection_list[k]))
        #                 # gt_ar[temp > 0.5,1:] = 0
                        
        #                 temp_img = np.copy(base)
        #                 temp_img[temp > 0.5,0] = 1
        #                 temp_img[temp > 0.5,1:] = 0
                                
        #                 my_gt_img = Image.fromarray(np.uint8(temp_img*255))
        #                 my_gt_img.save(os.path.join(save_path,'naive_matched_'+str(k)+'_'+str(m) +'.jpg' ))
            
        if not val:
            for k in range(len(my_tgt_poly)):
        
                
                '''
                MATCHED GT
                '''
                
                cur_gt = np.float32(np.copy(img_centers) > 0)
                cur_gt[gt_blob_mat == gt_blob_ids[my_tgt_poly[k]]] = 0.5
                
                # logging.error('BLOB MAT')
                
                cur_gt = np.stack([cur_gt, cur_gt, cur_gt],axis=-1)
                
                my_cur_gt = gt_one_hot_polys[my_tgt_poly[k]]
                
                gt_border = np.sum(origs[my_cur_gt],axis=0)
                cur_gt[gt_border > 0 , 0] = 1
                cur_gt[gt_border > 0 , 1:] = 0
                
                
                
                center_ar = np.zeros((196,200,3))
                
                center_ar[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),-1] = 1
                struct = ndimage.generate_binary_structure(2, 2)
           
                dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct,iterations = 3)
                # center_ar[...,-1] = dilated
                
                
                # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
                cur_gt[dilated,-1] = 1
                cur_gt[dilated,:-1] = 0
                
                my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
                my_gt_img.save(os.path.join(save_path,'poly_matched_gt_'+str(k) +'.jpg' ))
                
                # logging.error('GT SAVED')
                '''
                EST
                '''
                
                my_est = np.copy(my_base_est)
                my_cur_est = inter_poly_hamming[my_src_poly[k]] > 0
                
                # logging.error('CUR EST')
                
                # logging.error('MY CUR EST ' + str(my_cur_est.shape))
                # logging.error('ALL MADE LINES ' + str(all_made_lines.shape))
                my_border = np.clip(np.sum(all_made_lines[my_cur_est],axis=0),0,1)
                my_est[my_border > 0 , 0] = 1
                my_est[my_border > 0 , 1:] = 0
                
                '''
                ESTIMATED CENTER
                '''
                
                center_ar = np.zeros((196,200,3))
                center_ar[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),-1] = 1
                struct = ndimage.generate_binary_structure(2, 2)
           
                dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct,iterations = 3)
                # my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),-1] = 1
                # my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),:] = 0
                
                my_est[dilated,:] = 0
                
                my_est[dilated,-1] = 1
                # '''
                # VORONOI
                # '''
                # center_ar = np.zeros((196,200,3))
                # center_ar[int(poly_voronoi[k,1]*195),int(poly_voronoi[k,0]*199),1] = 1
                # struct = ndimage.generate_binary_structure(2, 2)
           
                # dilated = ndimage.binary_dilation(center_ar[...,1]>0, structure=struct,iterations = 3)
                # # my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),-1] = 1
                # # my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),:] = 0
                
                # my_est[dilated,:] = 0
                
                # my_est[dilated,1] = 1
                
                # my_est_img = Image.fromarray(np.uint8(my_est*255))
                # my_est_img.save(os.path.join(save_path,'poly_matched_est_'+str(k) +'.jpg' ))
                
        
        est_assoc = inter_dict['assoc_est'] 
        est_coeffs = inter_dict['src_boxes'].detach().cpu().numpy()
        
        est_coef_endpoints = np.concatenate([est_coeffs[:,0],est_coeffs[:,-1]],axis=-1)
        for k in range(len(est_coeffs)):
                
            cur_est = inter_dict['interpolated'].detach().cpu().numpy()[k,...]
            cur_est = my_line_maker(cur_est)
            
            first_one = add_endpoints_to_line(cur_est,est_coef_endpoints[k])
            tot = np.clip(first_one,0,1)
            temp_img = Image.fromarray(np.uint8( tot*255))
            temp_img.save(os.path.join(save_path,'est_line_'+str(k)+'.jpg' ))
          
            for m in range(len(est_assoc[k])):
                if est_assoc[k][m] > 0:
                    first_one = add_endpoints_to_line(cur_est,est_coef_endpoints[k])
                    temp_est = inter_points[m,...]
                    temp_est = my_line_maker(temp_est)
                    second_one = add_endpoints_to_line(temp_est,est_coef_endpoints[m])
                    tot = np.clip(first_one + second_one,0,1)
                    temp_img = Image.fromarray(np.uint8( tot*255))
                    temp_img.save(os.path.join(save_path,'matched_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
            
    except Exception as e:
        logging.error('MATCHED RESULTS SAVE ')
        logging.error(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
    

def save_metric_visuals(out, targets, metric_visuals, config,save_path, rnn=False, train_targets_order=None, naive=False):
    try:
        origs = targets['origs'].cpu().numpy()
        main_origs = np.copy(origs)
       
        
        if ((not rnn) & (not naive)):
            detected_poly_probs = out['poly_probs']  
            detected_poly_centers = out['poly_centers']  
            unique_hamming = out['poly_hamming'] 
        # logging.error('POLY ORDERED EST  ' + str(poly_ordered_est.shape))
        
        # logging.error('POLY TO RETURN  ' + str(poly_to_return))
        
        last_gt_intersection_list, last_calc_gt_intersection_list, last_est_intersection_list, to_store_dist, last_sel_ar, last_comp_ar, last_gt_res_ar, last_calc_gt_res_ar,  hausdorff_gt, last_matched_gt_polys, last_matched_est_polys,\
            last_poly_match_gt_indices, last_poly_match_est_indices,all_rnn_to_compare_gt,all_rnn_to_compare_est,\
                rnn_gt_indices, rnn_est_indices, indeed_matched_list = metric_visuals
                
            
        
        res_interpolated_list = out['interpolated_points']
        
        
        
        all_sel_lines = []
        for k in range(len(last_sel_ar)):
                
            cur_est = last_sel_ar[k]
            cur_est = np.float32(my_line_maker(cur_est))/255
            all_sel_lines.append(np.copy(cur_est)) 
            
            
            
        all_made_lines = []
        for k in range(len(res_interpolated_list)):
                
            cur_est = res_interpolated_list[k]
            cur_est = np.float32(my_line_maker(cur_est))/255
            all_made_lines.append(np.copy(cur_est)) 
            
        all_gt_lines = []
        for k in range(len(hausdorff_gt)):
                
            cur_gt = hausdorff_gt[k]
            cur_gt = np.float32(my_line_maker(cur_gt))/255
            all_gt_lines.append(np.copy(cur_gt)) 
               
        indeed_matched_list = np.concatenate([np.array(indeed_matched_list),len(all_gt_lines)+np.arange(5)],axis=0)
        
        n_points = res_interpolated_list[0].shape[0]
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
            
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
            
        boundary_lines = []
        
        boundary_lines.append(np.float32(my_line_maker(upper))/255)
        boundary_lines.append(np.float32(my_line_maker(left))/255)
        boundary_lines.append(np.float32(my_line_maker(right))/255)
        boundary_lines.append(np.float32(my_line_maker(bev_left))/255)
        boundary_lines.append(np.float32(my_line_maker(bev_right))/255)
        
        all_made_lines = np.concatenate([np.stack(all_made_lines,axis=0),np.stack(boundary_lines,axis=0)],axis=0)
        
        all_gt_lines = np.concatenate([np.stack(all_gt_lines,axis=0),np.stack(boundary_lines,axis=0)],axis=0)
      
        origs = np.float32(main_origs > 0)
        # boundaries = np.stack([upper, left, right, bev_left, bev_right],axis=0)
        origs = np.concatenate([origs, np.stack(boundary_lines,axis=0)],axis=0)
            
        
        # gt_polys = np.squeeze(targets['poly_list'].cpu().numpy())
        gt_blob_mat = np.squeeze(targets['blob_mat'].cpu().numpy())
        gt_blob_ids = np.squeeze(targets['blob_ids'].cpu().numpy())
        gt_poly_centers = np.squeeze(targets['poly_centers'].cpu().numpy())
        gt_one_hot_polys = np.squeeze(targets['poly_one_hots'].cpu().numpy()) > 0.5
        
        gt_poly_map = np.zeros((196,200,3))
        
        # '''
        # POLY GT
        # '''
        
        # for k in range(len(last_comp_ar)):
        #     my_cur_est = last_comp_ar[k] > 0
        #     my_gt = np.clip(np.sum(all_made_lines[my_cur_est],axis=0),0,1)
            
        #     cur_gt = np.stack([my_gt, my_gt, my_gt],axis=-1)
            
            
        #     center_ar = np.zeros((196,200,3))
        
        
        #     # logging.error('GT CENTER METRIC : ' + str(int(gt_poly_centers[k][1]*195)))
            
        #     center_ar[int(np.clip(0,gt_poly_centers[k][1]*195,1)),int(np.clip(0,gt_poly_centers[k][0]*199,1)),-1] = 1
        #     struct = ndimage.generate_binary_structure(2, 2)
       
        #     dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
        #     # center_ar[...,-1] = dilated
            
            
        #     # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
        #     cur_gt[dilated,-1] = 1
        #     cur_gt[dilated,:-1] = 0
            
        #     my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
        #     my_gt_img.save(os.path.join(save_path,'confusion_poly_gt_'+str(k) +'.jpg' ))
        '''
        POLY GT
        '''
        
        # for k in range(len(all_gt_lines)):
        #     my_gt_img = Image.fromarray(np.uint8(all_gt_lines[k]*255))
        #     my_gt_img.save(os.path.join(save_path,'gt_line_'+str(k) +'.jpg' ))
            
        #     my_gt_img = Image.fromarray(np.uint8(origs[k]*255))
        #     my_gt_img.save(os.path.join(save_path,'orig_'+str(k) +'.jpg' ))
            
        
        if ((not rnn) & (not naive)):
            for k in range(len(last_matched_gt_polys)):
                my_cur_est = last_matched_gt_polys[k] > 0
                my_gt = np.clip(np.sum(all_made_lines[my_cur_est],axis=0),0,1)
                
                cur_gt = np.stack([my_gt, my_gt, my_gt],axis=-1)
                
                
                center_ar = np.zeros((196,200,3))
            
            
                # logging.error('GT CENTER METRIC : ' + str(int(gt_poly_centers[k][1]*195)))
                
                center_ar[int(np.clip(0,gt_poly_centers[last_poly_match_gt_indices[k]][1]*195,1)),int(np.clip(0,gt_poly_centers[last_poly_match_gt_indices[k]][0]*199,1)),-1] = 1
                struct = ndimage.generate_binary_structure(2, 2)
           
                dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
                # center_ar[...,-1] = dilated
                
                
                # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
                cur_gt[dilated,-1] = 1
                cur_gt[dilated,:-1] = 0
                
                my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
                my_gt_img.save(os.path.join(save_path,'confusion_poly_gt_'+str(k) +'.jpg' ))
            # logging.error('SAVED GT METRIC')
            '''
            POLY EST
            '''
            for k in range(len(last_matched_est_polys )):
                my_cur_est = last_matched_est_polys[k] > 0
                my_gt = np.clip(np.sum(all_made_lines[my_cur_est],axis=0),0,1)
                
                cur_gt = np.stack([my_gt, my_gt, my_gt],axis=-1)
                
                
                center_ar = np.zeros((196,200,3))
                
                
                
                center_ar[int(detected_poly_centers[last_poly_match_est_indices[k]][1]*195),int(detected_poly_centers[last_poly_match_est_indices[k]][0]*199),-1] = 1
                struct = ndimage.generate_binary_structure(2, 2)
           
                dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
                # center_ar[...,-1] = dilated
                
                
                # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
                cur_gt[dilated,-1] = 1
                cur_gt[dilated,:-1] = 0
                
                my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
                my_gt_img.save(os.path.join(save_path,'confusion_poly_est_'+str(k) +'.jpg' )) 
                

            
        if rnn:
            gt_orders = targets['gt_order_labels']
#            logging.error('GT ORDER LABELS FROM DATASET ' + str(gt_orders.shape))
            for k in range(len(gt_orders)):
                gt_ar = np.copy(all_gt_lines[k])
                gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
                base = np.copy(gt_ar)
                for m in range(len(gt_orders[k])):
                    if gt_orders[k][m] > -1:
                        temp = all_gt_lines[gt_orders[k][m]]
                 
                        temp_img = np.copy(base)
                        temp_img[temp > 0.5,0] = 1
                        temp_img[temp > 0.5,1:] = 0
                        
                        temp_img = Image.fromarray(np.uint8(temp_img*255))  
                        temp_img.save(os.path.join(save_path,'gt_orders_'+str(k) +'_' + str(m)+'.jpg' ))
                    
     
        
            rnn_gt_indices = np.concatenate([rnn_gt_indices, len(rnn_gt_indices)+ np.arange(5)],axis=0)
            rnn_est_indices = np.concatenate([rnn_est_indices, len(rnn_est_indices)+ np.arange(5)],axis=0)
            
#            logging.error('ALL RNN TO COMPARE GT ' + str(len(all_rnn_to_compare_gt)))
#            
#            logging.error('CONCAT RNN GT INDICES ' + str(rnn_gt_indices.shape))
#            
#            logging.error('CONCAT RNN EST INDICES ' + str(rnn_est_indices.shape))
            
#            logging.error('METRIC RINN INDICES ' + str(rnn_est_indices))
            
            for k in range(len(all_rnn_to_compare_gt)):
        
                '''
                RNN GT
                '''
                cur_index = rnn_gt_indices[k]
                gt_ar = np.copy(all_gt_lines[cur_index])
                gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
                base = np.copy(gt_ar)
                for m in range(len(all_rnn_to_compare_gt[k])):
                    
                    if all_rnn_to_compare_gt[k][m] >= len(all_gt_lines):
                        
                        temp = all_gt_lines[all_rnn_to_compare_gt[k][m]-len(all_gt_lines)]
                    else:
                    
                        temp = all_gt_lines[rnn_gt_indices[all_rnn_to_compare_gt[k][m]]]
             
                    temp_img = np.copy(base)
#                    logging.error('BASE ' + str(base.shape))
#                    logging.error('all_rnn_to_compare_gt[k][m] ' + str(all_rnn_to_compare_gt[k][m]))
#                    logging.error('rnn_gt_indices[all_rnn_to_compare_gt[k][m]] ' + str(rnn_gt_indices[all_rnn_to_compare_gt[k][m]]))
                    
                    temp_img[temp > 0.5,0] = 1
                    temp_img[temp > 0.5,1:] = 0
                    
                    temp_img = Image.fromarray(np.uint8(temp_img*255))  
                    temp_img.save(os.path.join(save_path,'confusion_rnn_gt_inter_'+str(k) +'_' + str(m)+'.jpg' ))
                    
         
                '''
                RNN
                '''
                
                
                
                cur_index = rnn_est_indices[k]
                gt_ar = np.copy(all_made_lines[cur_index])
                gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
                base = np.copy(gt_ar)
                for m in range(len(all_rnn_to_compare_est[k])):
#                    logging.error('all_rnn_to_compare_est[k] ' + str(all_rnn_to_compare_est[k]))
#                    logging.error('rnn_gt_indices[all_rnn_to_compare_gt[k][m]] ' + str(rnn_gt_indices[all_rnn_to_compare_gt[k][m]]))
                    
                    tind = rnn_est_indices[all_rnn_to_compare_est[k][m]]
#                    logging.error('TIND ' + str(tind))
                    
                    temp = all_made_lines[tind]
             
                    temp_img = np.copy(base)
                    temp_img[temp > 0.5,0] = 1
                    temp_img[temp > 0.5,1:] = 0
                    
                    temp_img = Image.fromarray(np.uint8(temp_img*255))  
                    temp_img.save(os.path.join(save_path,'confusion_rnn_est_inter_'+str(k) +'_' + str(m)+'.jpg' ))
                    
            
            
        # '''
        # POLY EST
        # '''
        # for k in range(len(unique_hamming )):
        #     my_cur_est = unique_hamming[k] > 0
        #     my_gt = np.clip(np.sum(all_made_lines[my_cur_est],axis=0),0,1)
            
        #     cur_gt = np.stack([my_gt, my_gt, my_gt],axis=-1)
            
            
        #     center_ar = np.zeros((196,200,3))
            
            
            
        #     center_ar[int(detected_poly_centers[k][1]*195),int(detected_poly_centers[k][0]*199),-1] = 1
        #     struct = ndimage.generate_binary_structure(2, 2)
       
        #     dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
        #     # center_ar[...,-1] = dilated
            
            
        #     # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
        #     cur_gt[dilated,-1] = 1
        #     cur_gt[dilated,:-1] = 0
            
        #     my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
        #     my_gt_img.save(os.path.join(save_path,'confusion_poly_est_'+str(k) +'.jpg' ))
        
                
        '''
        ORDER GT
        '''
        
        for k in range(len(last_gt_intersection_list)):
    
            
            '''
            MATCHED GT
            '''
            
            
            gt_ar = np.copy(all_gt_lines[k])
            gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
            base = np.copy(gt_ar)
            for m in range(len(last_gt_intersection_list[k])):
                for n in range(len(last_gt_intersection_list[k][m])):
                    temp = all_gt_lines[last_gt_intersection_list[k][m][n]]
                     
                    # gt_ar[temp > 0.5,0] = 0.5 + 0.5*(m+1)/(len(last_gt_intersection_list[k]))
                    # gt_ar[temp > 0.5,1:] = 0
                    
                    temp_img = np.copy(base)
                    temp_img[temp > 0.5,0] = 1
                    temp_img[temp > 0.5,1:] = 0
                    
                    temp_img = Image.fromarray(np.uint8(temp_img*255))  
                    temp_img.save(os.path.join(save_path,'order_gt_inter_'+str(k) +'_' + str(m)+'_' + str(n)+'.jpg' ))
                    
            # my_gt_img = Image.fromarray(np.uint8(gt_ar*255))  
            # my_gt_img.save(os.path.join(save_path,'confusion_order_gt_'+str(k) +'.jpg' ))
            
            # logging.error('GT SAVED')
            '''
            EST
            '''
        for k in range(len(last_calc_gt_intersection_list)):  
            
            gt_ar = np.copy(all_gt_lines[indeed_matched_list[k]])
            gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
            base = np.copy(gt_ar)
            for m in range(len(last_calc_gt_intersection_list[k])):
                for n in range(len(last_calc_gt_intersection_list[k][m])):
                    temp = all_gt_lines[indeed_matched_list[last_calc_gt_intersection_list[k][m][n]]]
                     
                    # gt_ar[temp > 0.5,0] = 0.5 + 0.5*(m+1)/(len(last_gt_intersection_list[k]))
                    # gt_ar[temp > 0.5,1:] = 0
                    
                    temp_img = np.copy(base)
                    temp_img[temp > 0.5,0] = 1
                    temp_img[temp > 0.5,1:] = 0
                    
                    temp_img = Image.fromarray(np.uint8(temp_img*255))  
                    temp_img.save(os.path.join(save_path,'confusion_order_gt_inter_'+str(k) +'_' + str(m)+'_' + str(n)+'.jpg' ))
                    
                    
                    
            gt_ar = np.copy(all_sel_lines[k])
            gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
            base = np.copy(gt_ar)
            for m in range(len(last_est_intersection_list[k])):
                for n in range(len(last_est_intersection_list[k][m])):
                    temp = all_sel_lines[last_est_intersection_list[k][m][n]]
                    # gt_ar[temp > 0.5,0] = 0.5 + 0.5*(m+1)/(len(last_est_intersection_list[k]))
                    # gt_ar[temp > 0.5,1:] = 0
                    
                    
                    temp_img = np.copy(base)
                    temp_img[temp > 0.5,0] = 1
                    temp_img[temp > 0.5,1:] = 0
                    
                    temp_img = Image.fromarray(np.uint8(temp_img*255))  
                    temp_img.save(os.path.join(save_path,'confusion_order_est_inter_'+str(k) +'_' + str(m)+'_' + str(n)+'.jpg' ))
                    
            # my_gt_img = Image.fromarray(np.uint8(gt_ar*255))  
            # my_gt_img.save(os.path.join(save_path,'confusion_order_est_'+str(k) +'.jpg' ))
        
    except Exception as e:
        logging.error('REAL METRIC SAVE ')
        logging.error(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
    
        
def save_naive_results(out, naive_matched_labels, save_path):
    try:
        n_points = 100
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
            
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
            
        
        boundary_lines = []
        
        boundary_lines.append(np.float32(my_line_maker(upper))/255)
        boundary_lines.append(np.float32(my_line_maker(left))/255)
        boundary_lines.append(np.float32(my_line_maker(right))/255)
        boundary_lines.append(np.float32(my_line_maker(bev_left))/255)
        boundary_lines.append(np.float32(my_line_maker(bev_right))/255)
        
        res_interpolated_list = out['interpolated'].detach().cpu().numpy()
            
        logging.error('NAIVE SAVE '+ str(res_interpolated_list.shape))
            
        all_made_lines = []
        for k in range(len(res_interpolated_list)):
                
            cur_est = res_interpolated_list[k]
            cur_est = np.float32(my_line_maker(cur_est))/255
            all_made_lines.append(np.copy(cur_est)) 
        
        all_made_lines = np.stack(all_made_lines,axis=0)
      
        printed_estimateds = np.clip(np.sum(all_made_lines,axis=0),0,1)
        my_base_est = np.float32(np.copy(printed_estimateds) > 0)
        my_base_est = np.stack([my_base_est, my_base_est, my_base_est],axis=-1)
        
        all_made_lines = np.concatenate([all_made_lines,np.stack(boundary_lines,axis=0)],axis=0)
      
        
        if not np.any(naive_matched_labels == None):
            for k in range(len(naive_matched_labels)):
                gt_ar = np.copy(all_made_lines[k])
                gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
                base = np.copy(gt_ar)
    
                for m in range(len(naive_matched_labels[0])):
                    if naive_matched_labels[k,m] > 0:
                        temp = all_made_lines[m]
                         
                        # gt_ar[temp > 0.5,0] = 0.5 + 0.5*(m+1)/(len(last_gt_intersection_list[k]))
                        # gt_ar[temp > 0.5,1:] = 0
                        
                        temp_img = np.copy(base)
                        temp_img[temp > 0.5,0] = 1
                        temp_img[temp > 0.5,1:] = 0
                                
                        my_gt_img = Image.fromarray(np.uint8(temp_img*255))
                        my_gt_img.save(os.path.join(save_path,'naive_matched_'+str(k)+'_'+str(m) +'.jpg' ))
            
    except Exception as e:
        logging.error('NAIVE SAVE ')
        logging.error(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
        
        
def save_results_train(image, raw_out, out,matched_poly, targets, static_inter_dict, object_inter_dict, static_target_ids, object_target_ids, config,
                       gt_poly_indices=None, mod_ham=None, metric_visuals=[],train_targets_order=None, train_rnn_matched_labels=None, match_static_indices=None, rnn=False, rnn_gt=True,
                       naive_matched_labels=None, naive=False, poly_voronoi=None):
    
    image = process_image(image)
    
    os.makedirs(os.path.join(config.save_logdir,'train_images'),exist_ok=True)
    fileList = glob.glob(os.path.join(config.save_logdir,'train_images','*'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    # logging.error('LEN OF POST PROCESS ' + str(len(out)))
    
    for fr in range(len(image)):
        cur_img = Image.fromarray(np.uint8(image[fr,...]))
        cur_img.save(os.path.join(config.save_logdir,'train_images','image_'+str(fr)+'.jpg'))       
    
    if ((not rnn) & (not naive)):
        try:  
          
            save_matched_results(static_inter_dict, out, raw_out, naive_matched_labels, mod_ham, poly_voronoi, gt_poly_indices, targets[0],static_target_ids,config,os.path.join(config.save_logdir,'train_images'))
            # else:
                
        except Exception as e:
            logging.error("PROBLEM IN MATCHED TRAIN SAVE: " + str(e))
    # if naive:
    try:  
        # if pretrain_poly_indices == None:
        save_naive_results(static_inter_dict, naive_matched_labels,os.path.join(config.save_logdir,'train_images'))
        # else:
            
    except Exception as e:
        logging.error("PROBLEM IN NAIVE TRAIN SAVE: " + str(e))
        
        
    if rnn:
        save_train_order_rnn( raw_out,targets[0], train_targets_order , match_static_indices, config,os.path.join(config.save_logdir,'train_images'), train_rnn_matched_labels=train_rnn_matched_labels, gt=rnn_gt)
    
    try:  
        # if pretrain_poly_indices == None:
        save_metric_visuals(out, targets[0], metric_visuals, config,os.path.join(config.save_logdir,'train_images'), rnn=rnn, train_targets_order=train_targets_order, naive=naive)
        # else:
            
    except Exception as e:
        logging.error("PROBLEM IN METRIC VISUAL SAVE: " + str(e))
        
    try:  
        visual_masks_gt(np.uint8(image),targets,os.path.join(config.save_logdir,'train_images'))
    except Exception as e:
        logging.error("PROBLEM IN VISUAL GT TRAIN SAVE: " + str(e))

    try:
        
        visual_est(np.uint8(image),[out], matched_poly, raw_out, os.path.join(config.save_logdir,'train_images'))
    
    except Exception as e:
        logging.error("PROBLEM IN VISUAL EST TRAIN SAVE: " + str(e))
        
    
      
def save_results_eval(image, raw_out, out,matched_poly, targets, static_inter_dict, object_inter_dict, static_target_ids, object_target_ids, config,
                        gt_poly_indices=None, mod_ham=None, metric_visuals=[], rnn=False, common_poly=None):
#    
    image = process_image(image)
    
    base_path = os.path.join(config.save_logdir,'val_images',targets[0]['scene_name'],targets[0]['sample_token'])
#    
    os.makedirs(base_path,exist_ok=True)
    fileList = glob.glob(os.path.join(base_path,'*'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    # logging.error('LEN OF POST PROCESS ' + str(len(out)))
    
    # for fr in range(len(image)):
    #     cur_img = Image.fromarray(np.uint8(image[fr,...]))
    #     cur_img.save(os.path.join(base_path,'image.jpg'))       
    
    # if targets[0]['obj_exists']:
    #     try:
    #         save_matched_objects(object_inter_dict,targets[0],object_target_ids,config,base_path)
    #     except Exception as e:
    #         logging.error("PROBLEM IN MATCHED OBJECT VAL SAVE: " + str(e))
    
    # save_matched_results(static_inter_dict,targets[0],static_target_ids,config,base_path)
    # out = get_selected_estimates(out, thresh = 0.5)
    # try:  
    #     save_matched_results(static_inter_dict, raw_out, matched_poly, targets[0],static_target_ids,config,base_path)
    # try:  
    #     # if pretrain_poly_indices == None:
    #     save_matched_results(static_inter_dict, raw_out, mod_ham, gt_poly_indices, targets[0],static_target_ids,config,base_path)
    #     # else:
            
    # except Exception as e:
    #     logging.error("PROBLEM IN MATCHED EVAL SAVE: " + str(e))
        
    
    # try:  
    #     # if pretrain_poly_indices == None:
    #     save_metric_visuals(out, targets[0], metric_visuals, config,base_path)
    #     # else:
            
    # except Exception as e:
    #     logging.error("PROBLEM IN METRIC VISUAL SAVE: " + str(e))
        
        
    # try:  
    #     # if pretrain_poly_indices == None:
    #     save_metric_visuals(out, targets[0], metric_visuals, config, base_path, rnn=rnn)
    #     # else:
            
    # except Exception as e:
    #     logging.error("PROBLEM IN EVAL METRIC SAVE : " + str(e))
        
    # if not rnn:
#    try:  
#        # if pretrain_poly_indices == None:
#            save_matched_results(static_inter_dict, out, raw_out, None,None,None,None, targets[0],static_target_ids,config, base_path,val=True, common_poly = common_poly)
#        # else:
#            
#    except Exception as e:
#        logging.error("PROBLEM IN MATCHED TRAIN SAVE: " + str(e))
  
    
    try:
        visual_masks_gt(np.uint8(image),targets,base_path,name='_')
    except Exception as e:
        logging.error("PROBLEM IN VISUAL MASKS GT VAL SAVE: " + str(e))
        
    try:
        
        visual_est(np.uint8(image),out, matched_poly, raw_out, base_path,name='_')
    
    except Exception as e:
        logging.error("PROBLEM IN VISUAL EST VAL SAVE: " + str(e))
        
    
def img_saver(img,path):
    img = Image.fromarray(np.uint8(img))
    img.save(path)
    
    
    
def pinet_save_results_eval(image,out,coefs_list,boundaries_list,targets, poly_stuff,config):
#    
    image = process_image(image)
    
    base_path = os.path.join(config.save_logdir,'val_images',targets[0]['scene_name'],targets[0]['sample_token'])
#    
    save_path = base_path
    name = '_'
    os.makedirs(base_path,exist_ok=True)
    fileList = glob.glob(os.path.join(base_path,'*'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    # logging.error('LEN OF POST PROCESS ' + str(len(out)))
    
#    for fr in range(len(image)):
#        cur_img = Image.fromarray(np.uint8(image[fr,...]))
#        cur_img.save(os.path.join(base_path,'image.jpg'))       
    
  
    try:
        visual_masks_gt(np.uint8(image),targets,base_path,name='_')
    except Exception as e:
        logging.error("PROBLEM IN VISUAL MASKS GT VAL SAVE: " + str(e))

    
    res_img = out[-1]

    res = Image.fromarray(res_img[0][...,[2,1,0]])
    res.save(os.path.join(base_path,'est.jpg'))
    
    
    
    res_interpolated_list = []
    res_coef_list = []
    coef_all_roads = np.zeros((196,200))
    temp_dict = dict()
    for k in range(len(coefs_list)):
                

        control = coefs_list[k]
        
        coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        
        interpolated = bezier.interpolate_bezier(control,100)
        
        res_interpolated_list.append(np.copy(interpolated))
#        
#        line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
        line2 = my_line_maker(interpolated,size=(196,200))
        res_coef_list.append(line2)
        coef_all_roads = coef_all_roads + np.float32(line2)
    
    b=0
    temp_dict['coef_lines'] = res_coef_list
    
    temp_dict['interpolated_points'] = res_interpolated_list
    
    all_lanes = np.zeros((196,200))
    for k in range(len(res_coef_list)):
                
        lane_poly = convert_line_to_lane(coefs_list[k], lane_width = 3.5)
        can = np.zeros((196,200))

        render_polygon(can, lane_poly, shape=(196,200), value=1)
        
        all_lanes = all_lanes + can
        
        res_lane = Image.fromarray(np.uint8(255*can))
        
        res_coef = Image.fromarray(res_coef_list[k])
        if name==None:
            
            res_coef.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_interp_road_'+str(k)+'.jpg'))
            res_lane.save(os.path.join(save_path,'batch_'+str(b) + '_est_lane_'+str(k)+'.jpg'))
        
        else:
            
            res_coef.save(os.path.join(save_path,name + '_est_coef_interp_road_'+str(k)+'.jpg'))
            res_lane.save(os.path.join(save_path,name + '_est_lane_'+str(k)+'.jpg'))
        



    all_lanes = np.uint8(np.clip(all_lanes,0,1)*255)
    if name==None:
        
        temp_img = Image.fromarray(all_lanes)
        temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_all_lanes.jpg' ))       
        

        coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
        temp_img = Image.fromarray(coef_all_roads)
        temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_all_roads.jpg' ))       
    else:
        temp_img = Image.fromarray(all_lanes)
        temp_img.save(os.path.join(save_path,name + '_est_all_lanes.jpg' ))    
      
        coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
        temp_img = Image.fromarray(coef_all_roads)
        temp_img.save(os.path.join(save_path,name + '_est_coef_all_roads.jpg' ))    



    
    common_poly_centers, common_poly_one_hots, common_blob_mat, common_blob_ids, common_real_hots = poly_stuff
    if not np.any(common_blob_mat==None):
        cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(common_blob_mat) + 1))
    
        colored_image = cm(np.int64(common_blob_mat))
        
        Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'common_polys.png' ))
         
        
# cur_x = np.array([0.4,0.5,0.6])

# cur_y = np.array([0.9,0.8,0.7])
# calib = np.array([[1260,0,800],[0,1260,600],[0,0,1]])



def get_spline_for_pinet(x,y, calib, targets, argo=False):
    num_boundaries = len(x)
    
    
    if argo:
        calib[0] *= 1920/800
        calib[1] *= 1200/448
        img_width = 1920
        img_height = 1200
    else:
        calib[0] *= 1600/800
        calib[1] *= 900/448
        img_width = 1600
        img_height = 900
    
    
    logging.error('CALIB ' + str(calib))
    logging.error('IMG ' + str((img_height, img_width)))
    
    spline_list = []
    points_list = []
    for k in range(num_boundaries):
        
        yx = zip(list(np.array(y[k])/256), list(np.array(x[k])/512))
        
        yx = list(yx)
        
        yx = sorted(yx, key=lambda t: t[0])
        x_sorted = [x for y, x in yx]
        y_sorted = [y for y, x in yx]
        
        cur_x = np.array(x_sorted)
        cur_y = np.array(y_sorted)
        
#        cur_x = np.array(x[k])/512
#        cur_y = np.array(y[k])/256
        # logging.error('CUR X ')
        # logging.error(str(cur_x))
        
        # logging.error('CUR Y ')
        # logging.error(str(cur_y))
        
        cam_height = 1.7
        # z_dist = 2.5
        
        f = calib[0,0]
        y_center = calib[1,-1]
        z = cam_height*f/abs(cur_y*img_height - y_center)
        real_x = (z*img_width*cur_x - calib[0,-1]*z)/f
        
        real_x = (real_x + 25)/50
        
        z = 1 - (z - 2.5)/50
        
        logging.error('MAPPED X ')
        logging.error(str(real_x))
        logging.error('MAPPED Z ')
        logging.error(str(z))
                
        invalid = (z > 1) | (z < 0) | (real_x > 1) | (real_x < 0)
        
        valid = np.logical_not(invalid)
        
        if np.sum(valid) < 5:
            continue
        
        real_x = real_x[valid]
        z = z[valid]
        
        
        points = np.stack([real_x, z],axis=-1)
        
        
        
        points_list.append(points)
        res = bezier.fit_bezier(points, 3)[0]  
        
        spline_list.append(res)
        
    center_lines = []
    res_interpolated_list=[]
    out=dict()
    for boun in range(len(points_list)-1):
        
        cur_boun = np.copy(points_list[boun])
        cur_dists = []
        dist_mat_list = []
        for other in range(boun + 1,len(points_list)):
            
            to_comp = np.copy(points_list[other])
            
            dist_mat = cdist(cur_boun, to_comp,'euclidean')
            
            dist_mat_list.append(np.copy(dist_mat))
            
            dist = np.min(dist_mat,axis=-1)
            cur_dists.append(np.copy(dist))
            
        dist_ar = np.stack(cur_dists,axis=0)
        mean_dist = np.mean(dist_ar,axis=-1)
        
        selected_pair = np.argmin(mean_dist)
        real_id = np.arange(boun + 1,len(points_list))[selected_pair]
        
        pair = points_list[real_id]
        
        my_dist = dist_mat_list[selected_pair]
        
        pointwise_min = np.argmin(my_dist,axis=-1)
        
        other_points = pair[pointwise_min]
        
        centerline = (other_points + cur_boun)/2
        
        yx = np.array(sorted(list(centerline), key=lambda t: t[1]))
#        x_sorted = [x for x, y in yx]
#        y_sorted = [y for x, y in yx]
        
#        logging.error(str(yx))
        res = bezier.fit_bezier(yx, 3)[0]  
        interpolated = bezier.interpolate_bezier(res,100)
        res_interpolated_list.append(np.copy(interpolated))
        center_lines.append(res)
    
    
    # logging.error('PINET CENTER LINES '+str(len(center_lines)))
    
    if len(center_lines) > 0:
    
        coefs = np.stack(center_lines,axis=0)
        out['boxes'] = coefs
        
    
                
        out['interpolated_points'] = res_interpolated_list
        
        loss = cdist(np.reshape(coefs,(-1,6)), targets['control_points'].cpu().numpy())
        
        i,j=linear_sum_assignment(loss)        
            
        out['src_boxes'] = coefs[i]
        out['target_ids'] = (0,j)
        out['src_ids'] = (0,i)
    
    else:
        out['boxes'] = []
        
    
                
        out['interpolated_points'] = []
        
        
        out['src_boxes'] = []
        out['target_ids'] = 0
        out['src_ids'] = 0
    
        
    return center_lines,spline_list, out
    