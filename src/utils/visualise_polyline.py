import matplotlib
matplotlib.use('Agg') 
from matplotlib.cm import get_cmap
import numpy as np
import torch
import logging
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.colors as colors

import scipy.ndimage as ndimage
from src.utils import bezier
import cv2 
#import networkx as nx

from scipy.spatial.distance import cdist, directed_hausdorff
from baseline.Utils.utils import class_to_xy
from scipy.optimize import linear_sum_assignment

from skimage import measure
import sys
image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]

DETECTION_COLOR_DICT = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    # 'traffic_cone': 'C8',
                    # 'barrier': 'C9'
                    }

TEMP_COLOR_LIST = ['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']


COLOR_LIST = [np.array(colors.to_rgb(k)) for k in TEMP_COLOR_LIST]

COLOR_LIST.append(np.array([1,1,1]))

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   ]

# 'traffic_cone', 'barrier'

def convert_line_to_lane(coeffs, lane_width = 3.5):
    
    resolution = 0.25
    patch_size = (196,200)
    
    one_side = lane_width/2/resolution
    
    one_side_x = one_side/patch_size[1]
    one_side_y = one_side/patch_size[0]
    
    segments = len(coeffs) - 1
    
    new_coeffs_list1 = []
    new_coeffs_list2 = []
    
    for seg in range(segments):
        slope = (coeffs[seg+1,1] - coeffs[seg,1])/(coeffs[seg+1,0] - coeffs[seg,0] + 0.000001)
        
        inv_slope = -1/slope
        
        unit_vec_x = np.sqrt(1/(inv_slope**2 + 1))
        unit_vec_y = np.sqrt(1-unit_vec_x**2)*one_side_y
        unit_vec_x = unit_vec_x *one_side_x
        new_coeffs_list1.append(np.array([coeffs[seg,0] + unit_vec_x,coeffs[seg,1] + unit_vec_y]))
        new_coeffs_list1.append(np.array([coeffs[seg+1,0] + unit_vec_x,coeffs[seg+1,1] + unit_vec_y]))

        new_coeffs_list2.append(np.array([coeffs[seg,0] - unit_vec_x,coeffs[seg,1] - unit_vec_y]))
        new_coeffs_list2.append(np.array([coeffs[seg+1,0] - unit_vec_x,coeffs[seg+1,1] - unit_vec_y]))

    
    new_coeffs_list2_flipped = new_coeffs_list2[::-1]
    
    all_coeffs = new_coeffs_list1 + new_coeffs_list2_flipped
    all_coeffs = np.array(all_coeffs)
    
    return all_coeffs
    
    
    
# pt1 = np.array([0.5,0])
# pt2 = np.array([0.5,0.2])
# pt3 = np.array([0.5,0.4])
# coeffs = np.stack([pt1, pt2, pt3],axis=0)
    
    

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



def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def overlay_semantic_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img


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


def merged_hausdorff_match(out, target):
    
    # res_coef_list = out['interpolated_points']
    est_coefs = out['merged_coeffs']
    
    # est_coefs = np.reshape(est_coefs,(est_coefs.shape[0],-1))
    
    orig_coefs = target['control_points'].cpu().numpy()
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
    interpolated_origs = []
    
    for k in range(len(orig_coefs)):
        inter = bezier.interpolate_bezier(orig_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
    
    if len(est_coefs) == 0:
        return None,None, interpolated_origs
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    
    ind = np.argmin(dist_mat, axis=-1)
    min_vals = np.min(dist_mat,axis=-1)
    
  
        
        
    return min_vals, ind, interpolated_origs 
    
def hausdorff_match(out, target):
    
    # res_coef_list = out['interpolated_points']
    est_coefs = out['boxes']
#    logging.error('HAUS EST ' + str(est_coefs.shape))
    orig_coefs = np.copy(target['control_points'].cpu().numpy())
    
    
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
#    logging.error('HAUS EST ORIG ' + str(orig_coefs.shape))
    
    interpolated_origs = []
    
    for k in range(len(orig_coefs)):
        inter = bezier.interpolate_bezier(orig_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
    
    if len(est_coefs) == 0:
        out['src_boxes'] = est_coefs
        out['target_ids'] = (0,0)
        out['src_ids'] = (0,0)
        return None,None, interpolated_origs, out
    
    
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    # second_dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs[:,::-1,:],axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    # dist_mat = np.min(np.stack([dist_mat,second_dist_mat],axis=0),axis=0)
    ind = np.argmin(dist_mat, axis=-1)
    min_vals = np.min(dist_mat,axis=-1)
    
    
#    logging.error('HAUS CDIST TARGET ' + str(target['control_points'].cpu().numpy().shape))
    
    loss = cdist(np.reshape(est_coefs,(len(est_coefs),-1)), target['control_points'].cpu().numpy())
        
    i,j=linear_sum_assignment(loss)        
        
    out['src_boxes'] = est_coefs[i]
    out['target_ids'] = (0,j)
    out['src_ids'] = (0,i)

#    out['src_boxes'] = est_coefs
#    out['target_ids'] = (0,np.arange(len(orig_coefs)))
#    out['src_ids'] = (0,np.arange(len(orig_coefs)))
        
        
    return min_vals, ind, interpolated_origs , out  
    

def get_merged_coeffs(targets):
    

#    scores = targets['scores']
#    labels = targets['labels']
    coeffs = targets['boxes']
    
#    res_list = targets['lines'] 
#    res_coef_list = targets['coef_lines'] 
#    all_roads = targets['all_roads'] 
#    coef_all_roads = targets['coef_all_roads'] 
    assoc = targets['assoc'] 
    # start_assoc = targets['start'] 
    # fin_assoc = targets['fin'] 
           
    diag_mask = np.eye(len(assoc))
    
    # start_assoc = np.clip(start_assoc + diag_mask,0,1)
    
    # fin_assoc = np.clip(fin_assoc + diag_mask,0,1)
    
    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask
    
    corrected_coeffs = np.copy(coeffs)
    
    # gat_start = gather_all_ends(start_assoc)
    # gat_fin = gather_all_ends(fin_assoc)
    
    # '''
    # HANDLE STARTS
    # '''
    # for k in range(len(gat_start)):
    #     all_points=[]
        
    #     for m in gat_start[k]:
    #         all_points.append(corrected_coeffs[m,0])
            
        
    #     av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
    #     for m in gat_start[k]:
    #         corrected_coeffs[m,0] = av_p
            
    
    # for k in range(len(gat_fin)):
    #     all_points=[]
        
    #     for m in gat_fin[k]:
    #         all_points.append(corrected_coeffs[m,-1])
            
        
    #     av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
    #     for m in gat_fin[k]:
    #         corrected_coeffs[m,-1] = av_p
            
        
    
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

 
def get_polygons(targets, thresh):
        
        res = []
        b=0
        
        temp_dict = dict()
        
        # scores = targets[b]['scores'].detach().cpu().numpy()
       
        
        # labels = targets[b]['labels'].detach().cpu().numpy()
        coeffs = targets['boxes']
        # endpoints = targets[b]['endpoints'].detach().cpu().numpy()
        
        assoc = targets['assoc']

        # logging.error('ASSOC IN GET SELECTED ' + str(assoc.shape))
        
        all_ind = np.arange(len(coeffs))
        
        detected_coeffs = coeffs
        # detected_endpoints = endpoints[selecteds,...]
        
#        detected_con_matrix = assoc[selecteds]
#        detected_con_matrix = detected_con_matrix[:,selecteds]
        
        
        # all_roads = np.zeros((196,200,3),np.float32)
        # coef_all_roads = np.zeros((196,200,3),np.float32)
        if len(detected_coeffs) > 0:
            
            
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
        
def get_selected_polylines(targets, thresh = 0.5, poly_thresh=0.5,poly_hamming_thresh=0.5, training=True):
    
    temp_dict = dict()
    
    
    # probs = np.squeeze(targets['init_point_detection_softmaxed'].detach().cpu().numpy(), axis=0)
#    int_point_heatmap = targets['pred_init_point_softmaxed'].detach().cpu().numpy()
    
    if 'pred_polys' in targets:
        poly_locs = targets['pred_polys'].detach().cpu().numpy()
               
        # logging.error('PRED POLYS '+str(poly_locs.shape))
        selecteds = np.ones((len(poly_locs))) > 0
            
        if np.sum(selecteds) > 0:
            
            
            
            
            poly_xy = class_to_xy(poly_locs, 50)
            
            coeffs = poly_xy/49
           
                
            
            
            # logging.error('ASSOC IN GET SELECTED ' + str(assoc.shape))
            
            # selecteds = probs[...,-1] > thresh
            
            # detected_scores = np.ones((30))
            detected_coeffs = coeffs[selecteds,...]
            
            
            all_roads = np.zeros((196,200,3),np.float32)
            coef_all_roads = np.zeros((196,200,3),np.float32)
        #    if len(detected_scores) > 0:
                
            
            if 'poly_prob' in targets:
                # logging.error('ENTERED POLY SELECT ')
                poly_hamming = np.squeeze(targets['poly_hamming'].detach().cpu().numpy()) 
                poly_prob = np.squeeze(targets['poly_prob'].softmax(-1).detach().cpu().numpy())
                poly_centers = np.squeeze(targets['poly_centers'].detach().cpu().numpy())
                # temp_ham = poly_hamming[:,:-5]
            
                inter_poly_hamming = np.concatenate([poly_hamming[:,:len(selecteds)], poly_hamming[:,-5:]],axis=-1)
                
                
                detected_polys = poly_prob[:,1] > poly_thresh
                
                detected_poly_probs = poly_prob[detected_polys,1]
                if len(detected_poly_probs) > 0:
                
                    detected_poly_centers = poly_centers[detected_polys]
                    detected_poly_hamming = inter_poly_hamming[detected_polys] > poly_hamming_thresh
                    
                    not_all_zeros = np.logical_not(np.all(detected_poly_hamming, axis=-1)) 
                    detected_poly_hamming = detected_poly_hamming[not_all_zeros]
                    unique_hamming = np.unique(detected_poly_hamming,axis=0)
                    
                    temp_dict['poly_probs'] = detected_poly_probs
                    temp_dict['poly_centers'] = detected_poly_centers
                    temp_dict['poly_hamming'] = unique_hamming
                else:
                    temp_dict['poly_probs'] = []
                    temp_dict['poly_centers'] = []
                    temp_dict['poly_hamming'] = []
                    
                    
            if 'seq_estimates' in targets:
            
                seq_estimates = targets['seq_estimates'].detach().cpu().numpy() 
                
                selected_order_estimates = seq_estimates[:,:len(coeffs),:]
                
                double_selected_order_estimates = np.concatenate([selected_order_estimates[:,:, :len(coeffs)],selected_order_estimates[:,:, 50:]],axis=-1)
                temp_dict['selected_order_estimates'] = selected_order_estimates
                temp_dict['double_selected_order_estimates'] = double_selected_order_estimates
             
            else:
                temp_dict['selected_order_estimates'] = []
                temp_dict['double_selected_order_estimates'] = []
                
                
            
                
                    
            res_list = []
            res_coef_list=[]
            
            res_interpolated_list=[]
            # res_assoc_list = []
            
            for k in range(len(detected_coeffs)):
                
        
                control = detected_coeffs[k]
                
                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                
                interpolated = bezier.interpolate_bezier(control,100)
                
                res_interpolated_list.append(np.copy(interpolated))
                
        #            line = my_color_line_maker(interpolated,detected_endpoints[k],size=(196,200))
                line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
        #            res_list.append(line)
        #            res_coef_list.append(line2)
        #            all_roads = all_roads + np.float32(line)
                coef_all_roads = coef_all_roads + np.float32(line2)
            
            temp_dict['boxes'] = detected_coeffs
            # temp_dict['scores'] = detected_scores
            temp_dict['lines'] = res_list
            temp_dict['coef_lines'] = res_coef_list
            
            temp_dict['interpolated_points'] = res_interpolated_list
            
    #            temp_dict['all_roads'] = all_roads
            temp_dict['coef_all_roads'] = coef_all_roads
         
            
            temp_dict['assoc'] = targets['pred_assoc'].sigmoid().squeeze(0).detach().cpu().numpy()
            # temp_dict['start'] = targets['pred_start'].sigmoid().squeeze(0).detach().cpu().numpy()
            # temp_dict['fin'] = targets['pred_fin'].sigmoid().squeeze(0).detach().cpu().numpy()
            
            
            # to_merge = {'assoc': temp_dict['assoc'], 'start':temp_dict['start'],'fin':temp_dict['fin'], 'boxes':detected_coeffs}
            to_merge = {'assoc': temp_dict['assoc'], 'start':None,'fin':None, 'boxes':detected_coeffs}
            
            
            merged = get_merged_coeffs(to_merge)
            temp_dict['merged_coeffs'] = merged
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
          
            temp_dict['boxes'] = []
            temp_dict['lines'] = []
            temp_dict['coef_lines'] = []
            temp_dict['all_roads'] = []
            temp_dict['coef_all_roads'] = []
            temp_dict['labels'] = []
            temp_dict['assoc'] = []
            temp_dict['interpolated_points'] = []
            temp_dict['merged_interpolated_points'] = []
            temp_dict['merged_coeffs'] = []
    
    else:
        
        logging.error('DETECTED NOTHING')
        temp_dict['scores'] = []
      
        temp_dict['boxes'] = []
        temp_dict['lines'] = []
        temp_dict['coef_lines'] = []
        temp_dict['all_roads'] = []
        temp_dict['coef_all_roads'] = []
        temp_dict['labels'] = []
        temp_dict['assoc'] = []
        temp_dict['interpolated_points'] = []
        temp_dict['merged_interpolated_points'] = []
        temp_dict['merged_coeffs'] = []
        
 
        
    return temp_dict


def get_selected_inits(targets, thresh = 0.5):
    
    temp_dict = dict()
    
    
    # probs = np.squeeze(targets['init_point_detection_softmaxed'].detach().cpu().numpy(), axis=0)
    init_point_heatmap = targets['pred_init_point_softmaxed'].detach().cpu().numpy()
#    poly_locs = targets['pred_polys'].detach().cpu().numpy()
           
#    if not training:
    sth_exist = init_point_heatmap > thresh
    selecteds = np.where(init_point_heatmap > thresh)
#        
#    else:
#    selecteds = np.ones((len(poly_locs))) > 0
        
    if np.sum(sth_exist) > 0:
        
        init_row, init_col = selecteds
        
        to_send = np.stack([init_col,init_row],axis=-1)
        
    else:
        
        to_send=None
    
        
 
        
    return to_send
            
                
def get_vertices(adj):

    ins = []
    outs = []
    
    for k in range(len(adj)):
    #for k in range(7):
        for m in range(len(adj)):
        
            if adj[k,m] > 0.6:
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
def get_merged_network(targets):
    

#    scores = targets['scores']
#    labels = targets['labels']
   
    poly_locs = targets['pred_polys'].detach().cpu().numpy()
   
    assoc = np.squeeze(targets['pred_assoc'].sigmoid().detach().cpu().numpy(),axis=0) 
    # start_assoc = np.squeeze(targets['pred_start'].sigmoid().detach().cpu().numpy(),axis=0) 
    # fin_assoc = np.squeeze(targets['pred_fin'].sigmoid().detach().cpu().numpy() ,axis=0)
    
  

    poly_xy = class_to_xy(poly_locs, 50)
    
    coeffs = poly_xy/49
#    res_list = targets['lines'] 
#    res_coef_list = targets['coef_lines'] 
#    all_roads = targets['all_roads'] 
#    coef_all_roads = targets['coef_all_roads'] 
    # assoc = targets['assoc'] 
    # start_assoc = targets['start'] 
    # fin_assoc = targets['fin'] 
           
    diag_mask = np.eye(len(assoc))
    
    # start_assoc = np.clip(start_assoc + diag_mask,0,1)
    
    # fin_assoc = np.clip(fin_assoc + diag_mask,0,1)
    
    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask
    
    corrected_coeffs = np.copy(coeffs)
    
    # gat_start = gather_all_ends(start_assoc)
    # gat_fin = gather_all_ends(fin_assoc)
    
    # '''
    # HANDLE STARTS
    # '''
    # for k in range(len(gat_start)):
    #     all_points=[]
        
    #     for m in gat_start[k]:
    #         all_points.append(corrected_coeffs[m,0])
            
        
    #     av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
    #     for m in gat_start[k]:
    #         corrected_coeffs[m,0] = av_p
            
    
    # for k in range(len(gat_fin)):
    #     all_points=[]
        
    #     for m in gat_fin[k]:
    #         all_points.append(corrected_coeffs[m,-1])
            
        
    #     av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
    #     for m in gat_fin[k]:
    #         corrected_coeffs[m,-1] = av_p
            
        
    
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
        
    
def get_merged_lines(coeffs1,coeffs2):
    
#    base_control = coeffs[k]
                
#    coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
    
    interp_list = []

        
    control = coeffs1

    interpolated = bezier.interpolate_bezier(control)
    
    interp_list.append(interpolated)
    
    control = coeffs2

    interpolated = bezier.interpolate_bezier(control)
    
    interp_list.append(interpolated)
    
        
    all_points = np.concatenate(interp_list,axis=0)
        
    
    # logging.error('ALL POINTS ' + str(all_points.shape))
    # logging.error('LEN COEFFS1 ' + str(len(coeffs1)))
    new_coeffs = bezier.fit_bezier(all_points, len(coeffs1))[0]
        
    # new_coef_endpoints = np.concatenate([new_coeffs[0:1,:],new_coeffs[-1:,:]],axis=0)
    
    # new_interp = bezier.interpolate_bezier(new_coeffs)
    
    return None, new_coeffs

def visual_est(images,targets,save_path,name=None):
    b=0
    try:
#    probs = np.squeeze(targets['init_point_detection_softmaxed'].detach().cpu().numpy(), axis=0)
        if 'pred_init_point_softmaxed' in targets:
        
            init_point_heatmap = targets['pred_init_point_softmaxed'].detach().cpu().numpy()
            all_init_points = np.squeeze(init_point_heatmap)
        
#        est_assoc = np.squeeze(targets['pred_assoc'].sigmoid().detach().cpu().numpy(),axis=0) 
        # est_start = np.squeeze(targets['pred_start'].sigmoid().detach().cpu().numpy() ,axis=0)
        # est_fin = np.squeeze(targets['pred_fin'].sigmoid().detach().cpu().numpy() ,axis=0)
        
        
        # logging.error('EST ASSOC ' + str(est_assoc.shape))
        # logging.error('EST START ' + str(est_start.shape))
        # logging.error('EST FIN ' + str(est_fin.shape))
        
            
        if 'my_blob_mat' in targets:
            gt_blob_mat = np.squeeze(targets['my_blob_mat'])
          
            
            cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(gt_blob_mat) + 1))
            
            colored_image = cm(np.int64(gt_blob_mat))
            
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'estimate_common.png' ))
        
        
            
        if 'pred_polys' in targets:
            merged = get_merged_network(targets)
            
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
            poly_locs = targets['pred_polys'].detach().cpu().numpy()
       
            poly_xy = class_to_xy(poly_locs, 50)
            
            coeffs = poly_xy/49
           
            all_lanes = np.zeros((196,200))
    #    
    #    all_init_points = np.clip(np.sum(init_point_heatmap,axis=0),0,1)
        
        
            coef_all_roads = np.zeros((196,200,3))
            for k in range(len(coeffs)):
                
                lane_poly = convert_line_to_lane(coeffs[k], lane_width = 3.5)
                can = np.zeros((196,200))
        
                render_polygon(can, lane_poly, shape=(196,200), value=1)
                
                all_lanes = all_lanes + can
                
                res_lane = Image.fromarray(np.uint8(255*can))
                
                control = coeffs[k]
                    
                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                
                interpolated = bezier.interpolate_bezier(control,100)
                
                line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
                
                
                coef_all_roads = coef_all_roads + np.float32(line2)
                
        #        cur_init = init_point_heatmap[k]
        #        cur_init = gaussian_filter(cur_init, sigma=1)
                        
        #                logging.error('GAUSSIAN FILTERED')
        #        cur_init = cur_init/np.max(cur_init)
                        
        #        cur_init_point = Image.fromarray(np.uint8(255*cur_init))
                
                
        #        for m in range(len(est_assoc[k])):
        #            if est_assoc[k][m] > 0.5:
        #                control = coeffs[m]
        #            
        #                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        #                
        #                interpolated = bezier.interpolate_bezier(control,100)
        #                
        #                line3 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
        #                
        #                tot = np.clip(line2 + line3,0,1)
        #                temp_img = Image.fromarray(np.uint8( tot*255))
        #                temp_img.save(os.path.join(save_path,'matched_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
        #        
        #        for m in range(len(est_start[k])):
        #            if est_start[k][m] > 0.5:
        #                control = coeffs[m]
        #            
        #                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        #                
        #                interpolated = bezier.interpolate_bezier(control,100)
        #                
        #                line3 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
        #                
        #                tot = np.clip(line2 + line3,0,1)
        #                temp_img = Image.fromarray(np.uint8( tot*255))
        #                temp_img.save(os.path.join(save_path,'matched_start_from_'+str(k)+'to_'+str(m)+'.jpg' ))
        #        
        #        for m in range(len(est_fin[k])):
        #            if est_fin[k][m] > 0.5:
        #                control = coeffs[m]
        #            
        #                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        #                
        #                interpolated = bezier.interpolate_bezier(control,100)
        #                
        #                line3 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
        #                
        #                tot = np.clip(line2 + line3,0,1)
        #                temp_img = Image.fromarray(np.uint8( tot*255))
        #                temp_img.save(os.path.join(save_path,'matched_fin_from_'+str(k)+'to_'+str(m)+'.jpg' ))
        #        
        #        
                res = Image.fromarray(line2)
        ##        prob_img = Image.fromarray(np.uint8(255*np.ones((20,20))*probs[k,-1]))
                if name==None:
                    res.save(os.path.join(save_path,'batch_'+str(b) + '_est_interp_road_'+str(k)+'.jpg'))
                    
                    # res_lane.save(os.path.join(save_path,'batch_'+str(b) + '_est_lane_'+str(k)+'.jpg'))
        #            prob_img.save(os.path.join(save_path,'batch_'+str(b) + '_prob_'+str(k)+'.jpg'))
                    
        #            cur_init_point.save(os.path.join(save_path,'batch_'+str(b) + '_init_'+str(k)+'.jpg'))
                
                else:
                    res.save(os.path.join(save_path,name + '_est_interp_road_'+str(k)+'.jpg'))
          
                    # res_lane.save(os.path.join(save_path,name + '_est_lane_'+str(k)+'.jpg'))
        #            prob_img.save(os.path.join(save_path,name + '_prob_'+str(k)+'.jpg'))
        #            cur_init_point.save(os.path.join(save_path,name + '_init_'+str(k)+'.jpg'))
        #
        #    
                
                
        #                plt.figure()
        #                fig, ax = plt.subplots(1, figsize=(196,200))
        ##                axes = plt.gca()
        #                ax.set_xlim([0,1])
        #                ax.set_ylim([0,1])
        #                plt.plot(interpolated[:,0],interpolated[:,1])
        #                
        #                plt.savefig(os.path.join(save_path,'batch_'+str(b) + '_est_interp_road_'+str(k)+'.png'), bbox_inches='tight', pad_inches=0.0)   
        #                plt.close()  
            
                # merged, merged_coeffs = get_merged_lines(coeffs,assoc,k)
              
            all_lanes = np.uint8(np.clip(all_lanes,0,1)*255)
            if name==None:
                
                temp_img = Image.fromarray(all_lanes)
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_all_lanes.png' ))       
                
                temp_img = Image.fromarray(np.uint8(all_init_points*255))
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_all_init_points.png' ))       
                
                
                coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
                temp_img = Image.fromarray(coef_all_roads)
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_all_roads.png' ))       
            else:
                temp_img = Image.fromarray(all_lanes)
                temp_img.save(os.path.join(save_path,name + '_est_all_lanes.png' ))    
                
                temp_img = Image.fromarray(np.uint8(all_init_points*255))
                temp_img.save(os.path.join(save_path,name + '_est_all_init_points.png' ))    
                
                
                coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
                temp_img = Image.fromarray(coef_all_roads)
                temp_img.save(os.path.join(save_path,name + '_est_coef_all_roads.png' ))  
                
            
    except Exception as e:
        logging.error('VISUAL EST ')
        logging.error(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(str((exc_type, fname, exc_tb.tb_lineno)))

def visual_masks_gt(images,targets,save_path,name=None):
 
    
    for b in range(len(targets)):
        true_assoc = targets[b]['con_matrix']
        true_start_assoc = targets[b]['start_con_matrix']
        true_fin_assoc = targets[b]['fin_con_matrix']
        
        init_point_matrix = targets[b]['init_point_matrix'].cpu().numpy()
        
        sort_index = targets[b]['sort_index'].cpu().numpy()
        
        img_centers = targets[b]['center_img']
        
        img_centers = img_centers.cpu().numpy()
        
        orig_img_centers = targets[b]['orig_center_img']
        
        orig_img_centers = orig_img_centers.cpu().numpy()
        
        roads = targets[b]['roads'].cpu().numpy()[sort_index]
        
        all_endpoints = targets[b]['endpoints'].cpu().numpy()
        
        origs = targets[b]['origs'].cpu().numpy()
        
        all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        coef_all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        
        grid_all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))

        all_init_points = np.clip(np.sum(init_point_matrix,axis=0),0,1)
        
        orig_coefs = targets[b]['control_points'].cpu().numpy()[sort_index]
        grid_coefs = targets[b]['grid_sorted_control_points'].cpu().numpy()
        
        grid_coefs = grid_coefs/49
        grid_endpoints = get_grid_endpoints_from_coeffs(grid_coefs)
        coef_endpoints = get_endpoints_from_coeffs(orig_coefs)
    
        all_masks = targets[b]['mask'].cpu().numpy()
        occ_img = Image.fromarray(np.uint8(255*all_masks[1]))
        vis_img = Image.fromarray(np.uint8(255*all_masks[0]))
        
        # all_lanes = np.zeros((196,200))
        
        # drivable_area = targets[b]['bev_mask'].cpu().numpy()[0]
        # driv_img = Image.fromarray(np.uint8(255*drivable_area))
        
        # lane_area = targets[b]['static_mask'].cpu().numpy()[4]
        # lane_img = Image.fromarray(np.uint8(255*lane_area))
        
        # van_occ = targets[b]['bev_mask'].cpu().numpy()[-1]
        # van_occ = Image.fromarray(np.uint8(255*van_occ))
        if name==None:
            
            # lane_img.save(os.path.join(save_path,'batch_'+str(b) + '_lane_layer.jpg'))
            
            # van_occ.save(os.path.join(save_path,'batch_'+str(b) + '_van_occ.jpg'))
            # driv_img.save(os.path.join(save_path,'batch_'+str(b) + '_drivable.jpg'))
            occ_img.save(os.path.join(save_path,'batch_'+str(b) + '_occ.jpg'))
            vis_img.save(os.path.join(save_path,'batch_'+str(b) + '_vis.jpg'))
        else:
            # lane_img.save(os.path.join(save_path,name + '_lane_layer.jpg'))
            
            # van_occ.save(os.path.join(save_path,name + '_van_occ.jpg'))
            # driv_img.save(os.path.join(save_path,name + '_drivable.jpg'))
            occ_img.save(os.path.join(save_path,name + '_occ.jpg'))
            vis_img.save(os.path.join(save_path,name + '_vis.jpg'))
            
        for k in range(len(roads)):
#            for m in range(len(true_assoc[k])):
#                if true_assoc[k][m] > 0.5:
#                    first_one = add_endpoints_to_line(origs[k],coef_endpoints[k])
#                    second_one = add_endpoints_to_line(origs[m],coef_endpoints[m])
#                    tot = np.clip(first_one + second_one,0,1)
#                    temp_img = Image.fromarray(np.uint8( tot*255))
#                    temp_img.save(os.path.join(save_path,'gt_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
#            
#            for m in range(len(true_start_assoc[k])):
#                if true_start_assoc[k][m] > 0.5:
#                    first_one = add_endpoints_to_line(origs[k],coef_endpoints[k])
#                    second_one = add_endpoints_to_line(origs[m],coef_endpoints[m])
#                    tot = np.clip(first_one + second_one,0,1)
#                    temp_img = Image.fromarray(np.uint8( tot*255))
#                    temp_img.save(os.path.join(save_path,'gt_start_from_'+str(k)+'to_'+str(m)+'.jpg' ))
#            
#            
#            for m in range(len(true_fin_assoc[k])):
#                if true_fin_assoc[k][m] > 0.5:
#                    first_one = add_endpoints_to_line(origs[k],coef_endpoints[k])
#                    second_one = add_endpoints_to_line(origs[m],coef_endpoints[m])
#                    tot = np.clip(first_one + second_one,0,1)
#                    temp_img = Image.fromarray(np.uint8( tot*255))
#                    temp_img.save(os.path.join(save_path,'gt_fin_from_'+str(k)+'to_'+str(m)+'.jpg' ))
            
            
            
            cur_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),all_endpoints[k])
            cur_coef_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),coef_endpoints[k])
            
            
            cur_grids = grid_coefs[k]
            interpolated = bezier.interpolate_bezier(cur_grids,100)
            
            line = my_color_line_maker(interpolated,grid_endpoints[k],size=(196,200))
            
            grid_all_roads = grid_all_roads + np.copy(line)
            
            grid_img = Image.fromarray(line)
            
            
#            all_roads[img_centers == roads[k]] = 1
            temp_img = Image.fromarray(np.uint8(cur_full*255))
            
            temp_coef_img = Image.fromarray(np.uint8(cur_coef_full*255))
            
            all_roads = all_roads + cur_full
            coef_all_roads = coef_all_roads + cur_coef_full
            
            lane_poly = convert_line_to_lane(np.reshape(orig_coefs[k],(-1,2)), lane_width = 3.5)
            can = np.zeros((196,200))
    
            render_polygon(can, lane_poly, shape=(196,200), value=1)
            
            # all_lanes = all_lanes + can
            lane_img = Image.fromarray(np.uint8(can*255))
            
            
            cur_init = Image.fromarray(np.uint8(255*init_point_matrix[k])).resize((196,200))
            
            
#            if name==None:
##                orig_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_orig_road_'+str(k)+'.jpg' ))
#                grid_img.save(os.path.join(save_path,'batch_'+str(b) + '_grid_road_'+str(k)+'.jpg' ))
#                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_road_'+str(k)+'.jpg' ))
#                temp_coef_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_road_'+str(k)+'.jpg' ))
#                
#                lane_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_lane_'+str(k)+'.jpg' ))
#                
#                cur_init.save(os.path.join(save_path,'batch_'+str(b) + '_gt_gauss_init_'+str(k)+'.jpg' ))
#                
#                
#            
#            else:
##                orig_img.save(os.path.join(save_path,name + '_gt_orig_road_'+str(k)+'.jpg' ))
#                
#                # orig_temp_img.save(os.path.join(save_path,name + '_gt_comp_road_'+str(k)+'.jpg' ))
#                grid_img.save(os.path.join(save_path,name + '_grid_road_'+str(k)+'.jpg' ))
#                temp_img.save(os.path.join(save_path,name + '_gt_visible_road_'+str(k)+'.jpg' ))
#                temp_coef_img.save(os.path.join(save_path,name + '_gt_coef_visible_road_'+str(k)+'.jpg' ))
#                lane_img.save(os.path.join(save_path,name + '_gt_lane_'+str(k)+'.jpg' ))
#
#                cur_init.save(os.path.join(save_path,name + '_gt_gauss_init_'+str(k)+'.jpg' ))


                
        all_roads = np.clip(all_roads,0,1)
        coef_all_roads = np.clip(coef_all_roads,0,1)
        
        # all_lanes = np.clip(all_lanes,0,1)
        
        grid_all_roads = np.clip(grid_all_roads,0,1)
        
        
        if name==None:
            temp_img = Image.fromarray(np.uint8(grid_all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_grid_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(all_init_points*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_all_inits.png' ))
            
            # temp_img = Image.fromarray(np.uint8(all_lanes*255))
            # temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_all_lanes.png' ))
            
            # temp_img = Image.fromarray(np.uint8(all_orig_roads*255))
            # temp_img.save(os.path.join(save_path,'batch_'+str(b)  + '_gt_comp_all_roads.png' ))
            
        else:
            temp_img = Image.fromarray(np.uint8(grid_all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_grid_all_roads.png' ))
            
            
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_coef_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(all_init_points*255))
            temp_img.save(os.path.join(save_path,name + '_gt_all_inits.png' ))
            
            
            # temp_img = Image.fromarray(np.uint8(all_lanes*255))
            # temp_img.save(os.path.join(save_path,name + '_gt_all_lanes.png' ))
            
            # temp_img = Image.fromarray(np.uint8(all_orig_roads*255))
            # temp_img.save(os.path.join(save_path,name + '_gt_comp_all_roads.png' ))
     
     
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
    
def get_grid_endpoints_from_coeffs(coeffs):
    
    start = coeffs[:,0,:]
    end = coeffs[:,-1,:]
    
    return np.concatenate([start,end],axis=-1)


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
                my_gt = np.clip(np.sum(all_gt_lines[my_cur_est],axis=0),0,1)
                
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
    

def save_matched_results(outputs, out, matched_poly,naive_matched_labels,  targets, config,save_path, gt=False, common_poly=None,val=False):
    # _, target_ids = target_ids
    
    # inter_points = inter_dict['interpolated'].detach().cpu().numpy()
    try:
        img_centers = targets['center_img']
        
        
        real_poly_one_hots = targets["real_poly_one_hots"].cpu().numpy() > 0.5
        real_poly_centers = targets["real_poly_centers"].cpu().numpy()
        real_blob_mat = np.squeeze(targets['pre_blob_mat'].cpu().numpy())
        real_blob_ids = np.squeeze(targets['pre_blob_ids'].cpu().numpy())
        
        origs = targets['origs'].cpu().numpy()
        main_origs = np.copy(origs)
        img_centers = img_centers.cpu().numpy()
            
        '''
        POLYGON STUFF
        '''
        
        
        # logging.error('POLY ORDERED EST  ' + str(poly_ordered_est.shape))
        
        # logging.error('POLY TO RETURN  ' + str(poly_to_return))
        
        
        
        n_points = 100
        
#        if 'pred_init_point_softmaxed' in outputs:
#            init_point_heatmap = outputs['pred_init_point_softmaxed'].detach().cpu().numpy()
        
#        est_assoc = np.squeeze(outputs['pred_assoc'].sigmoid().detach().cpu().numpy(),axis=0) 
        # est_start = np.squeeze(outputs['pred_start'].sigmoid().detach().cpu().numpy() ,axis=0)
        # est_fin = np.squeeze(outputs['pred_fin'].sigmoid().detach().cpu().numpy() ,axis=0)
        
        if 'pred_polys' in outputs:
        
            poly_locs = outputs['pred_polys'].detach().cpu().numpy()
           
            poly_xy = class_to_xy(poly_locs, 50)
            
            est_coeffs = poly_xy/49
            
            if not val:

        
                poly_to_return = matched_poly
                
                
                
                my_src_poly = np.int64(poly_to_return[0][0].cpu().numpy())
                my_tgt_poly = np.int64(poly_to_return[0][1].cpu().numpy())
                
                est_poly_centers = np.squeeze(outputs['poly_centers'].detach().cpu().numpy())
                
                
                est_poly_one_hots = np.squeeze(outputs['poly_hamming'].sigmoid().detach().cpu().numpy()) > 0.5
                
            
                est_poly_one_hots = np.concatenate([est_poly_one_hots[:,:len(est_coeffs)], est_poly_one_hots[:,-5:]],axis=1)
            
            est_probs = np.ones((len(est_coeffs))) > 0
            orig_inter = bezier.batch_interpolate(est_coeffs, n_points) # N x n_points x 2
                 
                    
            all_made_lines = []
            for k in range(len(orig_inter)):
                    
                cur_est = orig_inter[k,...]
                cur_est = np.float32(my_line_maker(cur_est))/255
                all_made_lines.append(np.copy(cur_est)) 
                
            
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
            
            all_made_lines = np.concatenate([np.stack(all_made_lines,axis=0),np.stack(boundary_lines,axis=0)],axis=0)
            
            all_est_roads = np.clip(np.sum(all_made_lines[np.concatenate([est_probs,np.array([True,True,True,True,True])],axis=0)],axis=0),0,1)
            
            origs = np.float32(main_origs > 0)
            # boundaries = np.stack([upper, left, right, bev_left, bev_right],axis=0)
            origs = np.concatenate([origs, np.stack(boundary_lines,axis=0)],axis=0)
            
        
        # gt_polys = np.squeeze(targets['poly_list'].cpu().numpy())
        gt_blob_mat = np.squeeze(targets['blob_mat'].cpu().numpy())
        gt_blob_ids = np.squeeze(targets['blob_ids'].cpu().numpy())
        gt_poly_centers = np.squeeze(targets['poly_centers'].cpu().numpy())
        gt_one_hot_polys = np.squeeze(targets['poly_one_hots'].cpu().numpy()) > 0.5
        
        
        cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(gt_blob_mat) + 1))
        
        colored_image = cm(np.int64(gt_blob_mat))
        
        Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'poly_all_gt.png' ))
        
        if not val:
        
            real_poly_one_hots = targets["real_poly_one_hots"].cpu().numpy() > 0.5
            real_poly_centers = targets["real_poly_centers"].cpu().numpy()
            real_blob_mat = np.squeeze(targets['pre_blob_mat'].cpu().numpy())
            real_blob_ids = np.squeeze(targets['pre_blob_ids'].cpu().numpy())
            # all_blobs_img = Image.fromarray(np.uint8(gt_blob_mat), mode='P' )
            # all_blobs_img.save(os.path.join(save_path,'poly_all_gt.png' ))
            cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(real_blob_mat) + 1))
            
            colored_image = cm(np.int64(real_blob_mat))
            
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'poly_all_pre.png' ))
        else:
            # if not np.any(common_poly==None):
            common_poly_centers, common_poly_one_hots, common_blob_mat, common_blob_ids, common_real_hots = common_poly
            cm = plt.get_cmap('gist_rainbow',lut=np.int32(np.max(common_blob_mat) + 1))
        
            colored_image = cm(np.int64(common_blob_mat))
            
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(save_path,'common_polys.png' ))
 
                
        
        
        # all_blobs_img = Image.fromarray(np.uint8(real_blob_mat), mode='P' )
        # all_blobs_img.save(os.path.join(save_path,'poly_all_pre.png' ))
        if not val:
            for k in range(len(gt_one_hot_polys)):
            
                
                
                '''
                GT
                '''
                
                cur_gt = np.float32(np.copy(img_centers) > 0)
                cur_gt[gt_blob_mat == gt_blob_ids[k]] = 0.5
                
                # logging.error('BLOB MAT')
                
                cur_gt = np.stack([cur_gt, cur_gt, cur_gt],axis=-1)
                
                my_cur_gt = gt_one_hot_polys[k]
                
                gt_border = np.sum(origs[my_cur_gt],axis=0)
                cur_gt[gt_border > 0 , 0] = 1
                cur_gt[gt_border > 0 , 1:] = 0
                
                
                
                center_ar = np.zeros((196,200,3))
                
                center_ar[int(gt_poly_centers[k][1]*195),int(gt_poly_centers[k][0]*199),-1] = 1
                struct = ndimage.generate_binary_structure(2, 2)
           
                dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
                # center_ar[...,-1] = dilated
                
                
                # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
                cur_gt[dilated,-1] = 1
                cur_gt[dilated,:-1] = 0
                
                my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
                my_gt_img.save(os.path.join(save_path,'pre_poly_gt_'+str(k) +'.jpg' ))
                
            if not gt:
                
                for k in range(len(my_tgt_poly)):
                    
                    cur_gt = np.float32(np.copy(all_est_roads) > 0)
                    
                    
                    # logging.error('my_tgt_poly[k]' + str(my_tgt_poly[k]))
                    
                    cur_gt[real_blob_mat == real_blob_ids[my_tgt_poly[k]]] = 0.5
                    
                    # logging.error('BLOB MAT')
                    
                    cur_gt = np.stack([cur_gt, cur_gt, cur_gt],axis=-1)
                    
                    my_cur_gt = real_poly_one_hots[np.int64(my_tgt_poly[k])]
                    
                    gt_border = np.sum(all_made_lines[my_cur_gt > 0],axis=0)
                    cur_gt[gt_border > 0 , 0] = 1
                    cur_gt[gt_border > 0 , 1:] = 0
                    
                    
                    
                    center_ar = np.zeros((196,200,3))
                    
                    center_ar[int(real_poly_centers[my_tgt_poly[k]][1]*195),int(real_poly_centers[my_tgt_poly[k]][0]*199),-1] = 1
                    struct = ndimage.generate_binary_structure(2, 2)
               
                    dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
                    # center_ar[...,-1] = dilated
                    
                    
                    # cur_gt[int(gt_poly_centers[my_tgt_poly[k]][1]*195),int(gt_poly_centers[my_tgt_poly[k]][0]*199),:-1] = 0
                    cur_gt[dilated,-1] = 1
                    cur_gt[dilated,:-1] = 0
                    
                    my_gt_img = Image.fromarray(np.uint8(cur_gt*255))
                    my_gt_img.save(os.path.join(save_path,'pre_poly_matched_pre_'+str(k) +'.jpg' ))
                    
            else:
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
                    
           
            merged = get_merged_network(outputs)
                
      
            if len(merged) > 0:
            
                printed_estimateds = np.clip(np.sum(np.stack(merged,axis=0),axis=0), 0, 1)
         
                my_base_est = np.float32(np.copy(printed_estimateds) > 0)
                # my_base_est = np.stack([my_base_est, my_base_est, my_base_est],axis=-1)
                
    #        all_made_lines = np.concatenate([all_made_lines,np.stack(boundary_lines,axis=0)],axis=0)
            for k in range(len(my_src_poly)):     
                
                # logging.error('GT SAVED')
                '''
                EST
                '''
                my_est = np.copy(my_base_est)
                
                my_cur_est = est_poly_one_hots[my_src_poly[k]]
                
                # logging.error('CUR EST')
                
                # logging.error('MY CUR EST ' + str(my_cur_est.shape))
                
                # logging.error('ALL MADE LINES ' + str(all_made_lines.shape))
                
                my_border = np.clip(np.sum(all_made_lines[my_cur_est> 0],axis=0),0,1)
                my_est[my_border > 0 , 0] = 1
                my_est[my_border > 0 , 1:] = 0
                
    #            my_est = np.clip(np.sum(all_made_lines[my_cur_est > 0],axis=0),0,1)
                
    #            my_est = np.stack([my_est, my_est, my_est],axis=-1)
                center_ar = np.zeros((196,200,3))
                center_ar[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),-1] = 1
                struct = ndimage.generate_binary_structure(2, 2)
           
                dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct)
                # my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),-1] = 1
                my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),:] = 0
                my_est[dilated,-1] = 1
                
                # center_ar = np.zeros((196,200,3))
                # center_ar[int(poly_voronoi[k,1]*195),int(poly_voronoi[k,0]*199),1] = 1
                # struct = ndimage.generate_binary_structure(2, 2)
           
                # dilated = ndimage.binary_dilation(center_ar[...,-1]>0, structure=struct,iterations = 3)
                # # my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),-1] = 1
                # # my_est[int(est_poly_centers[my_src_poly[k]][1]*195),int(est_poly_centers[my_src_poly[k]][0]*199),:] = 0
                
                # my_est[dilated,:] = 0
                
                # my_est[dilated,1] = 1
                
                my_est_img = Image.fromarray(np.uint8(my_est*255))
                my_est_img.save(os.path.join(save_path,'pre_poly_matched_est_'+str(k) +'.jpg' ))
                
    
    except Exception as e:
        logging.error('MATCHED SAVE ')
        logging.error(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(str((exc_type, fname, exc_tb.tb_lineno)))

        
def save_naive_results(outputs, naive_matched_labels, fuzzy_coeffs,save_path):
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
        
        init_point_heatmap = outputs['pred_init_point_softmaxed'].detach().cpu().numpy()
        poly_locs = outputs['pred_polys'].detach().cpu().numpy()
       
        est_assoc = np.squeeze(outputs['pred_assoc'].sigmoid().detach().cpu().numpy(),axis=0) 
       
    
        poly_xy = class_to_xy(poly_locs, 50)
        
        est_coeffs = poly_xy/49
        
        est_probs = np.ones((len(est_coeffs))) > 0
        n_points=100
        res_interpolated_list = bezier.batch_interpolate(est_coeffs, n_points) # N x n_points x 2
             
        fuzzy_interpolated =   bezier.batch_interpolate(fuzzy_coeffs.detach().cpu().numpy(), n_points)
        # logging.error('NAIVE SAVE '+ str(res_interpolated_list.shape))
        all_fuzzy=[]
        all_made_lines = []
        for k in range(len(res_interpolated_list)):
                
            cur_est = res_interpolated_list[k]
            cur_est = np.float32(my_line_maker(cur_est))/255
            all_made_lines.append(np.copy(cur_est)) 
             
            
            cur_est = fuzzy_interpolated[k]
            cur_est = np.float32(my_line_maker(cur_est))/255
            all_fuzzy.append(np.copy(cur_est)) 
             
            my_gt_img = Image.fromarray(np.uint8(cur_est*255))
            my_gt_img.save(os.path.join(save_path,'fuzzy_'+str(k)+'.jpg' ))
            
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
        
        
def save_train_order_rnn( outputs, train_order ,config,save_path):
    
    init_point_heatmap = outputs['pred_init_point_softmaxed'].detach().cpu().numpy()
    poly_locs = outputs['pred_polys'].detach().cpu().numpy()
   
    est_assoc = np.squeeze(outputs['pred_assoc'].sigmoid().detach().cpu().numpy(),axis=0) 
   

    poly_xy = class_to_xy(poly_locs, 50)
    
    est_coeffs = poly_xy/49
    
    est_probs = np.ones((len(est_coeffs))) > 0
    n_points=100
    inter_points = bezier.batch_interpolate(est_coeffs, n_points) # N x n_points x 2
         
    
    # logging.error('TRAIN ORDER ' + str(train_order.shape))
    # logging.error('TRAIN ORDER SEL ' + str(train_order_sel.shape))
    
    all_made_lines = []
    for k in range(len(inter_points)):
            
        cur_est = inter_points[k,...]
        cur_est = np.float32(my_line_maker(cur_est))/255
        all_made_lines.append(np.copy(cur_est)) 
        
        
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
    
    all_made_lines = np.concatenate([np.stack(all_made_lines,axis=0),np.stack(boundary_lines,axis=0)],axis=0)
  
    
    for k in range(len(train_order)):
        gt_ar = np.copy(all_made_lines[k])
        gt_ar = np.stack([gt_ar, gt_ar, gt_ar],axis=-1)
        base = np.copy(gt_ar)
        for m in range(len(train_order[k])):
            if train_order[k,m] == 55:
                break
            elif train_order[k,m] >= 50:
                temp = all_made_lines[int(train_order[k][m] - 50 + len(est_coeffs))]
                
            else:
                temp = all_made_lines[int(train_order[k][m])]
     
            temp_img = np.copy(base)
            temp_img[temp > 0.5,0] = 1
            temp_img[temp > 0.5,1:] = 0
            
            temp_img = Image.fromarray(np.uint8(temp_img*255))  
            temp_img.save(os.path.join(save_path,'rnn_train_orders_'+str(k) +'_' + str(m)+'.jpg' ))
                        
    
    
def save_results_train(image,outputs, out, targets,  config, poly_indices, metric_visuals=None, rnn=False,
                       train_targets_order=None, gt_train=False,
                       naive_matched_labels=None, naive=False, poly_voronoi=None,fuzzy_coeffs=None):
    
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
    
    

        
    
    try:  
        visual_masks_gt(np.uint8(image),targets,os.path.join(config.save_logdir,'train_images'))
    except Exception as e:
        logging.error("PROBLEM IN VISUAL GT TRAIN SAVE: " + str(e))
    try:  
        visual_est(np.uint8(image),outputs,os.path.join(config.save_logdir,'train_images'))
    except Exception as e:
        logging.error("PROBLEM IN VISUAL EST TRAIN SAVE: " + str(e))
    
    
    if rnn:    
        save_train_order_rnn( outputs, train_targets_order ,config,os.path.join(config.save_logdir,'train_images'))
    
    try:  
    
        save_metric_visuals(out, targets[0], metric_visuals, config,os.path.join(config.save_logdir,'train_images'), rnn=rnn, train_targets_order=train_targets_order,naive=naive)
      
            
    except Exception as e:
        logging.error("PROBLEM IN METRIC VISUAL SAVE: " + str(e))
        
    if ((not rnn) & (not naive)):
        try:
            save_matched_results(outputs, out, poly_indices, naive_matched_labels,  targets[0],config,os.path.join(config.save_logdir,'train_images'), gt=gt_train)
        except Exception as e:
            logging.error("PROBLEM IN MATCHED TRAIN SAVE: " + str(e))
    
  
def save_results_eval(image,outputs, out,targets,  config, gt_train=False, common_poly=None):
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

    
    for fr in range(len(image)):
        cur_img = Image.fromarray(np.uint8(image[fr,...]))
        cur_img.save(os.path.join(base_path,'image.jpg'))       
    
 
#    try:
#        visual_masks_gt(np.uint8(image),targets,base_path,name='_')
#    except Exception as e:
#        logging.error("PROBLEM IN VISUAL MASKS GT VAL SAVE: " + str(e))
        
    try:     
        visual_est(np.uint8(image),out,base_path,name='_')
    except Exception as e:
        logging.error("PROBLEM IN VISUAL EST VAL SAVE: " + str(e))
        

        

#    try:
#        save_matched_results(outputs, out, None, None,  targets[0],config,base_path, gt=gt_train, common_poly=common_poly,val=True)
#    except Exception as e:
#        logging.error("PROBLEM IN MATCHED TRAIN SAVE: " + str(e))
#   
def img_saver(img,path):
    img = Image.fromarray(np.uint8(img))
    img.save(path)
#

