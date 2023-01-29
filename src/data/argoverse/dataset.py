import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.utils import bezier
from .utils import IMAGE_WIDTH, IMAGE_HEIGHT, ARGOVERSE_CLASS_NAMES
from ..utils import decode_binary_labels

import numpy as np
import logging
import sys
import cv2
from scipy.ndimage import gaussian_filter
from skimage import measure
import glob

class ArgoverseMapDataset(Dataset):

    def __init__(self, config, loader, am, 
                 log_names=None, train=True, pinet=False, gt_polygon_extraction=False):

        self.image_size = config.image_size
        self.config = config
        self.examples = []
        
        self.gt_polygon_extraction = gt_polygon_extraction
        self.pinet = pinet
        self.am = am
        self.calibs = dict()

        # Preload training examples from Argoverse train and test sets
        self.loader = loader

        self.n_control = 3
        self.camera = "ring_front_center"
        self.argoverse_data_root = os.path.join(config.nusc_root,'argoverse-tracking','all_logs')
        self.preload(loader, log_names)
        
        if not self.gt_polygon_extraction:
            if train:
                
                all_paths = glob.glob(os.path.join(config.poly_base_path,'train_polygon_dict*'))
                    
                dicts_list = []
                for pa in all_paths:
                    dicts_list.append(np.load(pa, allow_pickle=True).item())
                
                self.gt_poly_dict = dict()
                for di in dicts_list:
                    self.gt_poly_dict.update(di)
            
            else:
                all_paths = glob.glob(os.path.join(config.poly_base_path,'val_polygon_dict*'))
                
                dicts_list = []
                for pa in all_paths:
                    dicts_list.append(np.load(pa, allow_pickle=True).item())
                
                self.gt_poly_dict = dict()
                for di in dicts_list:
                    self.gt_poly_dict.update(di)
                
        logging.error('ARGO LOADED')
            
    def preload(self, loader, log_names=None):
        

        for my_scene_id in range(len(log_names)):
    
            log = loader.get(log_names[my_scene_id])
#            
            n_frames_in_scene = log.num_lidar_frame
            
            for k in range(n_frames_in_scene):
            
                timestamp = str(np.copy(loader._image_timestamp_list_sync[log_names[my_scene_id]][self.camera][k]))

                self.examples.append((timestamp,log_names[my_scene_id], k))
                
    
    def __len__(self):
        return len(self.examples)
    

    def __getitem__(self, idx):

        # Get the split, log and camera ids corresponding to the given timestamp
        try:
            timestamp, logid, ind = self.examples[idx]
            
            image = self.load_image(logid, timestamp)
            
            calib = self.load_calib(logid)
            
            # logging.error('IMAGE ' + str(np.any(image == None)))
            
            con_matrix,endpoints,\
        orig_img_centers, origs,mask,\
        to_return_centers,labels, roads,\
        coeffs, outgoings, incomings, problem= self.load_line_labels(timestamp, logid, ind)
    
       
            if problem:
                logging.error('THERE WAS PROBLEM')
                
                return (None, dict(), True)
            
            
            
            init_points = np.reshape(endpoints,(-1,2,2))[:,0]
            
            sorted_init_points, sort_index = self.get_sorted_init_points(init_points)

            temp_ar = np.zeros((len(sorted_init_points),100,100))
            for k in range(len(sorted_init_points)):
                temp_ar[k,int(np.clip(sorted_init_points[k,1]*100,0,99)),int(np.clip(sorted_init_points[k,0]*100,0,99))]=1
                
#                logging.error('PLACED 1')
                temp_ar[k] = gaussian_filter(temp_ar[k], sigma=0.1)
                
#                logging.error('GAUSSIAN FILTERED')
                temp_ar[k] = temp_ar[k]/np.max(temp_ar[k])
                
            
            # sorted_points = np.copy(np.ascontiguousarray(coeffs[sort_index,:]))
            sorted_points = np.copy(coeffs)
            grid_sorted_points = np.reshape(sorted_points,(-1,self.n_control ,2))
            grid_sorted_points[...,0]= np.int32(grid_sorted_points[...,0]*49)
            grid_sorted_points[...,1]= np.int32(grid_sorted_points[...,1]*49)
            
            my_grid_points = np.copy(np.ascontiguousarray(grid_sorted_points))
            
            # try:
                
            #     if self.gt_polygon_extraction:
                
            #         poly_centers, poly_one_hots, blob_mat, blob_ids = self.get_polygons(to_return_centers, roads)
            #     else:
            #         blob_mat, poly_centers, poly_one_hots, blob_ids = self.gt_poly_dict[logid+'_'+str(timestamp)]
                
            #     if np.any(blob_mat == None):
            #         return (None, dict(), True)
                

            # except Exception as e:
            #     return (None, dict(), True)
            #     logging.error('GT POLYGON EXCEPTION : ' + str(e))
            
            
            target = dict()
            intersections = self.get_order_labels(coeffs)
            target['gt_order_labels'] = torch.tensor(intersections).long()
            
            # target['blob_ids'] = torch.tensor(blob_ids).long()
            # target['blob_mat'] = torch.tensor(blob_mat).long()
            # target['poly_one_hots'] = torch.tensor(poly_one_hots).float()
            # target['poly_centers'] = torch.tensor(poly_centers).float()
            
            target['blob_ids'] = torch.tensor(0).long()
            target['blob_mat'] = torch.tensor(0).long()
            target['poly_one_hots'] =  torch.tensor(0).long()
            target['poly_centers'] =  torch.tensor(0).long()
            
        
            target['mask'] = torch.tensor(mask).float()
            target['bev_mask'] = torch.zeros(self.config.num_bev_classes,mask.shape[1],mask.shape[2])
            target['static_mask'] = torch.zeros(self.config.num_bev_classes,mask.shape[1],mask.shape[2])
            
            target['calib'] = calib.float()
            target['center_img'] = to_return_centers
            target['orig_center_img'] = orig_img_centers
            target['labels'] = labels.long()
            target['roads'] = torch.tensor(np.int64(roads)).long()
            target['control_points'] = torch.tensor(coeffs)
            target['con_matrix'] = torch.tensor(con_matrix)
            target['obj_exists'] = torch.tensor(False)
            
            target['init_point_matrix'] = torch.tensor(np.copy(np.ascontiguousarray(temp_ar))).float()
            
            target['sorted_control_points'] = torch.tensor(sorted_points).float()
            
            target['grid_sorted_control_points'] = torch.tensor(my_grid_points).float()
            
#           
            target['sort_index'] = torch.tensor(np.copy(np.ascontiguousarray(sort_index)))
            
#           
            target['endpoints'] = torch.tensor(endpoints).float()

            target['origs'] = torch.tensor(origs)
            
            target['scene_token'] = logid
            target['sample_token'] = timestamp
            
            target['data_token'] = logid
            
            target['scene_name'] = logid
            target['outgoings'] = outgoings
            target['incomings'] = incomings
            target['left_traffic'] = torch.tensor(False)
            return (image, target, False)
        
        except Exception as e:
            logging.error('ARGO DATALOADER ' + str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
       
            return (None, dict(), True)

    

    def get_order_labels(self, coeffs):
        
        
        n_points = 200
        coeffs = np.reshape(coeffs,(coeffs.shape[0],-1,2))
        res_ar = bezier.batch_interpolate(coeffs, n_points) # N x n_points x 2
        
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
        
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
        
        boundaries = np.stack([upper,  left, right, bev_left, bev_right],axis=0)
        
        res_ar = np.concatenate([res_ar, boundaries],axis=0)
        
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
        
        pad_size = 20
        
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
                    
                    # temp_temp=[]
                    for n in range(len(temp)):
                        if not (temp[n] in inter_so_far):
                            cur_list.append(temp[n])
                            inter_so_far.append(temp[n])
                            break
                    # if len(temp_temp)>0:
                    #     cur_list.append(temp_temp)
                    # logging.error('CUR LIST ' + str(cur_list))
            
            for n in range(pad_size - len(cur_list)):
                cur_list.append(-1)
                
            gt_intersection_list.append(cur_list)
        
        return np.array(gt_intersection_list)
        
    def get_polygons(self, img_centers, roads):
        
        n_points = 300
                
       
        
        '''
        BUILD BOUNDARY LINES
        '''
        upper = np.stack([np.linspace(0,1,n_points), np.zeros((n_points))],axis=-1)
        lower = np.stack([np.linspace(0,1,n_points), np.ones((n_points))],axis=-1)
        left = np.stack([ np.zeros((n_points)),np.linspace(0,1,n_points)],axis=-1)
        right = np.stack([np.ones((n_points)),np.linspace(0,1,n_points)],axis=-1)
        
        
        # bev_mask_path = 'C:\\winpy\\WPy64-3850\\codes\\simplice-net\\real_labels_4_batch_0_class_6.png'
        # bev_mask = np.array(Image.open(bev_mask_path), np.uint8)
        
        bev_left = np.stack([np.linspace(0,0.54,n_points), np.linspace(30/200,1,n_points)],axis=-1)
        
        bev_right = np.stack([np.linspace(0.46,1,n_points), np.linspace(1,30/200,n_points)],axis=-1)
        
        boundaries = np.stack([upper,  left, right, bev_left, bev_right, lower],axis=0)
        
        start_ind = np.max(roads) + 1
        
        roads = np.concatenate([roads,np.max(roads) + 1 + np.arange(6)],axis=0)
        
        
        
        img_centers[0,:] = start_ind
        img_centers[:,0] = start_ind + 1
        img_centers[:,-1] = start_ind + 2   
        
        bev_left_line = my_line_maker(bev_left) > 0
        img_centers[bev_left_line] = start_ind + 3
        bev_right_line = my_line_maker(bev_right) > 0
        
        img_centers[bev_right_line] = start_ind + 4

        img_centers[-1,:] = start_ind+5
        
        inter = 3*np.ones((len(roads), n_points, 2))
        for k in range(len(roads)):
            sel = img_centers == roads[k]
            
            locs = np.where(sel)
            
            sorted_x = locs[1]/img_centers.shape[1]
            sorted_y = locs[0]/img_centers.shape[0]
            
            inter[k, :len(sorted_x), 0] = sorted_x
            inter[k, :len(sorted_y), 1] = sorted_y
            
        
        grid_x = np.linspace(0,1,200)
        grid_y = np.linspace(0,1,196)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        
        road_mat = np.clip(img_centers,0,1)
        
        road_mat = np.uint8(1-road_mat)
        
        dist_trans = cv2.distanceTransform(road_mat,cv2.DIST_L2,5)
        
        # logging.error('DISTANCE TRANSFORM OKAY')
        
        thresh_dist = dist_trans > 2
        
        blob_mat, num_blobs = measure.label(thresh_dist, connectivity=2, return_num=True)
        
        one_hots = []
        
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
                
                # print(str(o_h))
                # plt.imshow((blob_mat == (k+1)) & thresh_dist)
                # o_h = np.concatenate([o_h[:len(orig_inter)+1], o_h[len(orig_inter)+2:])
                one_hots.append(np.copy(o_h))
                m_d = dist_trans * ((blob_mat == (k+1)) & thresh_dist)
                m_loc = np.argmax(m_d)
                x_center = mesh_x.flatten()[m_loc]
                y_center = mesh_y.flatten()[m_loc]
                my_blob_center = np.array([x_center, y_center])
                all_centers.append(np.copy(my_blob_center))
                
        
        
                
        return np.array(all_centers), np.array(one_hots), blob_mat, np.array(blob_ids)
    def load_image(self, log_id, timestamp):
        
        # Load image
#        loader = self.loader
#        image_file = loader._timestamp_image_dict[log_id][self.camera][timestamp]
        image_file = os.path.join(self.argoverse_data_root,log_id,'ring_front_center','ring_front_center_'+str(timestamp)+'.jpg')
    
        image = Image.open(image_file)
        
        
        image = np.array(image,np.float32)
        
        if self.pinet:
            image = cv2.resize(image, (512,256), cv2.INTER_LINEAR)[...,[2,1,0]]
        else:
            image = cv2.resize(image, (self.config.patch_size[0], self.config.patch_size[1]), cv2.INTER_LINEAR)
            image = np.float32(image)
            image = self.minmax_normalize(image, norm_range=(-1, 1))
        


        return to_tensor(image).float()
    
    def minmax_normalize(self,img, norm_range=(0, 1), orig_range=(0, 255)):
        # range(0, 1)
        norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
        # range(min_value, max_value)
        norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
        return norm_img
    def load_calib(self, log):

        # Get the loader for the current split
        loader = self.loader

        # Get intrinsics matrix and rescale to account for downsampling
        calib = np.copy(loader.get_calibration(self.camera, log).K[:,:3])
        calib[0] *= self.image_size[0] / IMAGE_WIDTH
        calib[1] *= self.image_size[1] / IMAGE_HEIGHT
        
        # Convert to a torch tensor
        return torch.from_numpy(calib)
    

    def load_labels(self, split, log, camera, timestamp):

        # Construct label path from example data
        label_path = os.path.join(self.label_root, split, log, camera, 
                                  timestamp, f'{camera}_{timestamp}.png')
        
        # Load encoded label image as a torch tensor
        encoded_labels = to_tensor(Image.open(label_path)).long()

        # Decode to binary labels
        num_class = len(ARGOVERSE_CLASS_NAMES)
        labels = decode_binary_labels(encoded_labels, num_class+ 1)
        labels, mask = labels[:-1], ~labels[-1]

        return labels, mask
      
    
    def line_endpoints(self, coeffs, inc, out, roads):
        try:
            roads = list(roads)
            new_coeffs = np.copy(np.array(coeffs))
            for k in range(len(coeffs)):
                if len(inc[k]) > 0:
                    other = roads.index(inc[k][0])
                    other_coef = coeffs[other]
                    
                    dist1 = np.sum(np.abs(new_coeffs[k,0] - other_coef[0]))
                    dist2 = np.sum(np.abs(new_coeffs[k,-1] - other_coef[0]))
                    dist3 = np.sum(np.abs(new_coeffs[k,0] - other_coef[-1]))
                    dist4 = np.sum(np.abs(new_coeffs[k,-1] - other_coef[-1]))
                    
                    min_one = np.squeeze(np.argmin(np.stack([dist1,dist2,dist3,dist4])))
                    
                    if min_one == 0:
                        temp = np.copy(new_coeffs[other,0])
                        new_coeffs[other,0] = new_coeffs[other,-1]
                        new_coeffs[other,-1] = temp
                    elif min_one == 1:
                        temp = np.copy(new_coeffs[other,0])
                        new_coeffs[other,0] = new_coeffs[other,-1]
                        new_coeffs[other,-1] = temp
                        
                        temp = np.copy(new_coeffs[k,0])
                        new_coeffs[k,0] = new_coeffs[k,-1]
                        new_coeffs[k,-1] = temp
                    elif min_one == 3:
                        temp = np.copy(new_coeffs[k,0])
                        new_coeffs[k,0] = new_coeffs[k,-1]
                        new_coeffs[k,-1] = temp
                
                if len(out[k]) > 0:
                    other = roads.index(out[k][0])
                    other_coef = coeffs[other]
                    
                    dist1 = np.sum(np.abs(new_coeffs[k,0] - other_coef[0]))
                    dist2 = np.sum(np.abs(new_coeffs[k,-1] - other_coef[0]))
                    dist3 = np.sum(np.abs(new_coeffs[k,0] - other_coef[-1]))
                    dist4 = np.sum(np.abs(new_coeffs[k,-1] - other_coef[-1]))
                    
                    min_one = np.squeeze(np.argmin(np.stack([dist1,dist2,dist3,dist4])))
                    
                    if min_one == 0:
                        temp = np.copy(new_coeffs[k,0])
                        new_coeffs[k,0] = new_coeffs[k,-1]
                        new_coeffs[k,-1] = temp
                
                        
                    elif min_one == 2:
                        temp = np.copy(new_coeffs[other,0])
                        new_coeffs[other,0] = new_coeffs[other,-1]
                        new_coeffs[other,-1] = temp
                        
                        temp = np.copy(new_coeffs[k,0])
                        new_coeffs[k,0] = new_coeffs[k,-1]
                        new_coeffs[k,-1] = temp
                    elif min_one == 3:
                        temp = np.copy(new_coeffs[other,0])
                        new_coeffs[other,0] = new_coeffs[other,-1]
                        new_coeffs[other,-1] = temp
                        
            return new_coeffs
        except Exception as e:
            logging.error('ENDPOINTS ' + str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
            return coeffs
        
#        inc, out  = self.get_line_orientation(k, roads, all_selected,selected_pred[k],selected_suc[k],selected_id)
#        
        
    def get_line_orientation(self, road, all_roads, all_selected,selected_pred,selected_suc,selected_id):
        try:
#            my_gt_id = selected_id[road]
#            road_id=all_roads[road]
            outgoing_id = []
            for tok in selected_suc:
     #                logging.error('OUTGOING ' + tok)
     
                 if tok in selected_id:
     
                     outgoing_id.append(all_selected[selected_id.index(tok)])
     
            incoming_id = []
            for tok in selected_pred:
     #                logging.error('INCOMING ' + tok)
                 if tok in selected_id:
     
                     incoming_id.append(all_selected[selected_id.index(tok)])
            
            return incoming_id, outgoing_id
        except Exception as e:
            logging.error('ORIENT ' + str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
            return [],[]

    def get_connectivity(self,roads,outgoings, incomings):
        try:
            con_matrix = np.zeros((len(roads),len(roads)))
            # logging.error('CON ROAD ' + str(roads))
            for k in range(len(roads)):
                
                con_matrix[k,k] = 0
                outs = outgoings[k]
                # logging.error('CON OUTS ' + str(outs))
                for ou in outs:
                    
                    
                    sel = ou 
                    if sel in roads:
                        
                        ind = roads.index(sel)
                        # logging.error('INCOM ' + str(incomings[ind]))
                        # if not (ou in incomings[ind]):
                        #     logging.error('OUT HAS NO IN')
                 
                    
                        con_matrix[k,ind] = 1
                    
            return con_matrix
        
        except Exception as e:
            
            logging.error('CONNECT ' + str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
            return None
    

        
    def get_sorted_init_points(self, points):
        
        '''
        FROM BOTTOM UP AND RIGHT TO LEFT
        '''
        x = points[:,0]*50
        y = points[:,1]*49
        
        place = 50*x + y
        
        sort_ind = np.argsort(place)
        
        sort_ind = np.flip(sort_ind)
        
        return points[sort_ind,:], sort_ind
        
    def get_visible_mask(self, instrinsics, image_width, extents, resolution):

        # Get calibration parameters
        fu, cu = instrinsics[0, 0], instrinsics[0, 2]
    
        # Construct a grid of image coordinates
        x1, z1, x2, z2 = extents
        x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
        ucoords = x / z[:, None] * fu + cu
    
        # Return all points which lie within the camera bounds
        return (ucoords >= 0) & (ucoords < image_width)

    
    def load_line_labels(self,timestamp,  log_id, ind):
        try:
            am = self.am
            camera = "ring_front_center"
            loader = self.loader
#          
            log = loader.get(log_id)
            
            ego_loc = np.copy(log.get_pose(ind).translation)
            
            query_x = ego_loc[0]
            query_y = ego_loc[1]
            
            city_SE3_egovehicle_mat = np.copy(log.get_pose(ind).rotation)
             
            transform_matrix = np.eye(3)
            transform_matrix[:2, :2] = city_SE3_egovehicle_mat[:2,:2]
#         
            calib = log.get_calibration("ring_front_center")
            
            city_name = log.city_name
#           
            lane_ids = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, 80.0)
            
            
            local_centerlines = []
            successors = []
            predecessors = []
            for lane_id in lane_ids:
                my_lane = am.city_lane_centerlines_dict[city_name][lane_id]
                
                predecessors.append(my_lane.predecessors)
                successors.append(my_lane.successors)
                
                centerl = my_lane.centerline
                if len(centerl[0]) == 2:
                    centerl = am.append_height_to_2d_city_pt_cloud(centerl, city_name)
                local_centerlines.append(centerl)

#         
            
            my_area = np.zeros((196,200))
            
            inter = 5
            vis_mask = np.copy(np.uint8(np.flipud(self.get_visible_mask(np.copy(calib.K), np.copy(calib.camera_config.img_width),
                                       self.config.map_extents, self.config.resolution))))
            
            temp_vis_mask = np.flipud(vis_mask)
            converted_lines = []
            all_selected = []
            
            selected_pred = []
            selected_suc = []
            selected_id = []
            
            
            for centerline_city_fr_id in range(len(lane_ids)):
                
                centerline_city_fr = local_centerlines[centerline_city_fr_id]
                
                ground_heights = am.get_ground_height_at_xy(centerline_city_fr, city_name)
        
                valid_idx = np.isnan(ground_heights)
                centerline_city_fr = centerline_city_fr[~valid_idx]
                
                centerline_egovehicle_fr = (centerline_city_fr - ego_loc) @ city_SE3_egovehicle_mat 
#           
                
                centerline_egovehicle_fr = centerline_egovehicle_fr[:,:2]
                centerline_egovehicle_fr[:,1] = -centerline_egovehicle_fr[:,1]+25
                centerline_egovehicle_fr[:,0] = centerline_egovehicle_fr[:,0]-1
                
                centerline_egovehicle_fr = centerline_egovehicle_fr*4
    #            centerline_uv_cam = cam_SE3_egovehicle.transform_point_cloud(centerline_egovehicle_fr)
                counter = 0
                to_draw = []
                to_sel_id = []
                for k in range(len(centerline_egovehicle_fr)):
                    
                    
                    
                    if ((centerline_egovehicle_fr[k][0] >= 0) & (centerline_egovehicle_fr[k][0] <= 195) & (centerline_egovehicle_fr[k][1] >= 0) & (centerline_egovehicle_fr[k][1] <= 199)):
                        if temp_vis_mask[int(np.clip(centerline_egovehicle_fr[k][0], 0, 195)), int(np.clip(centerline_egovehicle_fr[k][1], 0, 199))]>0:
                        
                            # to_draw.append((int(np.clip(centerline_egovehicle_fr[k][0], 0, 195)), int(np.clip(centerline_egovehicle_fr[k][1], 0, 199))))
                            to_draw.append((np.clip(centerline_egovehicle_fr[k][0], 0, 195), np.clip(centerline_egovehicle_fr[k][1], 0, 199)))
                            
                            to_sel_id.append(k)        
                            counter = counter + 1
                        
                if counter >= 3:
                    
                    for k in range(len(to_draw)-1):
                        for m in range(inter):
                            
                            my_area[int((m/inter)*to_draw[k][0] + (1 - m/inter)*to_draw[k+1][0]), int((m/inter)*to_draw[k][1] + (1 - m/inter)*to_draw[k+1][1])] = centerline_city_fr_id + 1
                    my_area[int(to_draw[0][0]), int(to_draw[0][1])] = centerline_city_fr_id + 1
                    my_area[int(to_draw[-1][0]), int(to_draw[-1][1])] = centerline_city_fr_id + 1
                    
#                    converted_lines.append(centerline_egovehicle_fr[np.array(to_sel_id)])
                    converted_lines.append(np.array(to_draw))
                    all_selected.append(centerline_city_fr_id + 1)
                    
                    
                    selected_pred.append(predecessors[centerline_city_fr_id])
                    selected_suc.append(successors[centerline_city_fr_id])
                    selected_id.append(lane_ids[centerline_city_fr_id])
                    
            # logging.error('CONVERTED ' + str(len(converted_lines)))
            
            
            my_area = np.flipud(my_area)
            
            vis_labels = np.stack([vis_mask,vis_mask],axis=0)
            
            orig_img_centers = my_area
            img_centers = orig_img_centers*np.uint16(vis_mask)
            
            
            
            roads = np.unique(img_centers)[1:]
            
            
            # logging.error('ROADS ' + str(roads))
            
            
            if len(converted_lines) < 1:
                logging.error('NOT ENOUGH CONVERTED LINES')
                return None,None,None,None,None,None, None,\
            None,None,None,None,\
           True
           
            new_to_feed = []
            for k in range(len(roads)):
                new_to_feed.append(converted_lines[all_selected.index(roads[k])])
            
            outgoings = []
            incomings = []
            coeffs_list = []
            endpoints=[]
            origs = []
    #        starts = []
            
            to_remove = []
            for k in range(len(roads)):
                
                sel = img_centers == roads[k]
                
                
                
                locs = np.where(sel)
#              
                sorted_x = locs[1]/img_centers.shape[1]
                sorted_y = locs[0]/img_centers.shape[0]
#               
                
                if len(sorted_x) < 3:
                    to_remove.append(roads[k])   
                    continue
                
                
                points = np.array(list(zip(sorted_x,sorted_y)))
#              
                res = bezier.fit_bezier(points, self.n_control)[0]
                
                
#                
                inc, out  = self.get_line_orientation(k, roads, all_selected,selected_pred[k],selected_suc[k],selected_id)
#                
                outgoings.append(out)
                incomings.append(inc)

                fin_res = res
                fin_res[0][0] = converted_lines[k][0][1]/200
                fin_res[0][1] = 1 - converted_lines[k][0][0]/196
                fin_res[-1][0] = converted_lines[k][-1][1]/200
                fin_res[-1][1] = 1 - converted_lines[k][-1][0]/196
                
                endpoints.append(np.stack([fin_res[0], fin_res[-1]],axis=0))

                fin_res = np.clip(fin_res,0,1)
                coeffs_list.append(np.reshape(np.float32(fin_res),(-1)))
        #       
                sel = np.float32(sel)
            
                origs.append(sel)
            
            if len(to_remove) > 0:
#            logging.error('TO REMOVE ' + str(to_remove))
                roads = list(roads)
                
                for k in to_remove:
                    img_centers[img_centers == k] = 0
                    
                    roads.remove(k)
                    
    #            roads = list(set(roads) - set(to_remove))
            
            else:
                roads = list(roads)
            
            if len(coeffs_list) == 0:
                logging.error('COEFFS ENPTY')
                return None,None,None,None,None,None, None,\
            None,None,None,None,\
           True
            
            con_matrix = self.get_connectivity(roads,outgoings, incomings)
    #        if con_matrix==None:
                
            to_return_centers = torch.tensor(np.int64(img_centers)).long()
            orig_img_centers = torch.tensor(np.int64(orig_img_centers)).long()
            labels = torch.ones(len(roads))
            
            endpoints = np.array(endpoints)
            return con_matrix,np.reshape(endpoints,(endpoints.shape[0],-1)),\
        orig_img_centers,np.stack(origs),\
        vis_labels,\
        to_return_centers,labels, roads,\
        np.array(coeffs_list), outgoings, incomings, False

        except Exception as e:
            logging.error('LINE LABELS ' + str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
            return None,None,None,None,None,None, None,\
                None,None,None,None,\
               True
        



def my_line_maker(points,size=(196,200)):
    
    res = np.zeros(size)
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    return np.uint8(res)


    