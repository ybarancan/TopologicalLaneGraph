import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from nuscenes import NuScenes
from torchvision.transforms.functional import to_tensor
import cv2
from .utils import CAMERA_NAMES, NUSCENES_CLASS_NAMES, iterate_samples
from ..utils import decode_binary_labels
from src.utils import bezier
import random
from src.data.nuscenes import utils as nusc_utils
from src.data import utils as vis_utils
from skimage import measure
#from scipy.ndimage import gaussian_filter
import sys
import numpy as np
import scipy.interpolate as si 
# from scipy.interpolate import UnivariateSpline
import logging
# import pwlf
from math import factorial
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']


class NuScenesMapDataset(Dataset):

    def __init__(self, nuscenes, config,map_apis,
                 scene_names=None, pinet=False, val=False,  polyline=False, gt_polygon_extraction=False):
        
        
        self.pinet=pinet
        
        self.config = config
        
        self.gt_polygon_extraction = gt_polygon_extraction
        
        self.nuscenes = nuscenes
        self.map_root = os.path.expandvars(config.nusc_root)
        
        self.line_label_root = self.config.line_label_root
        self.seg_label_root = self.config.seg_label_root
        
        self.static_label_root = self.config.static_label_root
        
        self.resolution = self.config.map_resolution
        self.extents = self.config.map_extents
        
        self.n_control = config.n_control_points
        
        if self.pinet:
            self.image_size = [512,256]
        self.image_size = config.patch_size

        # Preload the list of tokens in the dataset
        self.get_tokens(scene_names)

        self.map_apis = map_apis
        
        self.lines_dict = {}
        
        self.loc_dict = np.load(config.loc_dict_path,allow_pickle=True)
        
        self.obj_dict = np.load(config.obj_dict_path,allow_pickle=True)
        
        self.camera_matrix_dict = np.load(config.cam_matrix_dict_path, allow_pickle=True)   
        
        self.intrinsic_dict = np.load(config.intrinsic_dict_path, allow_pickle=True)   
        
        
        self.zoom_sampling_dict = np.load(config.zoom_sampling_dict_path, allow_pickle=True)   
       
        logging.error('TOTAL TOKENS ' + str(len(self.tokens)))
        self.val = val
        
        
        if not self.gt_polygon_extraction:
            if val:
           
                    
                base_path = config.poly_base_path
                
                all_paths = glob.glob(os.path.join(base_path,'val_gt_polygon*'))
                
                dicts_list = []
                for pa in all_paths:
                    dicts_list.append(np.load(pa, allow_pickle=True).item())
                
                self.gt_poly_dict = dict()
                for di in dicts_list:
                    self.gt_poly_dict.update(di)
            
        
     
            else:
                
                
                base_path = config.poly_base_path
                
               
                
                all_paths = glob.glob(os.path.join(base_path,'gt_polygon*'))
                
                dicts_list = []
                for pa in all_paths:
                    dicts_list.append(np.load(pa, allow_pickle=True).item())
                
                self.gt_poly_dict = dict()
                for di in dicts_list:
                    self.gt_poly_dict.update(di)
           
        self.seg_label_root = self.config.seg_label_root
        self.vis_label_root = self.config.vis_label_root

        for location in LOCATIONS:
            
            scene_map_api = self.map_apis[location]
            all_lines = scene_map_api.lane + scene_map_api.lane_connector
            all_lines_tokens = []
            for lin in all_lines:
                all_lines_tokens.append(lin['token'])
            
            self.lines_dict[location] = all_lines_tokens
            
    
        self.all_discretized_centers = {location : self.map_apis[location].discretize_centerlines(0.25)
                 for location in nusc_utils.LOCATIONS}
        

        # Allow PIL to load partially corrupted images
        # (otherwise training crashes at the most inconvenient possible times!)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        self.struct = ndimage.generate_binary_structure(2, 1)
        
        
        self.augment_steps=[0.5,1,1.5,2]
        
        
        
    
    def minmax_normalize(self,img, norm_range=(0, 1), orig_range=(0, 255)):
        # range(0, 1)
        norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
        # range(min_value, max_value)
        norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
        return norm_img

    def get_tokens(self, scene_names=None):
        
        self.tokens = list()
        
        # all_files = glob.glob(os.path.join(self.line_label_root,'*png'))
        # self.tokens = [file.split('/')[-1][:-4] for file in all_files]
        # Iterate over scenes
        for scene in self.nuscenes.scene:
            
            # Ignore scenes which don't belong to the current split
            if scene_names is not None and scene['name'] not in scene_names:
                continue
             
            # Iterate over samples
            for sample in iterate_samples(self.nuscenes, 
                                          scene['first_sample_token']):
                
                # Iterate over cameras
                for camera in CAMERA_NAMES:
                    self.tokens.append(sample['data'][camera])
        
        return self.tokens


    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        
        try:
            token = self.tokens[index]
            
            
     
            image = self.load_image(token, False, None)
            beta=0
            augment=False
            
            # logging.error('IMAGE OBTAINED')
            calib = self.load_calib(token)
            con_matrix,endpoints,mask, occ_labels, orig_img_centers,\
            origs, dilated, smoothed,scene_token,sample_token,to_return_centers, labels,roads,coeffs,\
            outgoings, incomings, singapore,problem = self.load_line_labels(token, augment, beta)
            # logging.error('GOT EVERYTHING')
            
            if problem:
                return (None, dict(), True)
            
           
            scene_name = self.nuscenes.get('scene', scene_token)['name']
            
            
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
            
            start_con_matrix, fin_con_matrix = self.get_start_fin_con(endpoints)
            target = dict()
#            try:
#                
#                if self.gt_polygon_extraction:
#                    poly_centers, poly_one_hots, blob_mat, blob_ids = self.get_polygons(orig_img_centers, roads)
#                
#                else:
#                    blob_mat, poly_centers, poly_one_hots, blob_ids = self.gt_poly_dict[token]
#                
#                if np.any(blob_mat == None):
#                    return (None, dict(), True)
#                
#
#                
#            except Exception as e:
#                logging.error('GT POLYGON EXCEPTION : ' + str(e))
#                
         
            
            intersections = self.get_order_labels(coeffs)
            
            target['gt_order_labels'] = torch.tensor(intersections).long()
            
#            target['blob_ids'] = torch.tensor(blob_ids).long()
#            target['blob_mat'] = torch.tensor(blob_mat).long()
#            target['poly_one_hots'] = torch.tensor(poly_one_hots).float()
#            target['poly_centers'] = torch.tensor(poly_centers).float()
            
            target['blob_ids'] = torch.tensor(0).long()
            target['blob_mat'] = torch.tensor(0).long()
            target['poly_one_hots'] =  torch.tensor(0).long()
            target['poly_centers'] =  torch.tensor(0).long()
            
            target['calib'] = calib
            target['center_img'] = to_return_centers
            target['orig_center_img'] = orig_img_centers
            target['labels'] = labels.long()
            target['roads'] = torch.tensor(np.int64(roads)).long()
            target['control_points'] = torch.tensor(coeffs)
            target['con_matrix'] = torch.tensor(con_matrix)
            
            target['start_con_matrix'] = torch.tensor(start_con_matrix)
            target['fin_con_matrix'] = torch.tensor(fin_con_matrix)

            
            target['init_point_matrix'] = torch.tensor(np.copy(np.ascontiguousarray(temp_ar)))
            
            target['sorted_control_points'] = torch.tensor(sorted_points)
            
            
            target['grid_sorted_control_points'] = torch.tensor(my_grid_points)
            
            target['sort_index'] = torch.tensor(np.copy(np.ascontiguousarray(sort_index)))
          
            target['endpoints'] = torch.tensor(endpoints)

            target['origs'] = torch.tensor(origs)
            target['smoothed'] = torch.tensor(smoothed)
            target['dilated'] = torch.tensor(dilated)
            target['mask'] = mask
            target['occ_mask'] = occ_labels
          
            target['scene_token'] = scene_token
            target['sample_token'] = sample_token
            target['data_token'] = token
            target['scene_name'] = scene_name
            target['outgoings'] = outgoings
            target['incomings'] = incomings
            target['left_traffic'] = torch.tensor(singapore)
            return (image, target, False)
        
        except Exception as e:
            logging.error('NUSC DATALOADER ' + str(e))
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
            # print(uniques)

                
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

    def load_seg_labels(self, token, scene_token):
        
        # Load label image as a torch tensor
        label_path = os.path.join(self.vis_label_root, token + '.png')
        
        ar = np.array(Image.open(label_path))
        
        ar = np.flipud(ar)
        
        encoded_labels = torch.tensor(ar.copy()).long()

        # Decode to binary labels
        # num_class = len(NUSCENES_CLASS_NAMES)
        vis_labels = decode_binary_labels(encoded_labels, 2)
        # labels, mask = labels[:-1], ~labels[-1]
        
       
        return vis_labels
    
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
        
    
    def load_image(self, token, augment, temp_ar):

        # Load image as a PIL image
        image = Image.open(self.nuscenes.get_sample_data_path(token))
        image = np.array(image,np.float32)
        
        if augment:
            
            write_row = np.reshape(temp_ar[...,0],(image.shape[0],image.shape[1]))
            
            write_col = np.reshape(temp_ar[...,1],(image.shape[0],image.shape[1]))
            
            total_mask = np.reshape(temp_ar[...,2],(image.shape[0],image.shape[1]))
            
            
            
            sampled = image[write_row.flatten(),write_col.flatten(),:]
    
            sampled = np.reshape(sampled,[image.shape[0],image.shape[1],3])
            
            sampled = sampled*np.stack([total_mask,total_mask,total_mask],axis=-1) + image*(1-np.stack([total_mask,total_mask,total_mask],axis=-1))
            
            image=sampled
            
        if self.pinet:
            image = cv2.resize(image, (512,256), cv2.INTER_LINEAR)[...,[2,1,0]]
        else:
            image = cv2.resize(image, (self.config.patch_size[0], self.config.patch_size[1]), cv2.INTER_LINEAR)
            image = np.float32(image)
            image = self.minmax_normalize(image, norm_range=(-1, 1))
        


        # Resize to input resolution
        
        
        # Convert to a torch tensor
        return to_tensor(image).float()
    

    def load_calib(self, token):

        # Load camera intrinsics matrix
        sample_data = self.nuscenes.get('sample_data', token)
        sensor = self.nuscenes.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = torch.tensor(sensor['camera_intrinsic'])

        # Scale calibration matrix to account for image downsampling
        intrinsics[0] *= self.image_size[0] / sample_data['width']
        intrinsics[1] *= self.image_size[1] / sample_data['height']
        return intrinsics
    

    def  get_line_orientation(self,sample_token, road,img_centers,loc,vis_mask, custom_endpoints=None, augment=False):
        
        try:
            scene_map_api = self.map_apis[loc]
            all_lines_tokens = self.lines_dict[loc]
            
            other_lines_tokens = []
            for l in self.lines_dict.keys():
                other_lines_tokens = other_lines_tokens + self.lines_dict[l]
            
            if augment:
                all_ends = custom_endpoints
            else:
                all_ends = self.loc_dict.item().get(sample_token)
            
            # my_ends = 
            
            my_row_id = img_centers.shape[0] - all_ends[road-1,:,0] - 1
            my_col_id = all_ends[road-1,:,1]
            
            my_rows = np.float32(my_row_id)/img_centers.shape[0]
            my_cols = np.float32(my_col_id)/img_centers.shape[1]
            
            
            
#            
#            start_row = np.float32(all_ends[road-1,0,0])/img_centers.shape[0]
#            start_col = np.float32(all_ends[road-1,0,1])/img_centers.shape[1]
#            
#            end_row = np.float32(all_ends[road-1,1,0])/img_centers.shape[0]
#            end_col = np.float32(all_ends[road-1,1,1])/img_centers.shape[1]
            
            token = all_lines_tokens[road-1]
     #            logging.error('TOKEN ' + token)
            outgoing_token = scene_map_api.get_outgoing_lane_ids(token)
            outgoing_id = []
            for tok in outgoing_token:
     #                logging.error('OUTGOING ' + tok)
     
                 if tok in all_lines_tokens:
     
                     outgoing_id.append(all_lines_tokens.index(tok))
     
                 else:
#                     logging.error('LINE ' + tok + ' not in lines')
                     if tok in other_lines_tokens:
                         logging.error('LINE ' + tok + ' is in other map')
                     else:
                         logging.error('LINE ' + tok + ' doesnt exist')
        
            incoming_token = scene_map_api.get_incoming_lane_ids(token)
            incoming_id = []
            for tok in incoming_token:
     #                logging.error('INCOMING ' + tok)
                 if tok in all_lines_tokens:
     
                     incoming_id.append(all_lines_tokens.index(tok))
     
                 else:
#                     logging.error('LINE ' + tok + ' not in lines')
                     if tok in other_lines_tokens:
                         logging.error('LINE ' + tok + ' is in other map')
                     else:
                         logging.error('LINE ' + tok + ' doesnt exist')
                
            return incoming_id, outgoing_id, np.stack([my_cols,my_rows],axis=-1),np.stack([my_col_id,my_row_id],axis=-1)
    
        except Exception as e:
            logging.error('ORIENT ' + str(e))
            
            return [],[],[],[]
        
        
    def get_start_fin_con(self,endpoints):
        
   
        start_matrix = np.zeros((len(endpoints),len(endpoints)))
        fin_matrix = np.zeros((len(endpoints),len(endpoints)))
        # logging.error('CON ROAD ' + str(roads))
        for k in range(len(endpoints)):
            
            start_dist = np.sum(np.abs(endpoints[k:k+1,:2] - endpoints[:,:2]),axis=-1) 
            
            start_matrix[k,:] = np.float32(start_dist < 0.01)
            
            end_dist = np.sum(np.abs(endpoints[k:k+1,2:] - endpoints[:,2:]),axis=-1) 
            
            fin_matrix[k,:] = np.float32(end_dist < 0.01)
            
            
            start_matrix[k,k] = 1
            fin_matrix[k,k] = 1
            
            
        return start_matrix, fin_matrix
  
    
    def get_connectivity(self,roads,outgoings, incomings):
        try:
            con_matrix = np.zeros((len(roads),len(roads)))
            # logging.error('CON ROAD ' + str(roads))
            for k in range(len(roads)):
                
                con_matrix[k,k] = 0
                outs = outgoings[k]
                # logging.error('CON OUTS ' + str(outs))
                for ou in outs:
                    
                    
                    sel = ou + 1
                    if sel in roads:
                        
                        ind = roads.index(sel)
                        # logging.error('INCOM ' + str(incomings[ind]))
                        # if not (ou in incomings[ind]):
                        #     logging.error('OUT HAS NO IN')
                 
                    
                        con_matrix[k,ind] = 1
                    
            return con_matrix
        
        except Exception as e:
            logging.error('CONNECT ' + str(e))
            return None
    
    
    
    def load_line_labels(self, token, augment, beta):
        
        try:
    
            sample_data = self.nuscenes.get('sample_data', token)
            sensor = self.nuscenes.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
            intrinsics = np.array(sensor['camera_intrinsic'])
        
            sample_token = sample_data['sample_token']
    #        logging.error('SAMPLE TOKEN ' + sample_token)
            sample = self.nuscenes.get('sample', sample_token)
            scene_token = sample['scene_token']
            
    #        logging.error('SCENE TOKEN ' + scene_token)
            scene = self.nuscenes.get('scene', scene_token)
    #        logging.error('SCENE OBTAINED')
            log = self.nuscenes.get('log', scene['log_token'])
            
    #        scene_map_api = self.map_apis[log['location']]
            # Load label image as a torch tensor
            label_path = os.path.join(self.line_label_root, token + '.png')
            # logging.error('PATH ' + str(label_path))
            orig_img_centers = cv2.imread(label_path, cv2.IMREAD_UNCHANGED )
            
            # logging.error('IMG CENTERS '+ str(img_centers.shape) + ' , ' + str(img_centers.dtype))
            # logging.error('ROADS '+ str(np.unique(img_centers)))
    #
            
            
            
    #        occ_labels = self.load_seg_labels(token, scene_token)
            # vis_labels, bev_labels, static_labels = self.load_seg_labels(token, scene_token)
            # np_mask = vis_labels.numpy()
            
            # vis_mask = np_mask[0]
            # vis_mask = np.array(Image.open('/cluster/home/cany/simplice-net/bev_mask.png'),np.uint8)
            # vis_mask = np.float32(vis_mask > 0)
            # vis_labels = np.stack([vis_mask, vis_mask],axis=0)
            # vis_labels = torch.tensor(vis_labels)
            
            vis_mask = vis_utils.get_visible_mask(intrinsics, sample_data['width'], 
                                      self.config.map_extents, self.config.map_resolution)
            vis_mask = np.flipud(vis_mask)
            vis_labels = np.stack([vis_mask, vis_mask],axis=0)
            vis_labels = torch.tensor(vis_labels)
            
            occ_labels = torch.tensor(np.zeros_like(vis_mask))
            
            # tot_mask = (1-occ_mask)*vis_mask
            
            
           
            
            if augment:
                
                orig_img_centers, trans_endpoints = nusc_utils.get_moved_centerlines(self.nuscenes, self.all_discretized_centers[log['location']], sample_data, self.extents, self.resolution, vis_mask,beta,orig_img_centers)
                orig_img_centers = np.flipud(orig_img_centers)
                
                # logging.error('AUGMENT PASSED ')
                
            # obj_to_return, center_width_orient, obj_exists = self.get_object_params(token,vis_mask,beta,augment)
            
            
            img_centers = orig_img_centers*np.uint16(vis_mask)
            
            roads = np.unique(img_centers)[1:]
        
    #        logging.error('ROADS ' + str(roads))
    #        all_lines_tokens = self.lines_dict[log['location']]
                
            outgoings = []
            incomings = []
            coeffs_list = []
            to_remove=[]
            dilated_list=[]
            smoothed = []
            origs = []
    #        starts = []
            endpoints = []
            
            
            singapore = 'singapore' in log['location']
            
            for k in range(len(roads)):
                
                sel = img_centers == roads[k]
                
                
                
                locs = np.where(sel)
                
                
                
                # logging.error('LOCS OBTAINED')
                
                # sorted_x, sorted_y = zip(*sorted(zip(list(locs[1]),list(locs[0]))))
                sorted_x = locs[1]/img_centers.shape[1]
                sorted_y = locs[0]/img_centers.shape[0]
                # my_x = []
                # my_y = []
                # for m in range(len(sorted_x)):
                #     if sorted_x[m] not in my_x:
                #         my_x.append(sorted_x[m]/img_centers.shape[1])
                #         my_y.append(sorted_y[m]/img_centers.shape[0])
                
                # logging.error('LOCS REMOVED DUPLICATES')
                
                if len(sorted_x) < 10:
                    to_remove.append(roads[k])   
                    continue
                
                if augment:
                    inc, out, endpoint, endpoint_id = self.get_line_orientation(token,roads[k],img_centers,log['location'],vis_mask,custom_endpoints=trans_endpoints, augment=True)
                else:
                    inc, out, endpoint, endpoint_id = self.get_line_orientation(token,roads[k],img_centers,log['location'],vis_mask,custom_endpoints=None, augment=False)
                
                if len(endpoint) == 0:
                    continue
                
                
                reshaped_endpoint = np.reshape(endpoint,(-1))
                endpoints.append(reshaped_endpoint)
                incomings.append(inc)
                outgoings.append(out)
                
                
                
                points = np.array(list(zip(sorted_x,sorted_y)))
                res = bezier.fit_bezier(points, self.n_control)[0]
                
    #            logging.error('RES ' + str(res))
    #            logging.error('ENDPOINT  ' + str(endpoint))
                
                
                start_res = res[0]
                end_res = res[-1]
                
    #            tol = 0.001
                
                
                first_diff = (np.sum(np.square(start_res - endpoint[0])) ) + (np.sum(np.square(end_res - endpoint[1])))
                second_diff = (np.sum(np.square(start_res - endpoint[1])) ) + (np.sum(np.square(end_res - endpoint[0])))
                if first_diff <= second_diff:
                    fin_res = res
                else:
                    fin_res = np.zeros_like(res)
                    for k in range(len(res)):
                        fin_res[len(res) - k - 1] = res[k]
                        
                fin_res = np.clip(fin_res,0,1)
                
    #            if (np.sum(np.square(start_res - endpoint[0])) <= tol) & (np.sum(np.square(end_res - endpoint[1])) <= tol):
    #                fin_res = res
    #            elif (np.sum(np.square(start_res - endpoint[1])) <= tol) & (np.sum(np.square(end_res - endpoint[0])) <= tol):
    #                fin_res = np.zeros_like(res)
    #                for k in range(len(res)):
    #                    fin_res[len(res) - k - 1] = res[k]
    #            elif (np.sum(np.square(end_res - endpoint[0])) < tol)& (np.sum(np.square(start_res - endpoint[1])) > tol):
    #                    logging.error('SOMETHING WRONG 1')
    #                    logging.error('RES ' + str(res))
    #                    logging.error('ENDPOINT  ' + str(endpoint))
    #                    fin_res = res
    #            elif (np.sum(np.square(start_res - endpoint[0])) > tol)& (np.sum(np.square(end_res - endpoint[1])) < tol):
    #                logging.error('SOMETHING WRONG 2')
    #                logging.error('RES ' + str(res))
    #                logging.error('ENDPOINT  ' + str(endpoint))
    #                fin_res = res
    #       
    #                
    #            else:
    #                logging.error('WTF')
    #                fin_res = res
                # logging.error('RES ' + str(res))
                # spl = UnivariateSpline(my_x, my_y)
                # coeffs = spl.get_knots()
                coeffs_list.append(np.reshape(np.float32(fin_res),(-1)))
            #    
                # logging.error('COEFFS OBTAINED ' + str(coeffs))
    #            logging.error('SEL ' + str(sel.shape))
                dilated = ndimage.binary_dilation(sel, structure=self.struct)
    #            logging.error('BASE DILATED ' + str(dilated.shape))
                dilated_list.append(np.copy(dilated))
                sel = np.float32(sel)
                gau = gaussian_filter(sel, sigma=2)
                
                smoothed.append(gau)
                
                origs.append(sel)
    #            sel = sel/np.max(sel)
                    
                
    #             token = all_lines_tokens[roads[k]-1]
    # #            logging.error('TOKEN ' + token)
    #             outgoing_token = scene_map_api.get_outgoing_lane_ids(token)
    #             outgoing_id = []
    #             for tok in outgoing_token:
    # #                logging.error('OUTGOING ' + tok)
    #                 outgoing_id.append(all_lines_tokens.index(tok))
                
    #             incoming_token = scene_map_api.get_incoming_lane_ids(token)
    #             incoming_id = []
    #             for tok in incoming_token:
    # #                logging.error('INCOMING ' + tok)
    #                 incoming_id.append(all_lines_tokens.index(tok))
                
                
    #             outgoings.append(np.array(outgoing_id))
    #             incomings.append(np.array(incoming_id))
                
    #        logging.error('TOM REMOVE')
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
                return None,None,None,None,\
            None,None,None,\
            None,None,None,\
            None,None,None,\
            None,None,None,True,True
    
            
            con_matrix = self.get_connectivity(roads,outgoings, incomings)
    #        if con_matrix==None:
                
            to_return_centers = torch.tensor(np.int64(img_centers)).long()
            orig_img_centers = torch.tensor(np.int64(orig_img_centers)).long()
            labels = torch.ones(len(roads))
            
    #        logging.error('DILATED LIST ' + str(len(dilated_list)))
    #        logging.error('DILATED LIST ELEMENT ' + str(dilated_list[0].shape))
    #        
        #     return obj_to_return, center_width_orient,con_matrix,np.array(endpoints),vis_labels,bev_labels,static_labels,\
        # orig_img_centers,np.stack(origs),np.stack(dilated_list),\
        # np.stack(smoothed),scene_token,sample_token,\
        # to_return_centers,labels, roads,\
        # np.array(coeffs_list), outgoings, incomings,singapore, False, obj_exists
    
            return con_matrix,np.array(endpoints),vis_labels,occ_labels,\
        orig_img_centers,np.stack(origs),np.stack(dilated_list),\
        np.stack(smoothed),scene_token,sample_token,\
        to_return_centers,labels, roads,\
        np.array(coeffs_list), outgoings, incomings,singapore, False
        
        except Exception as e:
            logging.error('LINE LOADING ' + str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(str((exc_type, fname, exc_tb.tb_lineno)))
       
            return None,None,None,None,\
            None,None,None,\
            None,None,None,\
            None,None,None,\
            None,None,None,True,True
        
            
def my_line_maker(points,size=(196,200)):
    
    res = np.zeros(size)
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    return np.uint8(res)
# x = [6, -2, 4, 6, 8, 14, 6]
# y = [-3, 2, 5, 0, 5, 2, -3]

# x = [0, 0, -0.5,-1 ,-1.5, -2, -2.5]
# y = [0, 0.5, 1, 1.5, 1.5, 1.5, 1.5]

# x = [0, 0, 0, 0, 0, 0, 0]
# y = [0, 0.5, 1, 1.5, 2, 2.5, 3]

# points = np.array(list(zip(x,y)))

# spl = si.UnivariateSpline(x, y)

# xmin, xmax = min(x), max(x) 
# ymin, ymax = min(y), max(y)

# n = len(x)
# plotpoints = 100

# k = 2
# knotspace = range(n)

# wend = 3
# w = [wend] + [1]*(n-2) + [wend]

# spline = si.UnivariateSpline(x, y, k=k , s=0.1, w=w)
# knots = spline.get_knots()
# knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))

# coeffs_x = getControlPoints(knots_full, k)
# coeffs_y = spline.get_coeffs()

# nsample = 100
# xP = np.linspace(x[0], x[-1], nsample)
# yP = spline(xP)



# num_points = len(x)
# num_knots = len(knots_full)
# ka = (knots_full[-1] - knots_full[0])/(num_points)
# knotsp = np.zeros(num_knots)
# for i in range(num_knots):
#     knotsp[i] = num_points - ((knots_full[-1] - knots_full[i]))/ka
    
    
# tckX = knotsp, coeffs_x, k
# tckY = knotsp, coeffs_y, k
# splineX = si.UnivariateSpline._from_tck(tckX)
# splineY = si.UnivariateSpline._from_tck(tckY)


# coeffs_p = getControlPoints(knotsp, k)

# tP = np.linspace(-3,  num_points+3, 100)
# xP = splineX(tP)
# yP = splineY(tP)


    
# xmin, xmax = min(x), max(x) 
# ymin, ymax = min(y), max(y)

# n = len(x)
# plotpoints = 100

# k = 1
# knotspace = np.linspace(0,n,4)
# knots = si.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
# knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))


# tckX = knots_full, x, k
# tckY = knots_full, y, k

# splineX = si.UnivariateSpline._from_tck(tckX)
# splineY = si.UnivariateSpline._from_tck(tckY)

# tP = np.linspace(knotspace[0], knotspace[-1], plotpoints)
# xP = splineX(tP)
# yP = splineY(tP)



# cp = getControlPoints(knots_full, k)

# plt.plot(xP,yP)


#nclass = 11
#grid_width = int((extents[2] - extents[0]) / resolution)
#grid_height = int((extents[3] - extents[1]) / resolution)
#masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)
#
#
#for obj in objs:
#    
#    
#    reshaped = np.reshape(obj[:8],(4,2))
#    reshaped[:,0] = 
#    
#    
#    # Render the rotated bounding box to the mask
#    render_polygon(masks[int(obj[-1])], np.reshape(obj[:8],(4,2)), extents, resolution)    


# token='8ce74f87c890440394505d215facdf96'
# label_path = '/srv/beegfs02/scratch/tracezuerich/data/cany/lanelines/segment/'+token+'.png'
# ar = Image.open(label_path)
#
# ar = np.flipud(ar)
#
# image = Image.open(nuscenes.get_sample_data_path(token))
# image = np.array(image,np.uint8)
#
# # Decode to binary labels
# # num_class = len(NUSCENES_CLASS_NAMES)
# vis_labels = numpy_decode_binary_labels(ar, 2)
# vis_mask = vis_labels[...,0]
#
# bev_label = np.array(Image.open( os.path.join('/srv/beegfs02/scratch/tracezuerich/data/cany/monomaps_labels_vanilla',  
#                                        token + '.png')),np.int32)
#        
#        
# bev_label = numpy_decode_binary_labels(bev_label,15)
# bev_label= np.flipud(bev_label)
#
# objs = obj_dict.item().get(token)
#    
# # logging.error('OBJS ' + str(objs.shape))
# #         map_extents: [-25., 1., 25., 50.]
#
# # # Spacing between adjacent grid cells in the map, in meters
# #         map_resolution: 0.25
#
# to_return=[]
# center_width_orient=[]
#
# all_objects_rendered = np.zeros((196,200))
#
# obj_exists = False
# for obj in objs:
#    
#     render_polygon(all_objects_rendered, np.reshape(obj[:8],(4,2)), extents, resolution)
#
#     reshaped = np.reshape(np.copy(obj)[:8],(4,2))
#     reshaped[:,0] = (reshaped[:,0] - map_extents[0])/(map_extents[2]-map_extents[0])
#    
#     reshaped[:,1] = 1 - (reshaped[:,1] - map_extents[1])/(map_extents[3]-map_extents[1])
#    
#    
#    
#     coords = (np.clip(np.int64(reshaped[:,1]*(map_extents[3]-map_extents[1])/resolution),0,195),
#               np.clip(np.int64(reshaped[:,0]*(map_extents[2]-map_extents[0])/resolution),0,199))
#    
#     # logging.error('COORDS ' + str(coords))
#    
#     inside = False
#     for k in range(4):
#         inside = inside | ((vis_mask[coords[0][k], coords[1][k]] > 0.5) & 
#                            ((reshaped[k,1] >= 0) & (reshaped[k,1] <= 1)) & 
#                            ((reshaped[k,0] >= 0) & (reshaped[k,0] <= 1)))
#            
#     if inside:
#        
#         # logging.error('INSIDE')
#         reshaped[:,1] = 1 - reshaped[:,1]
#         res_ar = np.zeros(5)
#        
#         temp=np.squeeze(np.zeros((9,1),np.float32))
#         temp[:8] = reshaped.flatten()
#         temp[-1] = obj[-1]
#         to_return.append(temp)
#            
#        
#         all_edges = np.zeros((4,2))
#         for k in range(4):
#             first_corner = reshaped[k%4]
#             second_corner = reshaped[(k+1)%4]
#        
#             all_edges[k,:]=np.copy(second_corner - first_corner)
#            
#         # logging.error('ALL EDGES ' + str(all_edges))
#            
#         all_lengths = np.sqrt(np.square(all_edges[:,0]) + np.square(all_edges[:,1]))
#         long_side = np.argmax(all_lengths)
#        
##                egim = np.sign(all_edges[long_side][1]/(all_edges[long_side][0] + 0.00001))*\
##                    np.abs(all_edges[long_side][1])/(all_lengths[long_side]  + 0.00001)
#         my_abs_cos = np.abs(all_edges[long_side][0])/(all_lengths[long_side]  + 0.00001)
#         my_sign = np.sign(all_edges[long_side][1]/(all_edges[long_side][0] + 0.00001))
#            
#       
#            # np.sqrt(all_edges[long_side][1]**2 + all_edges[long_side][0]**2)
#         angle = np.arccos(my_abs_cos*my_sign)
#        
#         center = np.mean(reshaped,axis=0)
#        
#         long_len = np.max(all_lengths)
#         short_len = np.min(all_lengths)
#        
#         res_ar[:2] = center
##                res_ar[4] = my_abs_cos
##                res_ar[5] = my_sign
#         res_ar[4] = angle
#         res_ar[2] = long_len
#         res_ar[3] = short_len
#        
#         center_width_orient.append(np.copy(res_ar))
#        
#         obj_exists = True
#    
# cwo = np.array(center_width_orient)    
#        
# back_con = five_params_to_corners(cwo)
#
# reshaped_render=np.zeros((196,200))
# back_render=np.zeros((196,200))
#
# for k in range(len(to_return)):
#     temp = to_return[k][:8].reshape(4,2)
# #    temp[:,0] = temp[:,0]*(map_extents[2]-map_extents[0])/resolution
# #    temp[:,1] = temp[:,1]*(map_extents[3]-map_extents[1])/resolution
#     render_polygon(reshaped_render, temp,(196,200))
#    
#     render_polygon(back_render, back_con[k],(196,200))

#    
#
# to_return[0]
# back_con[0]
#
#    
#sample_data = nuscenes.get('sample_data', token)
#sensor = nuscenes.get(
#'calibrated_sensor', sample_data['calibrated_sensor_token'])
#intrinsics = np.array(sensor['camera_intrinsic'])
#
#        



