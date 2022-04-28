import logging
import os
from datetime import datetime
from argparse import ArgumentParser
#from progressbar import progressbar

import torch

import torch.nn as nn

from src.detr.detr import build

from src.utils.configs import get_default_configuration_argo, load_config

from src.utils.confusion import BinaryConfusionMatrix
from src.data import data_factory

import src.utils.visualise as vis_tools

from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import glob


def evaluate(dataloader, model, criterion, postprocessors, confusion, config,args,
           thresh):
     
     model.eval()

     criterion.eval()

     logging.error('VALIDATION')
     # Iterate over dataset
     for i, batch in enumerate(tqdm(dataloader)):
        
        seq_images, targets, _ = batch
        if seq_images == None:
            continue
        seq_images = seq_images.cuda()
        cuda_targets = []
        for b in targets:
            temp_dict={}
            
            # target['gt_order_labels'] = torch.tensor(intersections).long()
            
            temp_dict['blob_mat'] = b['blob_mat'].cuda()
            temp_dict['poly_centers'] = b['poly_centers'].cuda()
            temp_dict['poly_one_hots'] = b['poly_one_hots'].cuda()
            


            temp_dict['calib'] = b['calib'].cuda()
            temp_dict['obj_exists'] = b['obj_exists'].cuda()
            temp_dict['center_img'] = b['center_img'].cuda()
            temp_dict['labels'] = b['labels'].cuda()
            temp_dict['roads'] = b['roads'].cuda()
            temp_dict['control_points'] = b['control_points'].cuda()
            temp_dict['con_matrix'] = b['con_matrix'].cuda()
            temp_dict['endpoints'] = b['endpoints'].cuda()
            temp_dict['mask'] = b['mask'].cuda()
            temp_dict['bev_mask'] = b['bev_mask'].cuda()
            temp_dict['static_mask'] = b['static_mask'].cuda()
            temp_dict['outgoings'] = b['outgoings']
            temp_dict['incomings'] = b['incomings']
            
            
            cuda_targets.append(temp_dict)
        
            
            
        # logging.error('SCENE ' + targets[0]['scene_name'])
         
        outputs = model(seq_images,cuda_targets[0]['calib'], False)
        
    
        global_thresh = thresh
        threshed_outputs = model.thresh_and_assoc_estimates(outputs,thresh=global_thresh)
        base_postprocessed = postprocessors['bbox'](threshed_outputs,torch.Tensor(np.tile(np.expand_dims(np.array(config.patch_size),axis=0),[seq_images.shape[0],1])).cuda(),objects=False)

        
        out = vis_tools.get_selected_estimates(base_postprocessed , outputs, thresh = global_thresh, do_polygons=(not base_version))
        
        time1 = time.time()
        
        poly_centers, poly_one_hots, blob_mat, blob_ids, real_hots = vis_tools.get_polygons(base_postprocessed, global_thresh)

#        logging.error('TIME FOR EXTRACTING POLYGONS ' + str(time.time() - time1))

        out[0]['my_blob_mat'] = blob_mat
        
        match_static_indices,  match_poly_indices = criterion.matcher(outputs, cuda_targets,  do_polygons=True)
        
        matched_static_outputs = criterion.get_assoc_estimates( outputs, match_static_indices)
        
        static_inter_dict, static_idx, static_target_ids = criterion.get_interpolated(matched_static_outputs, cuda_targets, match_static_indices)
        
        hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt = vis_tools.hausdorff_match(out[0], targets[0])
        merged_hausdorff_static_dist, merged_hausdorff_static_idx, merged_res_interpolated = vis_tools.merged_hausdorff_match(out[0], targets[0])
         
        try:
            poly_stuff=(poly_centers, poly_one_hots, blob_mat, blob_ids, real_hots)
#            poly_stuff=(None, None,None,None,None)
            confusion.update(out[0], static_inter_dict, hausdorff_gt, hausdorff_static_idx,  merged_hausdorff_static_idx, static_idx, static_target_ids, targets[0], poly_stuff=poly_stuff,do_common_poly=True, do_polys=True)

        except Exception as e:
            logging.error('EXCEPTION IN CONFUSION ')
            logging.error(str(e))
            continue
        
          
     return confusion

def load_checkpoint(path, model):
    
    ckpt = torch.load(path)
    logging.error('LOADED ' + path)
  
    if isinstance(model, nn.DataParallel):
        model = model.module
        
        
        
    model.load_state_dict(ckpt['model'],strict=False)
   
    
    logging.error('LOADED MY')
    return 0,0,0



# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_configuration_argo()

    

    return config


def create_experiment(config,  name):

    logdir = os.path.join(os.path.expandvars(config.logdir), name)
    print("\n==> Creating new experiment in directory:\n" + logdir)
    os.makedirs(logdir,exist_ok=True)
    os.makedirs(os.path.join(config.logdir,'val_images'),exist_ok=True)
    os.makedirs(os.path.join(config.logdir,'train_images'),exist_ok=True)
    
    # Display the config options on-screen
    print(config.dump())
    
    # Save the current config
    with open(os.path.join(logdir, 'config.yml'), 'w') as f:
        f.write(config.dump())
    
    return logdir



    
def freeze_backbone_layers(model):
    logging.error('MODEL FREEZE')
    for n, p in model.named_parameters():
#        logging.error('STR ' + str(n))
        if "backbone" in n and p.requires_grad:
            
#            if  (('block14' in n) |('block15' in n) |('block16' in n) |('block17' in n) |('block18' in n) 
#                 |('block19' in n) | ('block20' in n) | ('block21' in n) | ('spp' in n)):
            if  ( ('block18' in n) |('block19' in n) | ('block20' in n) | ('block21' in n) | ('spp' in n)):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
            # logging.error(str(n) + ', '+str(p.requires_grad))
    
#                logging.error(str(n) + ', '+str(p.requires_grad))

base_version = False


split_pe = True

apply_bev_pe = True
abs_bev = True

only_bev_pe=False


euler=False
intersection_mode='polygon'


base_dir = '/scratch_net/catweazle/cany/TPLR'


def main():
    large_parameters =  dict()
    large_parameters['hidden_dim'] = 256
    large_parameters['dim_feedforward'] = 512
    
    large_parameters['class_embed_dim']=256
    large_parameters['class_embed_num']=3
    
    large_parameters['poly_embed_dim']=256
    large_parameters['poly_embed_num']=3
    large_parameters['poly_exist_dim']=256
    large_parameters['poly_exist_num']=3
    large_parameters['poly_center_dim']=256
    large_parameters['poly_center_num']=3
    
    large_parameters['box_embed_dim']=256
    large_parameters['box_embed_num']=3
    large_parameters['endpoint_embed_dim']=256
    large_parameters['endpoint_embed_num']=3
    large_parameters['assoc_embed_dim']=256
    large_parameters['assoc_embed_last_dim']=128
    large_parameters['assoc_embed_num']=3
    large_parameters['assoc_classifier_dim']=256
    large_parameters['assoc_classifier_num']=3
    
    
    num_queries = 100
    num_enc_layers = 4
    num_dec_layers = 4
    
    
    
    num_poly_queries = 100
  
    model_name = 'argo_trans'  
   
    parser = ArgumentParser()
    parser.add_argument('--dgx', type=bool, default=euler,
                    help='whether it is on dgx')
    parser.add_argument('--resume', default=base_dir+'/'+model_name, 
                        help='path to an experiment to resume')
   # base_dir+'/'+model_name
    # '/scratch_net/catweazle/cany/lanefinder/'+model_name
    # '/scratch_net/catweazle/cany/lanefinder/'+model_name
    
    
    parser.add_argument('--use_gt_labels', type=bool, default=True,
                    help='whether it is on dgx')
    
    parser.add_argument('--split_pe', type=bool, default=split_pe,
                    help='whether it is on dgx')
    
  
    
    parser.add_argument('--only_bev_pe', type=bool, default=only_bev_pe,
                    help='whether it is on dgx')
    # '/scratch_net/catweazle/cany/lanefinder/combined_objects_4'
    parser.add_argument('--bev_pe', type=bool, default=apply_bev_pe,
                    help='whether it is on dgx')
    parser.add_argument('--abs_bev', type=bool, default=abs_bev,
                    help='whether it is on dgx')
    
  
    parser.add_argument('--apply_poly_loss', type=bool, default=True,
                    help='whether it is on dgx')
    
    parser.add_argument('--objects', type=bool, default=False,
                help='whether estimate objects')
    
    parser.add_argument('--num_poly_queries', default=num_poly_queries, type=int,
                    help="Number of query slots")
    
    
    parser.add_argument('--num_spline_points', default=3, type=int,
                help="Num object classes")
    
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--refiner_lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=num_enc_layers, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=num_dec_layers, type=int,
                        help="Number of decoding layers in the transformer")
    
    
    parser.add_argument('--dim_feedforward', default=large_parameters['dim_feedforward'], type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    
    
    parser.add_argument('--hidden_dim', default=large_parameters['hidden_dim'], type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--dropout', default=0.15, type=float,
                        help="Dropout applied in the transformer")
    
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=num_queries, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks',default=False,
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_obj_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_obj_cost_center', default=2, type=float,
                        help="Class coefficient in the matching cost")
    
    parser.add_argument('--set_obj_cost_image_center', default=2, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_len', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_orient', default=1, type=float,
                        help="Class coefficient in the matching cost")


    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_end', default=1, type=float,
                        help="L1 endpoint coefficient in the matching cost")
    
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")

    
    parser.add_argument('--set_cost_poly_hamming', default=3, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--set_cost_poly_exist', default=3, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--set_cost_poly_center', default=1, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients

    
    parser.add_argument('--polyline_loss_coef', default=3, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--assoc_loss_coef', default=2, type=float)
    parser.add_argument('--detection_loss_coef', default=3, type=float)
    parser.add_argument('--endpoints_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=3, type=float)
    parser.add_argument('--focal_loss_coef', default=0.1, type=float)
    
    parser.add_argument('--start_loss_coef', default=0, type=float)
    parser.add_argument('--fin_loss_coef', default=0, type=float)
    
    parser.add_argument('--loss_end_match_coef', default=1, type=float)
    parser.add_argument('--loss_naive_coef', default=0.5, type=float)
    
    parser.add_argument('--minimality_coef', default=5, type=float)
    
    
    parser.add_argument('--loss_poly_hamming_coef', default=4, type=float)
    parser.add_argument('--loss_poly_exist_coef', default=4, type=float)
    parser.add_argument('--loss_poly_center_coef', default=1, type=float)
    
    parser.add_argument('--poly_negative_coef', default=0.3, type=float)           
    
    
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--visible_loss_coef', default=1, type=float)
    
    parser.add_argument('--eos_coef', default=0.3, type=float,
                        help="Relative classification weight of the no-object class")
    

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
   
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval',default=False, action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--interval_start', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--interval_end', default=94, type=int,
                        help='number of distributed processes')
    parser.add_argument('--intersection_mode', type=str, default=intersection_mode,
                    help='whether it is on dgx')
    args = parser.parse_args()
    
    
    
    print('GOT ARGS ')
    logging.error(str(args))
    
    
    # Load configuration
    config = get_configuration(args)
    
    # Create a directory for the experiment
    logdir = create_experiment(config, model_name)
    
    config.save_logdir = logdir
    config.n_control_points = args.num_spline_points
    config.freeze()

    # Set default device
    # if len(config.gpus) > 0:
    #     torch.cuda.set_device(config.gpus[0])
    device = torch.device(args.device)
    # Setup experiment
    model, criterion, postprocessors = build(args, config,large_parameters)
    
    model.to(device)
    
   
    logging.error('BUILD ENTER DATA')
    _,_,val_loader, val_dataset = data_factory.build_argoverse_dataloader(config,args, val=True)
    
    logging.error('BUILT DATA')
    
  
    epoch, best_iou, iteration = load_checkpoint(os.path.join(base_dir,'final_ckpts', 'TR_MC_Argo.pth'),
                              model)

    
    logging.error('LOADED MY CHECKPOINT')

    
    freeze_backbone_layers(model)
   

 
    thresh = 0.3
    logging.error('THRESH')
    
    val_confusion = BinaryConfusionMatrix()
    
    val_con = evaluate(val_loader, model, criterion, postprocessors,val_confusion, config, args,thresh)
   

    static_res_dict = val_con.get_res_dict
    
    file1 = open(os.path.join(logdir,'val_res_.txt'),"a")
    
    for k in static_res_dict.keys():
        logging.error(str(k) + ' : ' + str(static_res_dict[k]))
        file1.write(str(k) + ' : ' + str(static_res_dict[k]) + ' \n')
    

    file1.close()    
        
       


if __name__ == '__main__':
    main()

                

