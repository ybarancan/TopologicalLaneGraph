import logging
import os
from datetime import datetime
from argparse import ArgumentParser

import torch

import torch.nn as nn

from src.detr.polyline_polygon import build
import json
from src.utils.configs import get_default_configuration,get_default_polyline_configuration, load_config


from src.utils.confusion import BinaryConfusionMatrix
from src.data import data_factory
import src.utils.visualise_polyline as vis_tools

from tqdm import tqdm
import numpy as np

def evaluate(dataloader, model, postprocessors, confusion,  config,args, thresh):
     
     model.eval()
     
     logging.error('VALIDATION')
     # Iterate over dataset
     for i, batch in enumerate(tqdm(dataloader)):
        
        seq_images, targets, _ = batch
        if seq_images == None:
            continue
        seq_images = seq_images.cuda()
        cuda_targets = []

        cuda_targets = []
        for b in targets:
            temp_dict={}
            
#            temp_dict['poly_list'] = b['blob_ids'].cuda()
            temp_dict['blob_mat'] = b['blob_mat'].cuda()
            temp_dict['poly_centers'] = b['poly_centers'].cuda()
            temp_dict['poly_one_hots'] = b['poly_one_hots'].cuda()
         
            temp_dict['calib'] = b['calib'].cuda()
            temp_dict['center_img'] = b['center_img'].cuda()
            temp_dict['labels'] = b['labels'].cuda()
            temp_dict['roads'] = b['roads'].cuda()
            temp_dict['control_points'] = b['control_points'].cuda()
            temp_dict['con_matrix'] = b['con_matrix'].cuda()
            
            
            temp_dict['init_point_matrix'] = b['init_point_matrix'].cuda()
            temp_dict['sorted_control_points'] = b['sorted_control_points'].cuda()
            temp_dict['grid_sorted_control_points'] = b['grid_sorted_control_points'].cuda()
            temp_dict['sort_index'] = b['sort_index'].cuda()
            
            temp_dict['endpoints'] = b['endpoints'].cuda()

            temp_dict['mask'] = b['mask'].cuda()

            temp_dict['left_traffic'] = b['left_traffic'].cuda()
            temp_dict['outgoings'] = b['outgoings']
            temp_dict['incomings'] = b['incomings']

            cuda_targets.append(temp_dict)
        
        
            
        static_thresh = thresh
        outputs = model(seq_images,cuda_targets[0]['calib'], cuda_targets[0]['grid_sorted_control_points'], targets[0]['left_traffic'], thresh=static_thresh,training=args.use_gt)
    
        out = vis_tools.get_selected_polylines(outputs , thresh = static_thresh)
        poly_centers, poly_one_hots, blob_mat, blob_ids, real_hots = vis_tools.get_polygons(out,  thresh = static_thresh)
        out['my_blob_mat'] = blob_mat
        hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt, out = vis_tools.hausdorff_match(out, targets[0])
        
        merged_hausdorff_static_dist, merged_hausdorff_static_idx, _ = vis_tools.merged_hausdorff_match(out, targets[0])
        
        poly_stuff=(poly_centers, poly_one_hots, blob_mat, blob_ids, real_hots)
        try:
#            poly_stuff=(None, None,None,None,None)
            confusion.update(out, None, hausdorff_gt, hausdorff_static_idx,merged_hausdorff_static_idx, None, None, targets[0],poly_stuff=poly_stuff, do_common_poly=True, polyline=True, do_polys=True)
        except Exception as e:
            logging.error('EXCEPTION IN CONFUSION ')
            logging.error(str(e))
            continue
        
        # vis_tools.save_results_eval(seq_images.cpu().numpy(),outputs,  out, targets, config,
        #                               gt_train=use_gt, common_poly=poly_stuff)
        
     return confusion


def load_checkpoint(path, model):
    
    ckpt = torch.load(path)
    
    logging.error('LOADED  ' + path)
    if isinstance(model, nn.DataParallel):
        model = model.module
        
    model.load_state_dict(ckpt['model'],strict=False)
  
#
    logging.error('LOADED MY')
    return 0,0,0
   
# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_polyline_configuration()

    
    return config


def create_experiment(config, name=None):

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
            

            p.requires_grad_(False)

split_pe = False

apply_bev_pe = False
abs_bev = False

use_gt = False

only_bev_pe=False


euler=True

intersection_mode = 'polygon'

model_name = 'poly_mc_'+str(use_gt)

base_dir = '/cluster/work/cvl/cany/simplice'

def main():

    parser = ArgumentParser()
    
    parser.add_argument('--exp', default='/cluster/home/cany/simplice-github/baseline/Experiments/mle.json', 
                        help='path to an experiment to resume')
    
    parser.add_argument('--use_gt', type=bool, default=use_gt,
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
    
    

    
    parser.add_argument('--num_spline_points', default=3, type=int,
                help="Num object classes")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
 
    parser.add_argument('--dim_feedforward', default=256, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    
    
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    
    
    parser.add_argument('--polyline_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--assoc_loss_coef', default=1, type=float)
    parser.add_argument('--detection_loss_coef', default=1, type=float)
    parser.add_argument('--endpoints_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=1, type=float)
    parser.add_argument('--focal_loss_coef', default=0.1, type=float)
    parser.add_argument('--init_points_loss_coef', default=10, type=float)

    
    parser.add_argument('--loss_poly_hamming_coef', default=0.5, type=float)
    parser.add_argument('--loss_poly_exist_coef', default=0.5, type=float)
    parser.add_argument('--loss_poly_center_coef', default=0.3, type=float)
    
    parser.add_argument('--minimality_coef', default=3, type=float)
    parser.add_argument('--loss_naive_coef', default=1, type=float)
    
    
    parser.add_argument('--loss_end_match_coef', default=1, type=float)
    

    parser.add_argument('--set_cost_poly_hamming', default=3, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--set_cost_poly_exist', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--set_cost_poly_center', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    

    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_end', default=1, type=float,
                        help="L1 endpoint coefficient in the matching cost")
    
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
  
    parser.add_argument('--eos_coef', default=0.01, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument('--intersection_mode', type=str, default=intersection_mode,
                    help='whether it is on dgx')
    args = parser.parse_args()
    
    print('GOT ARGS ')
    logging.error(str(args))
    
    # Load configuration
    config = get_configuration(args)
    
    opts = json.load(open(args.exp, 'r'))
    
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
    model, _, postprocessors = build(args, config,opts)
    
    model.to(device)
    
    train_loader,train_dataset, val_loader, val_dataset = data_factory.build_nuscenes_dataloader(config,args, val=True)

    epoch, best_iou, iteration = load_checkpoint(os.path.join(base_dir,'final_ckpts','PRNN_MC_Nusc.pth'),  model)
    
        
    logging.error('LOADED MY CHECKPOINT')
    
    freeze_backbone_layers(model)
   
    thresh = 0.3
    
    val_con = BinaryConfusionMatrix()
    val_con = evaluate(val_loader, model, postprocessors,val_con, config, args,thresh)
    
    
    static_res_dict = val_con.get_res_dict
    file1 = open(os.path.join(logdir,'val_res_'+str(use_gt)+'.txt'),"a")
    
    for k in static_res_dict.keys():
        logging.error(str(k) + ' : ' + str(static_res_dict[k]))
        file1.write(str(k) + ' : ' + str(static_res_dict[k]) + ' \n')
        
    file1.close()    
    
    

if __name__ == '__main__':
    main()

                

