import logging
import os
from datetime import datetime
from argparse import ArgumentParser
#from progressbar import progressbar

import torch

import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from src.detr.polyline_polygon import build
import json
from src.utils.configs import get_default_configuration,get_default_polyline_configuration_argo, load_config


from src.utils.confusion import BinaryConfusionMatrix
from src.data import data_factory

# import src.utils.visualise as vis_tools
import src.utils.visualise_polyline as vis_tools

from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import glob

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import random

image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]


def train(dataloader,dataset, model, criterion,  optimiser, postprocessors,summary,confusion, config,args, iteration):

    model.train()
    criterion.train()
    
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if hasattr(m, 'weight'):
                m.weight.requires_grad_(False)
            if hasattr(m, 'bias'):
                m.bias.requires_grad_(False)
            m.eval()
            
    
    
    loss_list=[]
    iou_list=[]
    
    data_loading_times=[]
    optimization_times=[]
    
    running_loss_dict={}
    
    time3 = time.time()
    for i, batch in enumerate(dataloader):
        
        if iteration % config.reset_confusion_interval == 0:
            confusion.reset()
      
        if batch[-1]:
            logging.error('PASSED BATCH')
            continue
     
        seq_images, targets, _ = batch
        
       
        global_thresh = 0.3
        cuda_targets = []
        for b in targets:
            temp_dict={}
            temp_dict['gt_order_labels'] = b['gt_order_labels'].cuda()
            temp_dict['blob_mat'] = b['blob_mat'].cuda()
            temp_dict['poly_centers'] = b['poly_centers'].cuda()
            temp_dict['poly_one_hots'] = b['poly_one_hots'].cuda()
            
      
            temp_dict['calib'] = b['calib'].cuda()
            temp_dict['center_img'] = b['center_img'].cuda()
            temp_dict['labels'] = b['labels'].cuda()
            temp_dict['roads'] = b['roads'].cuda()
            temp_dict['control_points'] = b['control_points'].cuda()
            temp_dict['con_matrix'] = b['con_matrix'].cuda()
            temp_dict['endpoints'] = b['endpoints'].cuda()
      
            temp_dict['mask'] = b['mask'].cuda()
         
            
            temp_dict['init_point_matrix'] = b['init_point_matrix'].cuda()
            temp_dict['sorted_control_points'] = b['sorted_control_points'].cuda()
            temp_dict['grid_sorted_control_points'] = b['grid_sorted_control_points'].cuda()
            temp_dict['sort_index'] = b['sort_index'].cuda()
     
            temp_dict['left_traffic'] = b['left_traffic'].cuda()
            temp_dict['outgoings'] = b['outgoings']
            temp_dict['incomings'] = b['incomings']
      
            cuda_targets.append(temp_dict)
        
        seq_images=seq_images.cuda()

        time2 = time.time()
        data_loading_times.append(time2-time3)
        
        outputs = model(seq_images,cuda_targets[0]['calib'], cuda_targets[0]['grid_sorted_control_points'], targets[0]['left_traffic'],training=True, iteration=iteration)

        
        loss_dict = criterion(outputs, cuda_targets)
        
     
        poly_indices = criterion.matcher(outputs, cuda_targets, thresh=global_thresh,do_polygons=True, gt_train=True)
        
        
        ham_loss = criterion.loss_poly_hamming(outputs, cuda_targets, poly_indices)
       
        exist_loss = criterion.loss_poly_exist(outputs, cuda_targets, poly_indices)

        loss_poly_center = criterion.loss_poly_center(outputs, cuda_targets, poly_indices)
    
    
        loss_dict.update(ham_loss)
        loss_dict.update(exist_loss)
        loss_dict.update(loss_poly_center)
        
        weight_dict = criterion.weight_dict
                
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Compute gradients and update parameters
        optimiser.zero_grad()
        # poly_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        
    
        
        optimiser.step()
        
        time3 = time.time()
        
        optimization_times.append(time3-time2)
        
        loss_list.append(loss.data.cpu().numpy())
        
        for k in loss_dict.keys():
        #     logging.error('LOSS ' + str(k) + ' : '+str(loss_dict[k]))
            
            if k in running_loss_dict.keys():
                running_loss_dict[k].append(loss_dict[k].data.cpu().numpy())
            else:
                running_loss_dict[k] = [loss_dict[k].data.cpu().numpy()]
        
   
        
#            Visualise
        if iteration % config.stats_interval == 0:

            out = vis_tools.get_selected_polylines(outputs , thresh = global_thresh)
            hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt, out = vis_tools.hausdorff_match(out, targets[0])
            
            merged_hausdorff_static_dist, merged_hausdorff_static_idx, _ = vis_tools.merged_hausdorff_match(out, targets[0])
            
            poly_stuff=(None,None,None,None,None)
            confusion.update(out, None, hausdorff_gt, hausdorff_static_idx,merged_hausdorff_static_idx, None, None, targets[0],poly_stuff=poly_stuff, do_common_poly=False, polyline=True, do_polys=True)
                
        # if iteration % config.vis_interval == 0:
            
            to_store_dist = confusion.to_store_dist
            last_sel_ar = confusion.last_sel_ar
            last_comp_ar = confusion.last_comp_ar
            last_gt_res_ar = confusion.last_gt_res_ar
            last_est_intersection_list = confusion.last_est_intersection_list
            last_gt_intersection_list = confusion.last_gt_intersection_list
            last_matched_gt_polys = confusion.last_matched_gt_polys
            last_matched_est_polys = confusion.last_matched_est_polys
            last_poly_match_gt_indices = confusion.last_poly_match_gt_indices 
            last_poly_match_est_indices = confusion.last_poly_match_est_indices 
            last_calc_gt_res_ar = confusion.last_calc_gt_res_ar
            last_calc_gt_intersection_list = confusion.last_calc_gt_intersection_list
            all_rnn_to_compare_gt = None
            all_rnn_to_compare_est = None
            
            rnn_gt_indices = None
            rnn_est_indices = None
            
            indeed_matched_list = confusion.indeed_matched_list
            
            metric_visuals = [last_gt_intersection_list, last_calc_gt_intersection_list, last_est_intersection_list, to_store_dist, last_sel_ar, last_comp_ar, last_gt_res_ar, last_calc_gt_res_ar, hausdorff_gt,
                              last_matched_gt_polys, last_matched_est_polys, last_poly_match_gt_indices, last_poly_match_est_indices,
                              all_rnn_to_compare_gt,all_rnn_to_compare_est, rnn_gt_indices, rnn_est_indices, indeed_matched_list]
            
            # fuzzy_coeffs = criterion.fuzzy_coeffs
            
            # naive_matched_labels = criterion.naive_matched_labels
            # vis_tools.save_results_train(seq_images.cpu().numpy(),outputs,  out, targets, config,
            #                               poly_indices ,metric_visuals=metric_visuals, naive_matched_labels=naive_matched_labels, fuzzy_coeffs=fuzzy_coeffs, gt_train=True)
        
        
                
        if iteration % config.log_interval == 0:
            
            
            logging.error('ITERATION ' + str(iteration))
             
            for k in running_loss_dict.keys():
                 logging.error('LOSS ' + str(k) +' : ' + str(np.mean(running_loss_dict[k]) ))
            
            
            logging.error('MEAN LOSS  : ' + str(np.mean(loss_list)))
            
            logging.error('STATIC MIOU : ' + str(np.mean(confusion.static_iou)) + ' TP : '+ str(confusion.static_line_tp)
                            + ' FP : '+ str(confusion.static_line_fp) + ' FN : '+ str(confusion.static_line_fn))
            
            logging.error('POLY IOU : ' + str(np.mean(confusion.poly_iou)) + ' TP : '+ str(confusion.poly_tp)
                            + ' FP : '+ str(confusion.poly_fp) + ' FN : '+ str(confusion.poly_fn))
            
            logging.error('PER IOU : ' + str(np.mean(confusion.per_poly_iou)) + ' TP : '+ str(confusion.per_poly_tp)
                            + ' FP : '+ str(confusion.per_poly_fp) + ' FN : '+ str(confusion.per_poly_fn))
            
            
            logging.error('ORDER ' + str(np.mean(confusion.order_error)))
            
            logging.error('Opt time : '+ str(np.mean(optimization_times)) +', data time : ' + str(np.mean(data_loading_times)))
            data_loading_times=[]
            optimization_times=[]
            
            summary.add_scalar('train/loss', float(np.mean(loss_list)), iteration)
            summary.add_scalar('train/miou', float(np.mean(iou_list)), iteration)
            iou_list=[]
            loss_list=[]
            running_loss_dict = {}
            confusion.reset
            
        iteration += 1
    return iteration, confusion

def evaluate(dataloader, model, criterion, postprocessors, confusion, summary, config,args, epoch):
     
     model.eval()
     iou_list=[]
     criterion.eval()

     static_thresh = 0.3
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
            temp_dict['gt_order_labels'] = b['gt_order_labels'].cuda()
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
        
        outputs = model(seq_images,cuda_targets[0]['calib'], cuda_targets[0]['grid_sorted_control_points'], targets[0]['left_traffic'], thresh=0.3,training=False)
        
        out = vis_tools.get_selected_polylines(outputs , thresh = static_thresh)
       
        hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt, out = vis_tools.hausdorff_match(out, targets[0])
     
        merged_hausdorff_static_dist, merged_hausdorff_static_idx, _ = vis_tools.merged_hausdorff_match(out, targets[0])
       
        poly_stuff=(None,None,None,None,None)
        confusion.update(out, None, hausdorff_gt, hausdorff_static_idx,merged_hausdorff_static_idx, None, None, targets[0],poly_stuff=poly_stuff, do_common_poly=False,polyline=True, do_polys=True)
            
#            vis_tools.save_results_eval(seq_images.cpu().numpy(), outputs,targets,  config)
        
        
     return confusion


def save_checkpoint(path, model, optimizer, scheduler, epoch, iteration,best_iou):

    if isinstance(model, nn.DataParallel):
        model = model.module
    
    ckpt = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'epoch' : epoch,
        'iteration': iteration,
        'best_iou' : best_iou
    }

    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, load_orig_ckpt=False):
    
    ckpt = torch.load(path)
 
    
    if load_orig_ckpt:
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        model.load_state_dict(ckpt['model'],strict=False)
        # model.load_state_dict(ckpt,strict=False)
        return 1, 0,0
    else:
        if isinstance(model, nn.DataParallel):
            model = model.module
            
            
            
        model.load_state_dict(ckpt['model'],strict=False)
  
    # Load optimiser state
        # optimizer.load_state_dict(ckpt['optimizer'])
    
        # Load scheduler state
        # scheduler.load_state_dict(ckpt['scheduler'])
        
        if 'iteration' not in ckpt.keys():
            to_return_iter = 0
        else:
            to_return_iter = ckpt['iteration']
        # to_return_iter = 0
        logging.error('LOADED MY')
        return ckpt['epoch'], ckpt['best_iou'],to_return_iter
    # return 0,0,0
def load_pretrained_backbone(path, model):
    

    
    ckpt = torch.load(path)

    model.load_state_dict(ckpt,strict=False)


# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_polyline_configuration_argo()


    return config


def create_experiment(config,  resume=None):

    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
        
    else:
        # Otherwise, generate a run directory based on the current time
        # name = datetime.now().strftime('{}_%y-%m-%d--%H-%M-%S').format('run')
        name = 'poly_argo'
        logdir = os.path.join(os.path.expandvars(config.logdir), name)
        print("\n==> Creating new experiment in directory:\n" + logdir)
        os.makedirs(logdir,exist_ok=True)
        os.makedirs(os.path.join(config.logdir,'val_images'),exist_ok=True)
        os.makedirs(os.path.join(config.logdir,'train_images'),exist_ok=True)
        os.makedirs(os.path.join(config.logdir,'train_figure_images'),exist_ok=True)
        
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



intersection_mode = 'polygon'

apply_poly_loss = True
def main():



    parser = ArgumentParser()
    parser.add_argument('--dgx', type=bool, default=False,
                    help='whether it is on dgx')
    parser.add_argument('--resume', default= None, 
                        help='path to an experiment to resume')
    
    parser.add_argument('--exp', default='/home/cany/TPLR/baseline/Experiments/mle.json', 
                        help='path to an experiment to resume')
   

    parser.add_argument('--split_pe', type=bool, default=False,
                    help='whether it is on dgx')
    parser.add_argument('--apply_poly_loss', type=bool, default=apply_poly_loss,
                    help='whether it is on dgx')
 
    parser.add_argument('--objects', type=bool, default=False,
                help='whether estimate objects')
    
  
    parser.add_argument('--num_spline_points', default=3, type=int,
                help="Num object classes")
    
    

    parser.add_argument('--lr', default=1e-5, type=float)
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
 
    parser.add_argument('--dim_feedforward', default=256, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    
    
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks',default=False,
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_obj_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_obj_cost_center', default=3, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_len', default=0.5, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_orient', default=1, type=float,
                        help="Class coefficient in the matching cost")

    
    
    parser.add_argument('--set_cost_poly_hamming', default=3, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--set_cost_poly_exist', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--set_cost_poly_center', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    
    
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=4, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_end', default=1, type=float,
                        help="L1 endpoint coefficient in the matching cost")
    
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    
    parser.add_argument('--object_detection_loss_coef', default=4, type=float)
    parser.add_argument('--object_center_loss_coef', default=3, type=float)
    parser.add_argument('--object_len_loss_coef', default=0.5, type=float)
    parser.add_argument('--object_orient_loss_coef', default=0.5, type=float)
    
    parser.add_argument('--polyline_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--assoc_loss_coef', default=1, type=float)
    parser.add_argument('--detection_loss_coef', default=1, type=float)
    parser.add_argument('--endpoints_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=2, type=float)
    parser.add_argument('--focal_loss_coef', default=0.5, type=float)
    parser.add_argument('--init_points_loss_coef', default=10, type=float)

    
    parser.add_argument('--loss_poly_hamming_coef', default=0.5, type=float)
    parser.add_argument('--loss_poly_exist_coef', default=0.5, type=float)
    parser.add_argument('--loss_poly_center_coef', default=0.3, type=float)
    
    parser.add_argument('--minimality_coef', default=3, type=float)
    parser.add_argument('--loss_naive_coef', default=1, type=float)
    
    
    parser.add_argument('--start_loss_coef', default=0, type=float)
    parser.add_argument('--fin_loss_coef', default=0, type=float)
    
    parser.add_argument('--loss_end_match_coef', default=1, type=float)
    
    
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--visible_loss_coef', default=1, type=float)
    
    parser.add_argument('--eos_coef', default=0.01, type=float,
                        help="Relative classification weight of the no-object class")
    
    parser.add_argument('--object_eos_coef', default=0.1, type=float,
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
    
    parser.add_argument('--intersection_mode', type=str, default=intersection_mode,
                    help='whether it is on dgx')
    args = parser.parse_args()
    
    
    print('GOT ARGS ')
    logging.error(str(args))
    
    # Load configuration
    config = get_configuration(args)
    
    opts = json.load(open(args.exp, 'r'))
    
    # Create a directory for the experiment
    logdir = create_experiment(config, args.resume)
    
    config.save_logdir = logdir
    config.n_control_points = args.num_spline_points
    config.freeze()
    
    summary = SummaryWriter(logdir)

    # Set default device
    # if len(config.gpus) > 0:
    #     torch.cuda.set_device(config.gpus[0])
    device = torch.device(args.device)
    # Setup experiment
    model, criterion, postprocessors = build(args, config,opts)
    
    model.to(device)
    

    train_loader,train_dataset, val_loader, val_dataset = data_factory.build_argoverse_dataloader(config, args)

    freeze_backbone_layers(model)
    
 
  
    param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
    ]
        
    
    
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Load checkpoint
    
    if config.load_pretrained_backbone:
        load_pretrained_backbone(config.backbone_ckpt_path, model.backbone.backbone_net)
        epoch, best_iou, iteration = 1,0,0

    else:
            
        epoch, best_iou, iteration = load_checkpoint(os.path.join(logdir, 'latest.pth'),
                                       model, optimizer, lr_scheduler)
#        
        logging.error('LOADED MY CHECKPOINT')
    
    freeze_backbone_layers(model)
   
    confusion = BinaryConfusionMatrix()
            
    # Main training loop
    while epoch <= config.num_epochs:
        
        print('\n\n=== Beginning epoch {} of {} ==='.format(epoch, 
                                                            config.num_epochs))
        

        iteration, confusion=train(train_loader, train_dataset,model, criterion,  optimizer,postprocessors , summary,confusion, config,args, iteration)
        logging.error('COCO FINISHED')
        
        lr_scheduler.step()
        
        save_checkpoint(os.path.join(logdir, 'latest.pth'), model, optimizer, 
                        lr_scheduler, epoch, iteration,best_iou)
        # Evaluate on the validation set
        if epoch % 5 == 0:
            val_confusion = BinaryConfusionMatrix()
            val_con = evaluate(val_loader, model, criterion, postprocessors,val_confusion,summary, config, args,epoch)
            static_res_dict = val_con.get_res_dict
            file1 = open(os.path.join(logdir,'val_res_thresh_'+'.txt'),"a")
            
            for k in static_res_dict.keys():
                logging.error(str(k) + ' : ' + str(static_res_dict[k]))
                file1.write(str(k) + ' : ' + str(static_res_dict[k]) + ' \n')
                
        
            file1.close()    
            val_iou = val_con.order_error
#            val_iou = val_iou = val_con.static_iou/2
            logging.error('VAL IOU ' + str(val_iou))
            # Update learning rate
            logging.error('BEST IOU ' + str(best_iou))
            # Save checkpoints
            
            # Save checkpoints
            if val_iou < best_iou:
                best_iou = val_iou
                logging.error('BEST IOU ' + str(val_iou))
                save_checkpoint(os.path.join(logdir, 'best.pth'), model, 
                                optimizer, lr_scheduler, epoch, iteration, best_iou)
            
        
        
        epoch += 1
    
    print("\nTraining complete!")



if __name__ == '__main__':
    main()

                

