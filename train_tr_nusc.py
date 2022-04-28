import logging
import os

from argparse import ArgumentParser

import torch

import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from src.detr.detr import build

from src.utils.configs import get_default_configuration


from src.utils.confusion import BinaryConfusionMatrix as NewBinaryConfusionMatrix
from src.data import data_factory

import src.utils.visualise as vis_tools
import torchvision
from tqdm import tqdm
import numpy as np

import time

image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]


def train(dataloader,dataset, model, criterion, optimiser, postprocessors,summary,confusion, config,args, iteration, transforms):

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
    
    poly_dict = dict()
    global_thresh = 0.4
    total_passed_batch = 0
    for i, batch in enumerate(dataloader):
        
        if iteration % config.reset_confusion_interval == 0:
            confusion.reset()
        
        if batch[-1]:
            logging.error('PASSED BATCH')
            total_passed_batch = total_passed_batch + 1
            continue
        
        seq_images, targets, _ = batch
        
        if seq_images == None:
            continue
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
            
            temp_dict['start_con_matrix'] = b['start_con_matrix'].cuda()
            temp_dict['fin_con_matrix'] = b['fin_con_matrix'].cuda()
            
            
            temp_dict['endpoints'] = b['endpoints'].cuda()
            temp_dict['smoothed'] = b['smoothed'].cuda()
            temp_dict['mask'] = b['mask'].cuda()

            temp_dict['dilated'] = b['dilated'].cuda()
            

            temp_dict['left_traffic'] = b['left_traffic'].cuda()
            temp_dict['outgoings'] = b['outgoings']
            temp_dict['incomings'] = b['incomings']
      
            cuda_targets.append(temp_dict)
        

        
        seq_images=seq_images.cuda()


        time2 = time.time()
        data_loading_times.append(time2-time3)
        
        outputs = model(seq_images,cuda_targets[0]['calib'], targets[0]['left_traffic'])
        
        loss_dict = criterion(outputs, cuda_targets)
        

        mod_ham, poly_indices, problem_in_match = criterion.matcher(outputs, cuda_targets, thresh=global_thresh, gt_train=True)
        
        if not problem_in_match:
        
            cuda_targets[0]['modified_gt_poly_one_hots'] = mod_ham.cuda()
            targets[0]['modified_gt_poly_one_hots'] = mod_ham
        
        
            ham_loss = criterion.loss_poly_hamming(outputs, cuda_targets, poly_indices)
            
            exist_loss = criterion.loss_poly_exist(outputs, cuda_targets, poly_indices)
            
            loss_poly_center = criterion.loss_poly_center(outputs, cuda_targets, poly_indices)
        
            loss_dict.update(ham_loss)
            loss_dict.update(exist_loss)
            loss_dict.update(loss_poly_center)
        
        weight_dict = criterion.weight_dict
     
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimiser.zero_grad()
       
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimiser.step()
  
        time3 = time.time()
        
        optimization_times.append(time3-time2)
        
        loss_list.append(loss.data.cpu().numpy())
        
        for k in loss_dict.keys():
     
            if 'loss' in k:
                if k in running_loss_dict.keys():
                    running_loss_dict[k].append(loss_dict[k].data.cpu().numpy())
                else:
                    running_loss_dict[k] = [loss_dict[k].data.cpu().numpy()]
        
            

        if iteration % config.vis_interval == 0:

            logging.error('SAVING SCENE ' + targets[0]['scene_token'])
            logging.error('SAVING SAMPLE ' + targets[0]['sample_token'])
            logging.error('SAVING DATA ' + targets[0]['data_token'])
            
            
            threshed_outputs = model.thresh_and_assoc_estimates(outputs,thresh=global_thresh)
            base_postprocessed = postprocessors['bbox'](threshed_outputs,torch.Tensor(np.tile(np.expand_dims(np.array(config.patch_size),axis=0),[seq_images.shape[0],1])).cuda(),objects=False)
    
            out = vis_tools.get_selected_estimates(base_postprocessed , outputs, thresh = global_thresh, do_polygons=True)
           
                
            match_static_indices,  match_poly_indices = criterion.matcher(outputs, cuda_targets, do_polygons=False)
            
            
            matched_static_outputs = criterion.get_assoc_estimates( outputs, match_static_indices)
            
            
            static_inter_dict, static_idx, static_target_ids = criterion.get_interpolated(matched_static_outputs, cuda_targets, match_static_indices)
            
            
            hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt = vis_tools.hausdorff_match(out[0], targets[0])
            merged_hausdorff_static_dist, merged_hausdorff_static_idx, merged_res_interpolated = vis_tools.merged_hausdorff_match(out[0], targets[0])
            
            try:
                confusion.update(out[0], static_inter_dict, hausdorff_gt, hausdorff_static_idx,  merged_hausdorff_static_idx, static_idx, static_target_ids, targets[0],  do_polys=True)
                
                to_store_dist = confusion.to_store_dist
                last_sel_ar = confusion.last_sel_ar
                last_comp_ar = confusion.last_comp_ar
                last_gt_res_ar = confusion.last_gt_res_ar
                
                last_calc_gt_res_ar = confusion.last_calc_gt_res_ar
                last_calc_gt_intersection_list = confusion.last_calc_gt_intersection_list
                
                last_est_intersection_list = confusion.last_est_intersection_list
                last_gt_intersection_list = confusion.last_gt_intersection_list
                last_matched_gt_polys = confusion.last_matched_gt_polys
                last_matched_est_polys = confusion.last_matched_est_polys
                last_poly_match_gt_indices = confusion.last_poly_match_gt_indices 
                last_poly_match_est_indices = confusion.last_poly_match_est_indices 
                    
                all_rnn_to_compare_gt = None
                all_rnn_to_compare_est = None
                
                rnn_gt_indices = None
                rnn_est_indices = None
                
                indeed_matched_list = confusion.indeed_matched_list
            except Exception as e:
                logging.error('EXCEPTION IN CONFUSION ')
                logging.error(str(e))
                continue
            
            
            poly_voronoi = criterion.poly_voronoi
            naive_matched_labels = criterion.naive_matched_labels
            
            metric_visuals = [last_gt_intersection_list, last_calc_gt_intersection_list, last_est_intersection_list, to_store_dist, last_sel_ar, last_comp_ar, last_gt_res_ar, last_calc_gt_res_ar, hausdorff_gt,
                              last_matched_gt_polys, last_matched_est_polys, last_poly_match_gt_indices, last_poly_match_est_indices,
                              all_rnn_to_compare_gt,all_rnn_to_compare_est, rnn_gt_indices, rnn_est_indices, indeed_matched_list]
            
            vis_tools.save_results_train(seq_images.cpu().numpy(),outputs,  out[0], match_poly_indices, targets, static_inter_dict,None, static_target_ids, None, config,
                                         gt_poly_indices=poly_indices, mod_ham=mod_ham, metric_visuals=metric_visuals,  naive_matched_labels=naive_matched_labels, poly_voronoi=poly_voronoi)
        

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
        
    # np.save('/cluster/work/cvl/cany/simplice/estimate_merged_polygon_dict.npy',poly_dict)
    return iteration, confusion

def evaluate(dataloader, model, criterion, postprocessors, confusion, summary, config,args, epoch):
     
     model.eval()
  
     criterion.eval()
  
     global_thresh = 0.3
     logging.error('VALIDATION')
     # Iterate over dataset
     for i, batch in enumerate(tqdm(dataloader)):
        
        seq_images, targets, _ = batch
        seq_images = seq_images.cuda()
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
            
            temp_dict['start_con_matrix'] = b['start_con_matrix'].cuda()
            temp_dict['fin_con_matrix'] = b['fin_con_matrix'].cuda()
            
            
            temp_dict['endpoints'] = b['endpoints'].cuda()
            temp_dict['smoothed'] = b['smoothed'].cuda()
            temp_dict['mask'] = b['mask'].cuda()
            
            temp_dict['dilated'] = b['dilated'].cuda()
            
            temp_dict['left_traffic'] = b['left_traffic'].cuda()
            temp_dict['outgoings'] = b['outgoings']
            temp_dict['incomings'] = b['incomings']

            cuda_targets.append(temp_dict)

        seq_images=seq_images.cuda()

        
        outputs = model(seq_images,cuda_targets[0]['calib'], targets[0]['left_traffic'])
        
        mod_ham, poly_indices, problem_in_match = criterion.matcher(outputs, cuda_targets, thresh=global_thresh)
        
        
        threshed_outputs = model.thresh_and_assoc_estimates(outputs,thresh=global_thresh)
        base_postprocessed = postprocessors['bbox'](threshed_outputs,torch.Tensor(np.tile(np.expand_dims(np.array(config.patch_size),axis=0),[seq_images.shape[0],1])).cuda(),objects=False)

        out = vis_tools.get_selected_estimates(base_postprocessed , outputs, thresh = global_thresh, do_polygons=True)
       
        
            
        match_static_indices,  match_poly_indices = criterion.matcher(outputs, cuda_targets,  do_polygons=False)
        
     
        matched_static_outputs = criterion.get_assoc_estimates( outputs, match_static_indices)
        
        static_inter_dict, static_idx, static_target_ids = criterion.get_interpolated(matched_static_outputs, cuda_targets, match_static_indices)
        
        hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt = vis_tools.hausdorff_match(out[0], targets[0])
        merged_hausdorff_static_dist, merged_hausdorff_static_idx, merged_res_interpolated = vis_tools.merged_hausdorff_match(out[0], targets[0])
        
        try:
            confusion.update(out[0], static_inter_dict, hausdorff_gt, hausdorff_static_idx,  merged_hausdorff_static_idx, static_idx, static_target_ids, targets[0],   do_polys=True)

        except Exception as e:
            logging.error('EXCEPTION IN CONFUSION ')
            logging.error(str(e))
            continue
        
        
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
            
        all_rnn_to_compare_gt = None
        all_rnn_to_compare_est = None
            
        rnn_gt_indices = None
        rnn_est_indices = None 
        
        metric_visuals = [last_gt_intersection_list, last_est_intersection_list, to_store_dist, last_sel_ar, last_comp_ar, last_gt_res_ar, hausdorff_gt,
                          last_matched_gt_polys, last_matched_est_polys, last_poly_match_gt_indices, last_poly_match_est_indices,
                          all_rnn_to_compare_gt,all_rnn_to_compare_est, rnn_gt_indices, rnn_est_indices]
        
        
        # vis_tools.save_results_eval(seq_images.cpu().numpy(),outputs,  out[0], match_poly_indices, targets, static_inter_dict,None, static_target_ids, None, config,
        #                                gt_poly_indices=poly_indices, mod_ham=mod_ham, metric_visuals=metric_visuals)
    
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
    logging.error('SAVED CHECKPOINT')

def load_checkpoint(path, model, optimizer, scheduler, load_orig_ckpt=False):

    ckpt = torch.load(path)
    
    logging.error('LOADED ' + path)
    
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
        # with torch.no_grad():
        #     model.left_object_embed.weight.copy_(model.object_embed.weight)
     
#     Load optimiser state
        optimizer.load_state_dict(ckpt['optimizer'])
    
#         Load scheduler state
        scheduler.load_state_dict(ckpt['scheduler'])
        
        if 'iteration' not in ckpt.keys():
            to_return_iter = 0
        else:
            to_return_iter = ckpt['iteration']
        # to_return_iter = 0
        logging.error('LOADED MY')
        return ckpt['epoch'], ckpt['best_iou'],to_return_iter

def load_pretrained_backbone(path, model):
    
    ckpt = torch.load(path)

    model.load_state_dict(ckpt,strict=False)
  

# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_configuration()

    
    return config


def create_experiment(config,  resume=None):

    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
        
    else:
        
            
        # name = 'maxi_poly_loss_split_'+str(abs_bev) +'_big'+str(True)  +'_refineTrue'
        name = 'TR_MC_nusc'
        # Otherwise, generate a run directory based on the current time
        # name = datetime.now().strftime('{}_%y-%m-%d--%H-%M-%S').format('run')
        # name = 'maxi_poly_loss_split_'+str(abs_bev)
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

train_only_main_network = False

apply_poly_loss = True

split_pe = True

apply_bev_pe = True
abs_bev = True

only_bev_pe=False

euler=True

intersection_mode='polygon'

num_object_classes = 8


base_dir = '/cluster/work/cvl/cany/TPLR'


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

    model_name = 'tplr_nusc'
  

    parser = ArgumentParser()
    parser.add_argument('--dgx', type=bool, default=euler,
                    help='whether it is on dgx')
    parser.add_argument('--resume', default=base_dir+'/'+model_name, 
                        help='path to an experiment to resume')
   # base_dir+'/'+model_name

    
    parser.add_argument('--use_gt_labels', type=bool, default=True,
                    help='whether it is on dgx')
    
    parser.add_argument('--split_pe', type=bool, default=split_pe,
                    help='whether it is on dgx')
    
    parser.add_argument('--object_refinement', type=bool, default=False,
                    help='whether it is on dgx')
    
    parser.add_argument('--only_big', type=bool, default=False,
                    help='whether it is on dgx')
    
    parser.add_argument('--only_bev_pe', type=bool, default=only_bev_pe,
                    help='whether it is on dgx')
    # '/scratch_net/catweazle/cany/lanefinder/combined_objects_4'
    parser.add_argument('--bev_pe', type=bool, default=apply_bev_pe,
                    help='whether it is on dgx')
    parser.add_argument('--abs_bev', type=bool, default=abs_bev,
                    help='whether it is on dgx')
    
  
    parser.add_argument('--apply_poly_loss', type=bool, default=apply_poly_loss,
                    help='whether it is on dgx')
    
  
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
    logdir = create_experiment(config, args.resume)
    
    logging.error('LOGDIR ' + str(logdir))
    
    config.save_logdir = logdir
    config.n_control_points = args.num_spline_points
    
    config.freeze()
    
    summary = SummaryWriter(logdir)

    # Set default device
    # if len(config.gpus) > 0:
    #     torch.cuda.set_device(config.gpus[0])
    device = torch.device(args.device)
    # Setup experiment
    model, criterion, postprocessors = build(args, config,large_parameters)
    
    model.to(device)
    
    train_loader,train_dataset, val_loader, val_dataset = data_factory.build_nuscenes_dataloader(config, args)
 
    freeze_backbone_layers(model)
    
  
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n  and p.requires_grad]},
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
        # if args.resume:
#        epoch, best_iou, iteration = load_checkpoint(os.path.join(logdir, 'latest.pth'),
#                                      model, optimizer, lr_scheduler)
        
        epoch, best_iou, iteration = load_checkpoint(os.path.join(base_dir,'final_ckpts', 'TR_MC_Nusc.pth'),
                                  model,  optimizer, lr_scheduler)
    
  
    
        logging.error('LOADED MY CHECKPOINT')
    
    
    freeze_backbone_layers(model)
   
    confusion = NewBinaryConfusionMatrix()
            
    transforms = torchvision.transforms.RandomAffine(degrees=15,translate=(0.1,0.1),scale=(0.9,1.1) )
    
    # Main training loop
    while epoch <= config.num_epochs:
        
        print('\n\n=== Beginning epoch {} of {} ==='.format(epoch, 
                                                            config.num_epochs))
        
        iteration, confusion=train(train_loader, train_dataset,model, criterion, optimizer, postprocessors , summary,confusion, config,args, iteration, transforms)
        
        logging.error('EPOCH FINISHED')
        
        lr_scheduler.step()
        
        save_checkpoint(os.path.join(logdir, 'latest.pth'), model, optimizer, 
                        lr_scheduler, epoch, iteration,best_iou)
        # Evaluate on the validation set
        if epoch % 5 == 0:
            val_confusion = NewBinaryConfusionMatrix()
            val_con = evaluate(val_loader, model, criterion, postprocessors,val_confusion,summary, config, args,epoch)
    
            static_res_dict = val_con.get_res_dict
            file1 = open(os.path.join(logdir,'val_res_thresh_'+'.txt'),"a")
            
            for k in static_res_dict.keys():
                logging.error(str(k) + ' : ' + str(static_res_dict[k]))
                file1.write(str(k) + ' : ' + str(static_res_dict[k]) + ' \n')

            
            file1.close()  

            val_iou = val_con.order_error
            logging.error('VAL IOU ' + str(val_iou))
            logging.error('BEST IOU ' + str(best_iou))
            
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

                

