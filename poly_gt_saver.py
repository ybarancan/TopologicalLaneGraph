import logging
import os
from datetime import datetime
from argparse import ArgumentParser

from src.utils.configs import get_default_configuration_argo, load_config

from src.data import data_factory

import numpy as np

image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]


def train(dataloader,dataset, args, config):

    
    poly_dict = dict()
    total_passed_batch = 0
    for i, batch in enumerate(dataloader):
        
       
        if batch[-1]:
            logging.error('PASSED BATCH')
            total_passed_batch = total_passed_batch + 1
            continue
        

        seq_images, targets, _ = batch
        
       
        logging.error('SAMPLE ' + str(i) + ' TOKEN : ' + targets[0]['scene_token'])
  

        poly_dict[targets[0]['scene_token']+'_'+str(targets[0]['sample_token'])] = [np.copy(targets[0]['blob_mat'].numpy()),np.copy(targets[0]['poly_centers'].numpy()),np.copy(targets[0]['poly_one_hots'].numpy()),np.copy(targets[0]['blob_ids'].numpy())]
        
      
        
    file_name = args.dataset_name
    
    if args.train_split:
        file_name = file_name + '_train_'
    else:
        file_name = file_name + '_val_'
    
    file_name = file_name + 'gt_polygon_dict_' + str(args.interval_start)+'_' + str(args.interval_end) + '.npy'
    
    np.save(os.path.join(config.poly_base_path, file_name),poly_dict)
    return 



# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_configuration_argo()

    # Load dataset options
    logging.error('DGX OPTION ' + str(args.dgx))
    if args.dgx:
        logging.error('IT IS IN DGX')
        config.merge_from_file( '/cluster/home/cany/TPLR/configs/euler.yml')
    


    return config


def create_experiment(config,  resume=None):

    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
        
    else:
        
            
        # name = 'maxi_poly_loss_split_'+str(abs_bev) +'_big'+str(True)  +'_refineTrue'
        name = 'poly_saver'
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


dataset_name = 'nuscenes' # argoverse


mini_version = False

poly_pretrain = True

do_objects = False

euler=False


num_object_classes = 8

base_dir = '/cluster/work/cvl/cany/TPLR'

model_name = 'tplr'

def main():

   

    parser = ArgumentParser()
    parser.add_argument('--dgx', type=bool, default=euler,
                    help='whether it is on dgx')
    parser.add_argument('--resume', default=base_dir+'/'+model_name, 
                        help='path to an experiment to resume')

 
    parser.add_argument('--only_big', type=bool, default=False,
                    help='whether it is on dgx')
  
    
    parser.add_argument('--estimate_object_on_image', type=bool, default=False,
                    help='whether it is on dgx')
    

    parser.add_argument('--objects', type=bool, default=False,
                help='whether estimate objects')
    

    parser.add_argument('--num_object_queries', default=100, type=int,
                    help="Number of query slots")
    
    parser.add_argument('--num_pedestrian_queries', default=50, type=int,
                    help="Number of query slots")
    
    parser.add_argument('--num_cycle_queries', default=50, type=int,
                    help="Number of query slots")
    
    
    parser.add_argument('--num_object_classes', default=num_object_classes, type=int,
                help="Num object classes")
    
    parser.add_argument('--num_spline_points', default=3, type=int,
                help="Num object classes")

    # dataset parameters
  
    parser.add_argument('--num_workers', default=2, type=int)

    '''
    SET THE DATASET NAME
    '''


    parser.add_argument('--dataset_name', default=dataset_name, type=str)

    '''
    SET THE INTERVAL HERE 
    '''

    # PRETRAIN SAVER INTERVAL
    parser.add_argument('--interval_start', default=60, type=int,
                        help='number of distributed processes')
    parser.add_argument('--interval_end', default=65, type=int,
                        help='number of distributed processes')

    '''
    SET THE SPLIT TO WORK ON
    True : TRAIN, False : EVAL
    '''
    parser.add_argument('--train_split', type=bool, default=True,
                    help='whether it is on dgx')
    
    

    args = parser.parse_args()
    
    
    print('GOT ARGS ')
    logging.error(str(args))
    
    
    logging.error('START ' + str(args.interval_start) + ' END ' + str(args.interval_end))
    
    # Load configuration
    config = get_configuration(args)
    
    # Create a directory for the experiment
    logdir = create_experiment(config, args.resume)
    
    logging.error('LOGDIR ' + str(logdir))
    
    config.save_logdir = logdir
    config.n_control_points = args.num_spline_points
    
    config.poly_pretrain = poly_pretrain
    
    config.freeze()
    
    if args.dataset_name == 'nuscenes':
        train_loader,train_dataset, val_loader,val_dataset = data_factory.build_nuscenes_dataloader(config, args, val=(not args.train_split), gt_polygon_extraction=True)
    elif args.dataset_name == 'argoverse':
        
        train_loader,train_dataset, val_loader,val_dataset = data_factory.build_argoverse_dataloader(config, args, val=(not args.train_split), gt_polygon_extraction=True)
    
    else:
        logging.error('DATASET ' + str(args.dataset_name) + ' is not supported')
    
    if args.train_split:
    
        train(train_loader, train_dataset,args, config)
    else:
        train(val_loader, val_dataset,args, config)
    
    exit()    
    



if __name__ == '__main__':
    main()

                

