import os
import torch
from torch.utils.data import DataLoader, RandomSampler

from nuscenes import NuScenes
from .nuscenes.dataset import NuScenesMapDataset
from .nuscenes.splits import TRAIN_SCENES, VAL_SCENES, CALIBRATION_SCENES

from argoverse.data_loading.argoverse_tracking_loader \
      import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap


from .argoverse.dataset import ArgoverseMapDataset
from .argoverse.splits import TRAIN_LOGS, VAL_LOGS
ALL_LOGS = TRAIN_LOGS + VAL_LOGS


from nuscenes.map_expansion.map_api import NuScenesMap
from src.data.nuscenes import utils as nusc_utils
import logging

def build_nuscenes_datasets(config,args, val=False, pinet=False, polyline=False, interval_val=False, gt_polygon_extraction=False):
    print('==> Loading NuScenes dataset...')
    nuscenes = NuScenes(config.nuscenes_version, 
                        os.path.expandvars(config.nusc_root))


    my_map_apis = { location : NuScenesMap(os.path.expandvars(config.nusc_root), location) 
             for location in nusc_utils.LOCATIONS }
    
    if gt_polygon_extraction:
        
        if not val:
            temp_scenes = TRAIN_SCENES + CALIBRATION_SCENES
            train_data = NuScenesMapDataset(nuscenes, config, my_map_apis, 
                                  temp_scenes[args.interval_start:args.interval_end]   , polyline=polyline, gt_polygon_extraction=gt_polygon_extraction)
            
            return train_data, None
    
        else:
            val_data = NuScenesMapDataset(nuscenes, config,my_map_apis, 
                                 VAL_SCENES[args.interval_start:args.interval_end], pinet=False, val=True , gt_polygon_extraction=gt_polygon_extraction)
            
            return None, val_data
    
    else:
        if not val:
            train_data = NuScenesMapDataset(nuscenes, config, my_map_apis, 
                                         TRAIN_SCENES, polyline=polyline, gt_polygon_extraction=gt_polygon_extraction)
            val_seqs = CALIBRATION_SCENES
        else:
            train_data=None
    
            val_seqs = VAL_SCENES
        
        val_data = NuScenesMapDataset(nuscenes, config,my_map_apis, 
                                     val_seqs, pinet=False, val=True , gt_polygon_extraction=gt_polygon_extraction)
        return train_data, val_data

def build_argoverse_datasets(config,args, val=False, pinet=False, gt_polygon_extraction=False):
      print('==> Loading Argoverse dataset...')
      dataroot = os.path.expandvars(config.nusc_root)
      trackroot = os.path.join(dataroot, 'argoverse-tracking')
      am = ArgoverseMap()
      # Load native argoverse splits
    
      loaders = {
          'train' : ArgoverseTrackingLoader(os.path.join(trackroot, 'all_logs')),
          # 'val' : ArgoverseTrackingLoader(os.path.join(trackroot, 'all_logs'))
      }
      # if gt_polygon_extraction:
      #     if not val:
      #         train_data = ArgoverseMapDataset(config, loaders['train'], am,
      #                                  TRAIN_LOGS[args.interval_start:args.interval_end],  train=True, pinet=False, gt_polygon_extraction=gt_polygon_extraction)
          
      #         return train_data, None
      #     else:
      #         val_data = ArgoverseMapDataset(config,loaders['train'], am, 
      #                                 VAL_LOGS[args.interval_start:args.interval_end], train=False, pinet=False, gt_polygon_extraction=gt_polygon_extraction)
      #         return None, val_data
          
            
      # else:
          
      train_data = ArgoverseMapDataset(config, loaders['train'], am,
                                   TRAIN_LOGS,  train=True, pinet=False, gt_polygon_extraction=gt_polygon_extraction)
      val_data = ArgoverseMapDataset(config,loaders['train'], am, 
                                  VAL_LOGS, train=False, pinet=False)
  
  
      return train_data, val_data

def my_collate(batch):
    # to_return = []
    # if batch is list:
    #     for b in range(len(batch)):
    #         for k in range(len(batch[b])):
        
    # logging.error('COLLATE ' + str(len(batch)))
    problem = False
    for b in range(len(batch)):
        problem = problem | batch[b][-1]
    if problem:
        return (None,None,True)
    
    else:    
        
        images = []
        targets = []
        for b in range(len(batch)):
            images.append(batch[b][0])
            targets.append(batch[b][1])
            
        return (torch.stack(images,dim=0), targets, False)


def build_argoverse_dataloader(config,args, val=False, pinet=False, gt_polygon_extraction=False):
    
      train_data, val_data = build_argoverse_datasets(config,args, val=val, pinet=pinet, gt_polygon_extraction=gt_polygon_extraction)
         
      if gt_polygon_extraction:
          if val:
              val_loader = DataLoader(val_data, 1, collate_fn = my_collate,
                              num_workers=1)
              
              return val_loader, val_data, val_loader,val_data
      
          else:
                train_loader = DataLoader(train_data, 1,   collate_fn = my_collate,
                                 num_workers=1)
                
                return train_loader, train_data, train_loader, train_data
            
            
      else:
          sampler = RandomSampler(train_data, False)
          train_loader = DataLoader(train_data, config.batch_size, sampler=sampler,  collate_fn = my_collate,
                                     num_workers=1)
        
          val_loader = DataLoader(val_data, 1, collate_fn = my_collate,
                                  num_workers=1)
          # val_loader, val_data
          return train_loader, train_data, val_loader,val_data
  
def build_nuscenes_dataloader(config,args, val=False, pinet=False,polyline=False,  gt_polygon_extraction=False):
    
    train_data, val_data = build_nuscenes_datasets(config,args, val=val, pinet=pinet, polyline=polyline, gt_polygon_extraction=gt_polygon_extraction)
    
    if gt_polygon_extraction:
        if val:
          val_loader = DataLoader(val_data, 1, collate_fn = my_collate,
                          num_workers=1)
          
          return val_loader, val_data, val_loader,val_data
  
        else:
            train_loader = DataLoader(train_data, 1,   collate_fn = my_collate,
                             num_workers=1)
            
            return train_loader, train_data, train_loader, train_data
        
    
    
    # sampler=sampler
    if not val:
        sampler = RandomSampler(train_data, False)
        train_loader = DataLoader(train_data, config.batch_size, sampler=sampler, collate_fn = my_collate,
                              num_workers=config.num_workers)
    else:
        train_loader=None
    val_loader = DataLoader(val_data, 1, collate_fn = my_collate,
                            num_workers=config.num_workers)
    
    return train_loader,train_data, val_loader, val_data
    

