import math
import torch
import torch.nn as nn
import logging
from .transformer import DenseTransformer

class TransformerPyramid(nn.Module):

    def __init__(self, in_channels, channels, resolution, extents, ymin, ymax, 
                 focal_length):
        
        super().__init__()
        self.transformers = nn.ModuleList()

#        for i in range(5):
        i = 1 #1/32
            # Scaled focal length for each transformer
            
          

        focal_x = focal_length[0] / pow(2, i + 3)
        focal_y = focal_length[1] / pow(2, i + 3)
        # Compute grid bounds for each transformer
        zmax = min(math.floor(focal_y * 2) * resolution, extents[3])
        zmin = math.floor(focal_y) * resolution if i < 4 else extents[1]
        subset_extents = [extents[0], zmin, extents[2], zmax]

        # Build transformers
        self.transformer = DenseTransformer(in_channels, channels, resolution, 
                               subset_extents, ymin, ymax, (focal_x,focal_y))
#        self.transformers.append(tfm)
    

    def forward(self, fmap, calib):
        
#        bev_feats = list()
#        for i, fmap in enumerate(feature_maps):
        i=1   
        
        # Scale calibration matrix to account for downsampling
        scale = 8 * 2 ** i
        calib_downsamp = calib.clone()
        calib_downsamp[:, :2] = calib[:, :2] / scale
#        logging.error('PYRAMID CALIB : ' + str(calib.shape))
#        logging.error('PYRAMID CALIBDOWN: ' + str(calib_downsamp.shape))
        # Apply orthographic transformation to each feature map separately
#        bev_feats.append(self.transformers[i](fmap, calib_downsamp))
        return self.transformer(fmap, calib_downsamp)
        # Combine birds-eye-view feature maps along the depth axis
#        return torch.cat(bev_feats[::-1], dim=-2)
