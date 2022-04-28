import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
#from .. import utils

class Resampler(nn.Module):

    def __init__(self, resolution, extents):
        super().__init__()

        # Store z positions of the near and far planes
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents)


    def forward(self, features, calib):

        # Copy grid to the correct device
        self.grid = self.grid.to(features)
        
        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # is zero, so we only need a 2x2 submatrix of the original 3x3 matrix
        # calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)
        
        # to_add = torch.matmul(calib.squeeze(), torch.tensor(np.expand_dims(np.array([0,1.5,1]), axis=-1)).float().cuda())
        # to_add = to_add[1]/to_add[2]
        # logging.error('TO ADD ' + str(to_add))
        # to_add = features.size(-2) - to_add
        
        calib = calib.view(-1, 1, 1, 3, 3)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / cam_coords[..., -1]
        # logging.error('U COORDS ' + str(torch.min(ucoords)) +', '+ str(torch.max(ucoords)))
        
        
        
        ucoords = ucoords / features.size(-1) * 2 -1
        # logging.error('U COORDS ' + str(torch.min(ucoords)) +', '+ str(torch.max(ucoords)))
        
        # Normalize z coordinates
        zcoords = cam_coords[..., 1] / cam_coords[..., -1]
        
        # zcoords = zcoords + to_add
        zcoords = zcoords/ features.size(-2) * 2 - 1
        
        # zcoords = (zcoords-self.near) / (self.far-self.near) * 2 - 1
        # logging.error('Z COORDS ' + str(torch.min(zcoords)) +', '+ str(torch.max(zcoords)))
        
        # logging.error('Z COORDS ' + str(zcoords))
        
        # Resample 3D feature map
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


def _make_grid(resolution, extents):

    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    z1 = z1 + 2
    z2 = z2 + 2
    # (z2 + z1) - 
    zz, xx = torch.meshgrid(
        (z2 + z1) -torch.arange(z1, z2, resolution), torch.arange(x1 , x2, resolution))
    
    # logging.error('ZZ')
    # for k in range(len(zz)):
    #     logging.error(str(zz))
        
    # logging.error('XX')
    # for k in range(len(xx)):
    #     logging.error(str(xx))
    
    return torch.stack([xx, 1.7*torch.ones_like(xx), zz], dim=-1)