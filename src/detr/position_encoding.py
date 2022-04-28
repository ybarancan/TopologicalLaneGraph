# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from src.detr.util.misc import NestedTensor
import logging

# range_x = np.arange(50)
# range_y = np.arange(28)
# cur_y,cur_x  = np.meshgrid(range_y,range_x, indexing='ij')
# cam_height = 1.7
 
# f = 800/32
# y_center = 14
# y_embed = cam_height*f/(cur_y - y_center + 0.1)
# x_embed = (y_embed*cur_x - 25*y_embed)/f

# to_remove = (y_embed < 0) | (y_embed > 50)
            





# x_embed[y_embed < 0] = 0
# x_embed[y_embed > 50] = 0

# y_embed[y_embed < 0] = 0
# y_embed[y_embed > 50] = 0

# #   y_embed = torch.log(y_embed.clamp(-50,50)/50 + 2)
# # x_embed = torch.log((x_embed.clamp(-40,40) + 40)/40 + 1)

 
# # y_embed = y_embed.unsqueeze(0).cumsum(1, dtype=torch.float32) 
# # x_embed = x_embed.unsqueeze(0).cumsum(2, dtype=torch.float32) 

# y_embed = np.clip(y_embed,-50,50)/50 + 1
# x_embed = np.clip(x_embed,-40,40)/40 + 1

# x_embed[to_remove] = 0
# x_embed[to_remove] = 0

# y_embed[to_remove] = 0
# y_embed[to_remove] = 0

# y_embed = y_embed[::-1,:]


# y_embed = np.cumsum(y_embed,0) 
# x_embed = np.cumsum(x_embed,1) 



# eps = 1e-6
# y_embed = y_embed[::-1,:]
# y_embed = y_embed / (y_embed[:1, :] + eps) 
# x_embed = x_embed / (x_embed[:, -1:] + eps) 

# x_embed[to_remove] = 1
# y_embed[to_remove] = 1

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, split=False,temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        self.split = split
        if self.split:
            self.num_pos_feats = self.num_pos_feats/2
        

    def forward(self,x,calib=None, bev=False, abs_bev=True):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask
        
        if bev:
            range_x = torch.arange(50).cuda()
            range_y = torch.arange(28).cuda()
            cur_y, cur_x = torch.meshgrid(range_y,range_x)
            cam_height = 1.7
         
            f = calib[0,0] + 1e-02
            y_center = calib[1,-1]
            cur_h = cur_y - y_center + 0.1
            cur_h[torch.abs(cur_h) < 0.1] = 0.1
            # if torch.abs(cur_h) < 0.1:
            #     y_embed = cam_height*f/(0.1)
            # else:
            y_embed = cam_height*f/(cur_h)
                
            x_embed = (y_embed*cur_x - calib[0,-1]*y_embed)/f
            
            to_remove = (y_embed < 0) | (y_embed > 50)
            
            x_embed[y_embed < 0] = 0
            x_embed[y_embed > 50] = 0
            
            y_embed[y_embed < 0] = 0
            y_embed[y_embed > 50] = 0
            # logging.error('Y EMBED FIRST')
            # # logging.error(str(y_embed))
            
            # for k in range(len(y_embed)):
            #     logging.error(str(y_embed[k]))
            # logging.error('Y EMBED ')
            # logging.error(str(y_embed))
            # logging.error(str(torch.max(y_embed)))
            # logging.error(str(torch.min(y_embed)))
            # logging.error('X EMBED ')
            # logging.error(str(x_embed))
            # logging.error(str(torch.max(x_embed)))
            # logging.error(str(torch.min(x_embed)))
            
            # logging.error('Y EMBED')
            # logging.error(str(y_embed))
            
            # for k in range(len(y_embed)):
            #     logging.error(str(y_embed[k]))
            
            # logging.error('X EMBED')
            # logging.error(str(x_embed))
            
            if abs_bev:
                # y_embed = y_embed.clamp(-60,60)/60 + 2
                # x_embed = (x_embed.clamp(-40,40) + 40)/40 + 1
                
                                
                y_embed = y_embed.clamp(-50,50)/50 + 2
                x_embed = x_embed.clamp(-40,40)/40 + 2
                
                x_embed[to_remove] = 1
                x_embed[to_remove] = 1
                
                y_embed[to_remove] = 1
                y_embed[to_remove] = 1
                
                y_embed = torch.flip(y_embed,dims=[0])
                
                # logging.error('Y EMBED AFTER FIRST FLIP')
                # # logging.error(str(y_embed))
                
                # for k in range(len(y_embed)):
                #     logging.error(str(y_embed[k]))
                    
                x_embed = torch.log(x_embed)
                
                y_embed = torch.log(y_embed)
                
                y_embed = y_embed.unsqueeze(0).cumsum(1, dtype=torch.float32) 
                x_embed = x_embed.unsqueeze(0).cumsum(2, dtype=torch.float32)
                
                eps = 1e-4
                y_embed = torch.flip(y_embed,dims=[1])
                y_embed = y_embed / (y_embed[:,:1, :] + eps) 
                x_embed = x_embed / (x_embed[:,:, -1:] + eps) 
                
                x_embed[0,to_remove] = 1
                y_embed[0,to_remove] = 1
                
                x_embed = x_embed * self.scale
                y_embed = y_embed * self.scale

            else:
                                
                y_embed = y_embed.clamp(-50,50)/50 + 1
                x_embed = x_embed.clamp(-40,40)/40 + 1
                
                x_embed[to_remove] = 0
                x_embed[to_remove] = 0
                
                y_embed[to_remove] = 0
                y_embed[to_remove] = 0
                
                y_embed = torch.flip(y_embed,dims=[0])
                
                # logging.error('Y EMBED AFTER FIRST FLIP')
                # # logging.error(str(y_embed))
                
                # for k in range(len(y_embed)):
                #     logging.error(str(y_embed[k]))
                
                
                y_embed = y_embed.unsqueeze(0).cumsum(1, dtype=torch.float32) 
                x_embed = x_embed.unsqueeze(0).cumsum(2, dtype=torch.float32)
                
                eps = 1e-6
                y_embed = torch.flip(y_embed,dims=[1])
                y_embed = y_embed / (y_embed[:,:1, :] + eps) 
                x_embed = x_embed / (x_embed[:,:, -1:] + eps) 
                
                x_embed[0,to_remove] = 1
                y_embed[0,to_remove] = 1
                
                x_embed = x_embed * self.scale
                y_embed = y_embed * self.scale
            
        
            # logging.error('Y EMBED FIN')
            # # logging.error(str(y_embed))
            
            # for k in range(len(y_embed[0])):
            #     logging.error(str(y_embed[0,k]))
            
        
            # logging.error('X EMBED')
            # # logging.error(str(y_embed))
            
            # for k in range(len(x_embed[0])):
            #     logging.error(str(x_embed[0,k]))
            
        else:
            not_mask = torch.ones_like(x)
            not_mask = not_mask[:,0,...]
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
                
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale



        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, split=args.split_pe, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
