# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from src.detr.util.misc import NestedTensor, is_main_process

from src.detr.position_encoding import build_position_encoding
from src.deeplab.net import SPPNet

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# class Backbone(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.deeplab = SPPNet()
        
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # self.num_channels = num_channels

    # def forward(self, images):
    #     xs = self.body(images)
    #     out = {}
    #     for name, x in xs.items():
          
    #         out[name] = x
    #     return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.backbone_net = backbone
    def forward(self, images, calib,abs_bev=True):
        xs, low = self[0](images)
        out= []
        low_out = []
        pos = []
        bev_pos = []
        # for name, x in xs.items():
        out.append(xs)
        low_out.append(low)
        # position encoding
        pos.append(self[1](xs, bev=False).to(xs.dtype))
        bev_pos.append(self[1](xs, calib,bev=True, abs_bev=abs_bev).to(xs.dtype))

        return out, low_out,pos, bev_pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # train_backbone = args.lr_backbone > 0
    # return_interm_layers = args.masks
    # backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    backbone = SPPNet()
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
