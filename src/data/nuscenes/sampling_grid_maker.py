#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:07:33 2021

@author: cany
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from nuscenes import NuScenes
from torchvision.transforms.functional import to_tensor
import cv2
from .utils import CAMERA_NAMES, NUSCENES_CLASS_NAMES, iterate_samples
from ..utils import decode_binary_labels
from src.utils import bezier
import random
from src.data.nuscenes import utils as nusc_utils

import numpy as np
import scipy.interpolate as si 
# from scipy.interpolate import UnivariateSpline
import logging
# import pwlf
from math import factorial
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']


camera_matrix_dict = np.load('/scratch_net/catweazle/cany/lanefinder/camera_matrix_dict.npy', allow_pickle=True)   

intrinsic_dict = np.load('/scratch_net/catweazle/cany/lanefinder/intrinsic_dict.npy', allow_pickle=True)   

augment_steps=[0.5,1,1.5,2]
my_dict = dict()

for loc in intrinsic_dict.item().keys():
    my_list = []
    for k in augment_steps:

        write_row, write_col, total_mask = nusc_utils.zoom_augment_grids((900,1600,3),intrinsic_dict.item().get(loc),
                                                                                 camera_matrix_dict.item().get(loc)[:3,-1], k)
        
        
        temp = np.stack([write_row.flatten(),write_col.flatten(),total_mask.flatten()],axis=-1)
        my_list.append(np.copy(temp))
        
    my_dict[loc] = np.copy(np.stack(my_list,axis=0))



np.save('/scratch_net/catweazle/cany/lanefinder/zoom_sampling_dict.npy', my_dict)