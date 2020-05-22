# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:13:57 2019

@author: yyk17
"""

import torch
import scipy.io as sio
import os

checkpoint=torch.load(os.getcwd()+'/checkpoint/ckptbunny.t7')
weights=checkpoint['net']

conv1_weight=weights['conv1.weight'].data.cpu().numpy()
conv2_weight=weights['conv2.weight'].data.cpu().numpy()
conv3_weight=weights['conv3.weight'].data.cpu().numpy()
conv4_weight=weights['conv4.weight'].data.cpu().numpy()
conv5_weight=weights['conv5.weight'].data.cpu().numpy()

fc1_weight=weights['fc1.weight'].data.cpu().numpy()
fc2_weight=weights['fc2.weight'].data.cpu().numpy()

if os.path.exists(os.getcwd()+'/data') == 0:
   os.makedirs(os.getcwd()+'/data')
   print('creat_path:', os.getcwd()+'/data')

sio.savemat(os.getcwd()+'/data/bunnyweight.mat',{'conv1_weight':conv1_weight,\
   'conv2_weight':conv2_weight,'conv3_weight':conv3_weight,'conv4_weight':conv4_weight,\
   'conv5_weight':conv5_weight,'fc1_weight':fc1_weight,'fc2_weight':fc2_weight})








