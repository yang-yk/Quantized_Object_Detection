# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:11:34 2019

@author: yyk17
"""

import torch
from load_dataset import*
import torch,time,os,random
from iou_func import *
import scipy.io as sio


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from ann_model import*

names='bunny'
CHECKPOINT_PATH='./checkpoint/ckpt'+names+'.t7'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = os.getcwd()+'/' + names


# todo
test_dataset = MyDataset(data_path, 'test_aps')

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,drop_last = True)

model = ANN_model()

checkpoint=torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['net'])
model.to(device)


running_loss = running_iou =  0.
start_time = time.time()
output_data = []
model.eval()

for i, (images, labels) in enumerate(test_loader):
        
        

        model.zero_grad()
        #optimizer.zero_grad()
        images = images.float().to(device)
        outputs = model(images)

        targets = labels[:, 0:4] / 128

        iou_value = box_iou(targets, outputs, types=None) # calculate box_iou value

        running_iou += iou_value.mean().item()

        # save experiment results
        temp = torch.cat((box_scale(targets).cpu(), box_scale(outputs).cpu(), iou_value.cpu(),
                          labels[:, 4].view(batch_size, 1) ),
                         dim=1).detach_()

        if i == 0:
            output_data = temp
        else:
            output_data = torch.cat((output_data, temp), dim=0)
        
        if i==50:
            print('showing features')
            output,features=model.show_feature(images)
            print('Done')

print('iou_value', float(running_iou) / (i + 1))

feature_name=['input_image','input','conv1','avg_pool1','conv2','avg_pool2','conv3','avg_pool3','conv4',\
             'conv5','fc_input','fc1','fc2']

for i in range(len(feature_name)):
    features[feature_name[i]]=features[feature_name[i]].data.cpu().numpy()

if os.path.exists(os.getcwd()+'/data') == 0:
   os.makedirs(os.getcwd()+'/data')
   print('creat_path:', os.getcwd()+'/data')

sio.savemat(os.getcwd()+'/data/bunnyfeatures.mat',{feature_name[0]:features[feature_name[0]],\
                                        feature_name[1]:features[feature_name[1]],\
                                        feature_name[2]:features[feature_name[2]],\
                                        feature_name[3]:features[feature_name[3]],\
                                        feature_name[4]:features[feature_name[4]],\
                                        feature_name[5]:features[feature_name[5]],\
                                        feature_name[6]:features[feature_name[6]],\
                                        feature_name[7]:features[feature_name[7]],\
                                        feature_name[8]:features[feature_name[8]],\
                                        feature_name[9]:features[feature_name[9]],\
                                        feature_name[10]:features[feature_name[10]],\
                                        feature_name[11]:features[feature_name[11]],\
                                        feature_name[12]:features[feature_name[12]]})
    
    
    
    
    
