# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:15:40 2019

@author: yyk17
"""
import numpy as np
import cv2
import scipy.io as sio
import os
import numpy as np

path= os.getcwd()+'/'
names='bunny'
showtype='test'

data=np.load(path+names+'/'+showtype+'_data_x.npy')
images=data

print(images.shape)
frame_num=len(images[0,0,:])

image_resize=np.zeros([128,128,frame_num],dtype='uint8')

images=np.transpose(images,[1,0,2])

for i in range(frame_num):
    #print(images[:,:,i].shape)
    image_resize[:,:,i]=cv2.resize(images[:,:,i],(128,128),interpolation=cv2.INTER_LINEAR)



image = image_resize
print('image_shape:',image.shape)


#print(image.shape)

#result_path='/home/yyk17/yangyk/TianjiChip/Quan_NFS_result/'
result_path=path+'result/'
position=sio.loadmat(result_path+names+'/ann_'+names+'.mat')
position=position['data']
position=position[:,4:10]
position=position.astype(np.uint16)
position=np.transpose(position)

num_index=0
frame_num=len(image[0,0,:])
frames=np.min([frame_num,len(position[0,:])])

print('Frame Number:%d'%(frames))

for i in range(frames):
    
    x = position[0,num_index]
    y = position[1,num_index]
    w = position[2,num_index]
    h = position[3,num_index]
    print(x,y,w,h)
    frame_index=position[5,num_index]-position[5,0]
    
    image_show=np.zeros([128,128,3],dtype='uint8')
    #image_show=np.zeros([240,180,3],dtype='uint8')
    image_show[:,:,0]=image[:,:,frame_index]
    image_show[:,:,1]=np.round(0.7*image[:,:,frame_index])
    image_show[:,:,2]=np.round(0.6*image[:,:,frame_index])
    #print(x,y,w,h)

    #image1=image1.astype(np.uint8)
    #image_show=np.transpose(image_show,[1,0,2])
    
    
    cv2.rectangle(image_show,(np.round(x-w/2),np.round(y-h/2),w,h),(0,255,255),2)
    #cv2.rectangle(image_show,(np.round(x),np.round(y+h),w,h),(0,255,255),2)
    cv2.imshow('image',image_show)
    if (cv2.waitKey(30) & 0xFF) == ord('q'):
        a=1
    num_index=num_index+1
    
