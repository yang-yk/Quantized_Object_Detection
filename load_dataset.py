# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:14:11 2018

@author: yjwu
"""
from __future__ import print_function
import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import h5py
import cv2

class MyDataset(data.Dataset):
    def __init__(self, path='load_test.mat',method = 'h',wins = 24):
        if method=='h':
            data = h5py.File(path)
            image,label = data['image'],data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image[:,:,:,:,:wins]).float()
            self.labels = torch.from_numpy(label).float()


        elif method == 'train_aps':
            #path_x = path + '/train_aps_x.mat'
            #path_y = path + '/train_aps_y.mat'
            #image, label = h5py.File(path_x)['train_aps_x'], h5py.File(path_y)['train_aps_y']

            path_x = path + '/train_data_x.npy'
            path_y = path + '/train_data_y.npy'
            image = np.load(path_x)
            label = np.load(path_y)


            for y in range(len(label[1, :])):
                # print(data_y[:,y])
                label[0, y] = np.round(label[0, y] * 128 / 240)
                label[1, y] = np.round(label[1, y] * 128 / 180)
                label[2, y] = np.round(label[2, y] * 128 / 240)
                label[3, y] = np.round(label[3, y] * 128 / 180)
                label[4, y] = label[4, y]

            train_images = np.zeros([128, 128, image.shape[2]])
            from skimage.transform import resize
            for id in range(train_images.shape[2]):
                train_images[:, :, id] = resize(image[:, :, id], (128, 128)) * 255



            images = torch.from_numpy(np.array(train_images)).float()

            self.images = images.permute([2, 0, 1]) # todo
            self.images = self.images.view(-1, 1, 128, 128) # todo
            labels = torch.from_numpy(np.array(label)).float()

            self.labels = labels.permute([1, 0]) # todo


        elif method == 'test_aps':
            #path_x = path + '/test_aps_x.mat'
            #path_y = path + '/test_aps_y.mat'
            #image, label = h5py.File(path_x)['test_aps_x'], h5py.File(path_y)['test_aps_y']
            #image_ = np.array(image)

            path_x = path + '/test_data_x.npy'
            path_y = path + '/test_data_y.npy'
            image = np.load(path_x)
            label = np.load(path_y)

            import scipy.misc


            x = label[:,0][0]
            y = label[:, 0][1]
            w = label[:, 0][2]
            h = label[:, 0][3]
            img0 = np.transpose(image[:,:,0])

            #import matplotlib as plt
            #plt.figure()
            #plt.imshow(img0)
            #plt.Rectangle((x, y), w, h)
            #plt.savefig('test.png')
            import cv2
            image_show = np.zeros([180, 240, 3], dtype='uint8')
            image_show[:, :, 0] = img0[:, :]
            image_show[:, :, 1] = np.round(0.7 * img0[:, :])
            image_show[:, :, 2] = np.round(0.6 * img0[:, :])

            cv2.rectangle(image_show,(np.int(np.round(x - w / 2)), np.int(np.round(y - h / 2)), w, h), (0, 255, 255), 2)
            #cv2.rectangle(np.transpose(image_show,[1,0,2]), (np.int(np.round(x - w / 2)), np.int(np.round(y - h / 2)), w, h), (0, 255, 255), 2)
            #cv2.imshow('image', image_show)
            cv2.imwrite('/home/yyk17/yangyk/TianjiChip/quan_final/Quan_test0310/test.jpg', image_show)

            for y in range(len(label[1, :])):
                # print(data_y[:,y])
                label[0, y] = np.round(label[0, y] * 128 / 240)
                label[1, y] = np.round(label[1, y] * 128 / 180)
                label[2, y] = np.round(label[2, y] * 128 / 240)
                label[3, y] = np.round(label[3, y] * 128 / 180)
                label[4, y] = label[4, y]

            from skimage.transform import resize

            train_images = np.zeros([128,128,image.shape[2]])
            for id in range(train_images.shape[2]):
                train_images[:,:,id] = resize(image[:,:,id], (128, 128))*255


            x = label[:, 0][0]
            y = label[:, 0][1]
            w = label[:, 0][2]
            h = label[:, 0][3]

            img0 = np.transpose(train_images[:, :, 0])
            import cv2
            image_show = np.zeros([128, 128, 3], dtype='uint8')
            image_show[:, :, 0] = img0[:, :]
            image_show[:, :, 1] = np.round(0.7 * img0[:, :])
            image_show[:, :, 2] = np.round(0.6 * img0[:, :])

            cv2.rectangle(image_show, (np.int(np.round(x - w / 2)), np.int(np.round(y - h / 2)), w, h), (0, 255, 255),
                          2)
            cv2.imwrite('/home/yyk17/yangyk/TianjiChip/quan_final/Quan_test0310/test_resized.jpg', image_show)



            for id in range(image.shape[2]):
                train_images[:,:,id] = np.uint8(resize(image[:,:,id], (128, 128))*255)




            images = torch.from_numpy(np.array(train_images)).float()

            self.images = images.permute([2,0,1]) # todo
            self.images = self.images.view(-1, 1, 128, 128) # todo
            labels = torch.from_numpy(np.array(label)).float()
            self.labels = labels.permute([1,0]) # todo



        elif method == 'test_dvs':
            path_x = path + '/test_dvs_x.mat'
            path_y = path + '/test_dvs_y.mat'
            image, label = h5py.File(path_x)['test_dvs_x'], h5py.File(path_y)['test_dvs_y']

            images = torch.from_numpy(np.array(image)).float()
            labels = torch.from_numpy(np.array(label)).float()

            self.images = images.permute([4, 1, 2, 3, 0])
            self.labels = labels.permute([1,0])


        elif method == 'train_dvs':
            path_x = path + '/train_dvs_x.mat'
            path_y = path + '/train_dvs_y.mat'
            image, label = h5py.File(path_x)['train_dvs_x'], h5py.File(path_y)['train_dvs_y']

            images = torch.from_numpy(np.array(image)).float()
            labels = torch.from_numpy(np.array(label)).float()

            self.images = images.permute([4, 1, 2, 3, 0])
            self.labels = labels.permute([1,0])


        else:
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image']).float()
            self.labels = torch.from_numpy(data['label']).float()
            self.images = self.images[:,:,:,:,:wins]

        self.num_sample = int((len(self.images)//100)*100)
        print(self.images.size(),self.labels.size())

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample