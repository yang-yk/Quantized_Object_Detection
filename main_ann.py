# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:52:58 2019

@author: yangyk
"""
import sys
from load_dataset import*
import torch,time,os,random
import scipy.io as io
from iou_func import *
from torchsummary import summary
import cv2
import os
#############################################################################################################
# todo
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from ann_model import*
names='bunny'
print('todo: test ' + names + 'datasets')

#############################################################################################################


### load dataset
data_path = os.getcwd()+'/' + names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# todo
test_dataset = MyDataset(data_path, 'test_aps')

train_dataset = MyDataset(data_path, 'train_aps')


# default loading file
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last = True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,drop_last = True)


best_acc = 0
acc_record = list([])

model = ANN_model()
model.to(device)
summary(model, input_size=(1, 128, 128))
criterion = nn.MSELoss() # choose loss function, e.g. crossentroye, Mse
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # optimizer, e.g. Adam (lr = 1e-3), SGD (lr = 0.1-0.3)

# training
best_iou = 0.
for epoch in range(num_epochs):
    running_loss = running_iou = 0.
    start_time = time.time()
    model.train() # if u use dropout, open training process
    for i, (images, labels_) in enumerate(train_loader):
        labels = labels_[:, 0:4]/128
        images.float()
        model.zero_grad(), optimizer.zero_grad()
        optimizer.zero_grad()
        images=torch.tensor(images)
        images = images.float().to(device)
        outputs = model(images)
        loss = criterion(outputs.cpu(), labels)
        loss.backward()
        optimizer.step()

        iou_value = box_iou(outputs, labels)
        running_loss += loss.item()
        running_iou += iou_value.item()

    print('Training - Epoch [%d/%d], Step [%d/%d], Loss: %.8f, Iou : %.8f'
          % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss / (i + 1),
             running_iou / (i + 1)))
    print('Time elasped:', time.time() - start_time)
    correct = total = 0.



    ### testing
    running_loss = running_iou =  0.
    start_time = time.time()

    output_data = []
    model.eval()  # close  training process

    for i, (images, labels) in enumerate(test_loader):

        model.zero_grad()
        optimizer.zero_grad()

        images = cv2.resize(np.array(images[0, 0, :, :].cpu()), (128, 128), interpolation=cv2.INTER_LINEAR)
        images = images.reshape([1, 1, 128, 128])
        images = torch.tensor(images)

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

    print('iou_value', float(running_iou) / (i + 1))

    #save_result_path = '/home/yyk17/yangyk/TianjiChip/Quan_NFS_result/' + names
    #if os.path.exists(save_result_path) == 0:
    #    os.makedirs(save_result_path)
    #    print('create_path:', save_result_path)

    #io.savemat('/home/yyk17/yangyk/TianjiChip/Quan_NFS_result/' + names + '/ann_' + names + '.mat', {'data': output_data.detach().numpy()})
    #io.savemat('/home/yyk17/yangyk/TianjiChip/NFS_result/' + names + '/ann_' + names + '.mat', {'data': output_data})


    if epoch > 1 and running_iou >= best_iou:
        print(running_iou)
        best_iou = running_iou
        print('Saving Data......')

        save_result_path = os.getcwd()+'/result/' + names
        if os.path.exists(save_result_path) == 0:
           os.makedirs(save_result_path)
           print('create_path:', save_result_path)

        io.savemat(save_result_path + '/ann_' + names + '.mat', {'data': output_data.detach().numpy()})



        state = {
            'net': model.state_dict(),

            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
