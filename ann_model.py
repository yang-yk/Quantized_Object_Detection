import torch.nn as nn
import torch.nn.functional as F
from lin_layer import *
from lin_connect import *

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


batch_size = 1
num_epochs = 20

# todo
cfg = [(1, 16, 2, 0, 3), # in_planes, out_planes, stride, padding, kernel_size
       #MP2
       (16, 32, 1, 0, 3),
       #MP2
       (32, 64, 1, 0, 3),
       #MP2
       (64, 64, 1, 0, 3),
       # MP2
        (64, 128, 2, 0, 3),
       ]

fc = [128, 4]

# todo
probs = 0.0 # dropout
print('comment:',probs)

class ANN_model(nn.Module):

    def __init__(self):
        super(ANN_model, self).__init__()

        self.quanA=LinQuant(bit_width=8, with_sign=True, lin_back=True)

        in_planes, out_planes, stride, padding, kernel_size = cfg[0]
        self.conv1 = QuantConv2d(in_planes, out_planes, stride = stride, padding = padding, kernel_size = kernel_size)
        

        #self.bn1 = nn.BatchNorm2d(out_planes) # todo

        in_planes, out_planes, stride, padding, kernel_size = cfg[1]
        self.conv2 = QuantConv2d(in_planes, out_planes, stride=stride, padding=padding, kernel_size=kernel_size)
        
        #self.bn2 = nn.BatchNorm2d(out_planes)

        in_planes, out_planes, stride, padding, kernel_size = cfg[2]
        self.conv3 = QuantConv2d(in_planes, out_planes, stride=stride, padding=padding, kernel_size=kernel_size)
        
        #self.bn3 = nn.BatchNorm2d(out_planes)

        in_planes, out_planes, stride, padding, kernel_size = cfg[3]
        self.conv4 = QuantConv2d(in_planes, out_planes, stride=stride, padding=padding, kernel_size=kernel_size)

        in_planes, out_planes, stride, padding, kernel_size = cfg[4]
        self.conv5 = QuantConv2d(in_planes, out_planes, stride=stride, padding=padding, kernel_size=kernel_size)
        
        #self.bn5 = nn.BatchNorm2d(out_planes)

        self.fc1 = LinearQuant(128, fc[0]) # todo
        

        self.fc2 = LinearQuant(fc[0],fc[1] )
        


    def forward(self, input):

        #temp = input[0,0,:,:]
        #print(torch.max(temp))
        #print(torch.min(temp))

        input = (input-128)/128
        #temp = input[0,0,:,:]
        #print(temp*128)
        #print(torch.max(temp))
        #print(torch.min(temp))
        #print('input_conv1:,',input.shape)

        h1 = self.conv1(input).relu() # 80 * 60
        h1 = self.quanA.apply(h1)
        #print(h1[0,0,:,:]*128*128)

        #h1 = self.bn1(h1)
        #print('input_pool1:,',h1.shape)
        p1 = F.avg_pool2d(h1, 2)
        p1 = self.quanA.apply(p1)


        #print('input_conv2:,',p1.shape)
        h2 = self.conv2(p1).relu() # 40 * 30
        #h2 = self.bn2(h2)
        h2 = F.dropout(h2, p = probs, training= self.training) # todo
        h2 = self.quanA.apply(h2)

        #print('input_pool2:,',h2.shape)
        p2 = F.avg_pool2d(h2, 2) # 20 * 15
        p2 = self.quanA.apply(p2)


        #print('input_conv3:,',p2.shape)
        h3 = self.conv3(p2).relu()
        #h3 = self.bn3(h3)
        h3 = F.dropout(h3, p=probs, training=self.training)
        h3 = self.quanA.apply(h3)

        #print('input_pool3:,',h3.shape)
        p3 = F.avg_pool2d(h3, 2)
        p3 = self.quanA.apply(p3)

        #print('input_conv4:,',p3.shape)
        h4 = self.conv4(p3).relu()
        #h4 = self.bn4(h4)
        h4 = F.dropout(h4, p=probs, training=self.training)
        h4 = self.quanA.apply(h4)
        
        #print('input_conv5:,',h4.shape)
        h5 = self.conv5(h4).relu()
        h5 = self.quanA.apply(h5)
        #h5 = self.bn5(h5)
        
        #print('view_input:,',h5.shape)
        x = h5.view(-1,128) # todo

        x = F.dropout(x, p = probs, training=self.training)
        
        #print('input_fc1:,',x.shape)
        h1 = self.fc1(x).relu() # todo
        h1 = self.quanA.apply(h1)
        
        #print('input_fc2:,',h1.shape)
        h2 = self.fc2(h1) # todo
        #print(h2*128*128)
        h2 = self.quanA.apply(h2)
        #print(h2*128)
        return  h2
        
    def show_feature(self, input):

        
        features={}  
        
        features.update({'input_image':input})
        input = (input-128)/128
        features.update({'input':input})
        

        h1 = self.conv1(input).relu() # 80 * 60
        h1 = self.quanA.apply(h1)
        features.update({'conv1':h1})
        
        p1 = F.avg_pool2d(h1, 2)
        p1 = self.quanA.apply(p1)
        features.update({'avg_pool1':p1})


        
        h2 = self.conv2(p1).relu() # 40 * 30
        h2 = F.dropout(h2, p = probs, training= self.training) # todo
        h2 = self.quanA.apply(h2)
        features.update({'conv2':h2})

       
        p2 = F.avg_pool2d(h2, 2) # 20 * 15
        p2 = self.quanA.apply(p2)
        features.update({'avg_pool2':p2})


       
        h3 = self.conv3(p2).relu()
        h3 = F.dropout(h3, p=probs, training=self.training)
        h3 = self.quanA.apply(h3)
        features.update({'conv3':h3})

        
        p3 = F.avg_pool2d(h3, 2)
        p3 = self.quanA.apply(p3)
        features.update({'avg_pool3':p3})

        
        h4 = self.conv4(p3).relu()
        h4 = F.dropout(h4, p=probs, training=self.training)
        h4 = self.quanA.apply(h4)
        features.update({'conv4':h4})
        
        
        h5 = self.conv5(h4).relu()
        h5 = self.quanA.apply(h5)
        features.update({'conv5':h5})
        
        
        x = h5.view(-1,128) # todo
        features.update({'fc_input':x})

        x = F.dropout(x, p = probs, training=self.training)
        
        h1 = self.fc1(x).relu() # todo
        h1 = self.quanA.apply(h1)
        features.update({'fc1':h1})

        h2 = self.fc2(h1) # todo
        h2 = self.quanA.apply(h2)
        features.update({'fc2':h2})
        
        return  h2,features






