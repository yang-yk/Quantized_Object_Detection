import torch
import numpy
def class_decision(data):
    y1, y2 = data[:,0], data[:,4]
    f =  (y1<1)*(y2<1) + (1<=y1) * (1<=y2) * (y1<80) * (y2<80) + (80<=y1) * (80<=y2) * (y1< 160) * (y2< 160) + \
         (160<=y1) * (160<=y2)
    acc = f.sum().float()/f.size()[0]
    print('classification acc : ',acc)
    return acc

## obtain iou boxing
def box_iou(labels, outputs, types = 'mean'):
    output_buf, label_buf = box_scale(outputs), box_scale(labels)
    box1, box2 = get_box(output_buf), get_box(label_buf)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:, 2], box2[:,3]
    # (x1, y1) bottom left ; (x2,y2)  top right
    # print(box1,box2)

    nums = b1_x1.size()[0]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # intersection area
    inter_area = torch.clamp( (inter_rect_x2 - inter_rect_x1) + 1 , min = 0) \
                 * torch.clamp( (inter_rect_y2 - inter_rect_y1) + 1 , min = 0)


    # union area
    b1_area = (   (b1_x2 - b1_x1) + 1) * ( (b1_y2 - b1_y1) + 1)
    b2_area = (   (b2_x2 - b2_x1) + 1) * (  (b2_y2 - b2_y1) + 1)

    iou = torch.zeros(nums,1)
    for i in range(nums):
        if b1_area[i] + b2_area[i] - inter_area[i] > 1e-15:
            iou[i] = inter_area[i] / (b1_area[i] + b2_area[i] - inter_area[i] )
        else:
            iou[i] = 0

    if types == 'mean':
        iou = iou.mean()
    return iou.detach_()



## obtain iou boxing
def coincide_iou(labels, outputs, types = 'mean'): #

    # in this function ,we calculate the coincide area between the label and outputs.
    # note that the order cannot change ,and box2 use get_box_enlarger function which has different defintion.
    output_buf, label_buf = box_scale(outputs), box_scale(labels)
    box1, box2 = get_box(output_buf), get_box_enlarge(label_buf)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:, 2], box2[:,3]
    # (x1, y1) bottom left ; (x2,y2)  top right
    # print(box1,box2)

    nums = b1_x1.size()[0]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # intersection area
    inter_area = torch.clamp( (inter_rect_x2 - inter_rect_x1) + 1 , min = 0) \
                 * torch.clamp( (inter_rect_y2 - inter_rect_y1) + 1 , min = 0)


    # union area
    b1_area = (   (b1_x2 - b1_x1) + 1) * ( (b1_y2 - b1_y1) + 1)
    b2_area = (   (b2_x2 - b2_x1) + 1) * (  (b2_y2 - b2_y1) + 1)

    iou = torch.zeros(nums,1)
    for i in range(nums):
        if b1_area[i] + b2_area[i] - inter_area[i] > 1e-15:
            iou[i] = inter_area[i] /b1_area[i]
        else:
            iou[i] = 0

    if types == 'mean':
        iou = iou.mean()
    return iou.detach(),





def get_box(cor_index):
    box = torch.zeros(cor_index.size())
    box[:,0] = cor_index[:,0] - cor_index[:,2]/2
    box[:,1] = cor_index[:,1] - cor_index[:,3]/2

    box[:,2] = cor_index[:,0] + cor_index[:,2]/2
    box[:,3] = cor_index[:,1] + cor_index[:,3]/2

    return box


def get_box_enlarge(cor_index):
    box = torch.zeros(cor_index.size())
    box[:,0] = cor_index[:,0] - cor_index[:,2]
    box[:,1] = cor_index[:,1] - cor_index[:,3]

    box[:,2] = cor_index[:,0] + cor_index[:,2]/2 * 1.3
    box[:,3] = cor_index[:,1] + cor_index[:,3]/2 * 1.3

    return box


def box_scale(cor_index):
    box_label = torch.zeros(cor_index.size())
    cor_index[cor_index < 0] = 0

    if box_label.max() < 2:
        box_label[:, 0:2] = cor_index[:, 0:2] * 128
        box_label[:, 2:4] = cor_index[:, 2:4] * 128

    else:
        box_label[:, 0:2] = cor_index[:, 0:2]
        box_label[:, 2:4] = cor_index[:, 2:4]
    return box_label

