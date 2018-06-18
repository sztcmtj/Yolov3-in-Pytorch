from pycocotools.coco import COCO
import torch
import random
from utils.box_utils import *
from torchvision.transforms.functional import hflip
import cv2
import math
import pdb

def get_coco_class_name_map(anno):
    coco=COCO(anno)
    cats = coco.loadCats(coco.getCatIds())
    class_2_id = {}
    id_2_class = {}
    for pair in cats:
        id_2_class[pair['id']] = pair['name']
        class_2_id[pair['name']] = pair['id']
    return class_2_id,id_2_class

def synthesize_bbox_id(targets,id_map):
    labels = []
    bboxes = []
    for target in targets:
        label = id_map[target['category_id']]
        bbox = torch.Tensor(target['bbox']).view(1,4)
        # get rid of targets that are too small
        # try number 17975 data, there is even a wrong data with h = 0,took me so long to debug
        if bbox[:,2:].min().item() < 8.:
            continue
        labels.append(label)
        bboxes.append(bbox)
    return labels,bboxes

class Coco_loader():
    def __init__(self,config,dataset,transform,batch_size=8,hflip=True,shuffle=True):
        self.len = len(dataset)//batch_size
        self.total_num = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.current = 0
        self.transform = transform
        self.config = config
        self.shuffle = shuffle
        
    def __len__(self):
        return self.len
    def __iter__(self):
        return self
    def __next__(self):
        if self.current >= self.total_num:
            raise StopIteration
       
        if self.current < self.total_num - self.batch_size:
            iter_num = self.batch_size
        else:
            iter_num = self.total_num - self.current
        imgs = []
        labels_group = []
        bboxes_group = []
            
        for i in range(iter_num):
            if self.shuffle:
                choice = random.choice(range(len(self.dataset)))
            else:
                choice = self.current

            img,targets = self.dataset[choice]
            if targets == []:
                self.current += 1
                continue
            labels,bboxes = synthesize_bbox_id(targets,self.dataset.maps[0])
            if len(bboxes) == 0:
                self.current += 1
                continue
            labels = torch.LongTensor(labels)
            bboxes = torch.cat(bboxes)
            bboxes = adjust_bbox(img.size,self.config.input_size,bboxes)
            if hflip:
                if random.random() > 0.5:
                    img = hflip(img)
                    bboxes = horizontal_flip_boxes(bboxes,self.config.input_size)
            bboxes = xywh_2_xcycwh(bboxes)
            img = self.transform(img)
            imgs.append(img.unsqueeze(0))
            labels_group.append(labels)
            bboxes_group.append(bboxes)
            self.current += 1
        return torch.cat(imgs),labels_group,bboxes_group
            
def arrange_bbox_label(device,coco_anchors,labels_group,bboxes_group):
    """
    根据ground truth的w和h,通过和预设anchors的比对找出最匹配的anchor
    根据匹配上的anchor划分成大、中、小三组，
    每一个尺度返回3个参数，labels_group，bboxes_group，best_anchors_idx
    best_anchors_idx属于[0,1,2],代表每一组里的3个anchors选了哪个
    注意如果某个尺度没有匹配上，该尺度对应的参数返回的是空集
    """
    labels_group_small = []
    labels_group_medium = []
    labels_group_large = []
    bboxes_group_small = []
    bboxes_group_medium = []
    bboxes_group_large = []
    best_anchors_idx_group_small = []
    best_anchors_idx_group_medium = []
    best_anchors_idx_group_large = []
    coco_anchors = torch.tensor(coco_anchors,dtype=torch.float32,device=device)
    anchor_generated_boxes = torch.cat([torch.zeros_like(coco_anchors),coco_anchors],dim=1)
    for labels,bboxes in zip(labels_group,bboxes_group):
        assert len(labels) == len(bboxes),'labels and bboxes numbers not match'
        boxes_wh_only = bboxes.clone()
        boxes_wh_only[:,:2] = 0.
        best_anchors_iou,best_anchors_idx = torch.max(cal_ious(anchor_generated_boxes,boxes_wh_only).to(device),dim=0)
        labels_small = []
        labels_medium = []
        labels_large = []
        bboxes_small = []
        bboxes_medium = []
        bboxes_large = []
        best_anchors_idx_small = []
        best_anchors_idx_medium = []
        best_anchors_idx_large = []
        for i in range(len(best_anchors_idx)):
            if best_anchors_idx[i].item() in range(0,3):
                labels_small.append(labels[i].item())
                bboxes_small.append(bboxes[i].view(1,-1))
                best_anchors_idx_small.append(best_anchors_idx[i].item())
            elif best_anchors_idx[i].item() in range(3,6):
                labels_medium.append(labels[i].item())
                bboxes_medium.append(bboxes[i].view(1,-1))
                best_anchors_idx_medium.append(best_anchors_idx[i].item()-3)
            else:
                labels_large.append(labels[i].item())
                bboxes_large.append(bboxes[i].view(1,-1))
                best_anchors_idx_large.append(best_anchors_idx[i].item()-6)
        labels_group_small.append(labels_small)
        labels_group_medium.append(labels_medium)
        labels_group_large.append(labels_large)
        bboxes_group_small.append(bboxes_small)
        bboxes_group_medium.append(bboxes_medium)
        bboxes_group_large.append(bboxes_large)
        best_anchors_idx_group_small.append(best_anchors_idx_small)
        best_anchors_idx_group_medium.append(best_anchors_idx_medium)
        best_anchors_idx_group_large.append(best_anchors_idx_large)
    return labels_group_small,labels_group_medium,labels_group_large,\
            bboxes_group_small,bboxes_group_medium,bboxes_group_large,\
            best_anchors_idx_group_small,best_anchors_idx_group_medium,best_anchors_idx_group_large

def prepare_loss_input(device,feature,labels_group,bboxes_group,best_anchors_idx_group):
    input_idx = []
    final_labels = []
    final_bboxes = []
    final_anchors_idx = []
    """
    并不是每一张图片都会在大、中、小三个尺度匹配上对应的GT object。
    如果某个尺度上没有匹配上，就没必要送进下一层去计算loss
    所以这里对一个batch里面的图片进行刷选，生成最终的loss输入
    """
    for i,(labels,bboxes,idx) in enumerate(zip(labels_group,bboxes_group,best_anchors_idx_group)):
        assert len(labels) == len(bboxes) == len(idx),'labels and bboxes and anchors idx number not match'
        if len(labels) != 0:
            input_idx.append(i)
            final_labels.append(torch.tensor(labels,dtype=torch.long,device=device))
            final_bboxes.append(torch.cat(bboxes).to(device))
            final_anchors_idx.append(torch.tensor(idx,dtype=torch.long,device=device))
    final_feature = feature[input_idx]
    return final_feature,final_bboxes,final_labels,final_anchors_idx

def generate_multi_label(device,gt_ij,labels,nC):
    """mainly for avoiding cases where 
    there are more than 1 object in one grid"""
    multi_labels = torch.zeros([len(gt_ij),nC],device=device)

    for i in range(len(gt_ij)):
        multi_labels[i][labels[i]] = 1
        for j in range(i+1,len(gt_ij)):
            if torch.sum(torch.abs(gt_ij[i] - gt_ij[j])).item() == 0:
                multi_labels[i][labels[j]] = 1
                multi_labels[j][labels[i]] = 1
    
    return multi_labels