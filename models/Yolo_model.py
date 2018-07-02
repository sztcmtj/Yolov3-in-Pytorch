import random
from pretrainedmodels import models
import torch
from torch.nn import Sequential, MaxPool2d, Conv2d, LeakyReLU, BatchNorm2d, Upsample, Module, ModuleList, BCEWithLogitsLoss, CrossEntropyLoss
from utils.utils import enumerate_shifted_anchor
from utils.box_utils import *
from collections import namedtuple
from torch.nn import functional as F

import pdb

class res50_pyramid(Module):
    def __init__(self):
        super(res50_pyramid, self).__init__()
        self.model = models.resnet50()
        del self.model.avgpool
        del self.model.fc
        del self.model.last_linear

    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        p3 = x
        x = self.model.layer3(x)
        p2 = x
        p1 = self.model.layer4(x)
        return p1,p2,p3

def Basic_block(c_in,c_out):
    return Sequential(
        Conv2d(c_in, c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True),
        LeakyReLU(0.1))

def Squeeze_block(c_in,c_out):
    return Sequential(
            Conv2d(c_in,c_out, kernel_size=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True),
            LeakyReLU(0.1))

def make_pyramids_classifier(c_in,c_out,num_classes):
    classifier_num = 3*(num_classes+5)
    Pyramid_net = Sequential(
        Squeeze_block(c_in,c_out),
        Basic_block(c_out,c_out*2),
        Squeeze_block(c_out*2,c_out),
        Basic_block(c_out,c_out*2),
        Squeeze_block(c_out*2,c_out)
        )
    Classifier_net = Sequential(
        Basic_block(c_out,c_out*2),
        Squeeze_block(c_out*2,classifier_num)
        )
    return Pyramid_net,Classifier_net

class Yolo_model(Module):
    def __init__(self,conf):
        super(Yolo_model, self).__init__()
        self.num_class = conf.class_num
        self.res50_pyramid = res50_pyramid()
        self.head = Yolo_head(conf)
        self.pyramid1,self.classifier1 = make_pyramids_classifier(2048,1024,self.num_class)
        self.squeeze1 = Squeeze_block(1024,512)
        self.pyramid2,self.classifier2 = make_pyramids_classifier(1536,512,self.num_class)
        self.squeeze2 = Squeeze_block(512,256)
        self.pyramid3,self.classifier3 = make_pyramids_classifier(768,256,self.num_class)
        self.upsample = Upsample(scale_factor=2, mode='bilinear',align_corners=False)
    
    def forward(self,x):
        p1,p2,p3 = self.res50_pyramid(x)
        x = self.pyramid1(p1)
        y1 = self.classifier1(x)
        x = self.squeeze1(x)
        x = self.upsample(x)
        x = torch.cat((x,p2),dim=1)
        x = self.pyramid2(x)
        y2 = self.classifier2(x)
        x = self.squeeze2(x)
        x = self.upsample(x)
        x = torch.cat((x,p3),dim=1)
        x = self.pyramid3(x)
        y3 = self.classifier3(x)
        return self.head([y1,y2,y3])
    
    def update_input_size(self,conf):
        self.head.feature_sizes = [int(conf.input_size/scale) for scale in conf.scales]
        self.head.shiftxy_group = [enumerate_shifted_anchor(1,feature_size,feature_size).to(conf.device) \
                              for feature_size in self.head.feature_sizes]

class Yolo_head(Module):
    def __init__(self,conf):
        super(Yolo_head, self).__init__()
        anchors_tensor = torch.tensor(conf.coco_anchors)
        self.anchors_group = [anchors_tensor[6:,:].to(conf.device,dtype=torch.float),
                              anchors_tensor[3:6,:].to(conf.device,dtype=torch.float),
                              anchors_tensor[:3,:].to(conf.device,dtype=torch.float)]
        self.feature_sizes = [int(conf.input_size/scale) for scale in conf.scales]
        self.shiftxy_group = [enumerate_shifted_anchor(1,feature_size,feature_size).to(conf.device) \
                              for feature_size in self.feature_sizes]
        self.loss_inputs = namedtuple('loss_inputs',['loss_feats','pred_bboxes_group'])
        self.prediction = namedtuple('prediction',['pred_bboxes_group','confi_group','cls_pred_group'])
        self.conf = conf

    def forward(self, feats):
        pred_bboxes_group = []
        loss_feats = []

        for feat, anchors, feature_size, shift_xy, scale \
        in zip(feats,self.anchors_group, self.feature_sizes,self.shiftxy_group,self.conf.scales):
            nB = feat.shape[0]
            feat = feat.view(nB,self.conf.num_anchors,5+self.conf.class_num,feature_size,feature_size)
            feat = feat.view(nB*self.conf.num_anchors,5+self.conf.class_num,feature_size*feature_size)
            feat = feat.transpose(1,2).contiguous()
            feat = feat.view(nB,self.conf.num_anchors,feature_size,feature_size,5+self.conf.class_num)
            
            loss_feats.append(feat)
            feat_detach = feat.detach().clone()
                
            xcyc = (F.sigmoid(feat_detach[:,:,:,:,:2])+ shift_xy.unsqueeze(0).unsqueeze(0)) * scale
            wh = torch.exp(feat_detach[:,:,:,:,2:4]) * (anchors).unsqueeze(1).unsqueeze(1).unsqueeze(0)
            pred_bboxes = torch.cat([xcyc,wh],dim=-1)
            pred_bboxes_group.append(pred_bboxes)
        output = self.loss_inputs(loss_feats,pred_bboxes_group)
        return output
    
def build_targets(conf,pred_bboxes_group,bboxes_group,labels_group,anchors_group,warm_up = False):
    assert len(pred_bboxes_group) == len(anchors_group), 'anchor layers mismatch !'
    assert len(pred_bboxes_group[random.randint(0,2)]) == len(bboxes_group) == len(labels_group),\
    'batch_size mismatch !'
    nA = len(pred_bboxes_group)
    nB = len(bboxes_group)
    anchors_concat = torch.cat(anchors_group).unsqueeze(0)
    anchor_maxes = anchors_concat / 2.
    anchor_mins = -anchor_maxes
    targets = [torch.zeros([nB,nA,nF,nF,5],device=conf.device)\
               for nF in [bboxes.shape[2] for bboxes in pred_bboxes_group]]
    gt_mask = [torch.zeros([nB,nA,nF,nF],device=conf.device)\
               for nF in [bboxes.shape[2] for bboxes in pred_bboxes_group]]
    coord_mask = [torch.zeros([nB,nA,nF,nF],device=conf.device)\
               for nF in [bboxes.shape[2] for bboxes in pred_bboxes_group]]
    conf_weight = [conf.noobject_scale * torch.ones([nB,nA,nF,nF],device=conf.device)\
               for nF in [bboxes.shape[2] for bboxes in pred_bboxes_group]]
    
    if warm_up:
        for l in range(3):
            targets[l][...,:2] = 0.5
            targets[l][...,2:4] = 0.
            coord_mask[l].fill_(1)
    
    for b in range(nB):
        bboxes_xy = bboxes_group[b][:,:2]
        bboxes_wh = bboxes_group[b][:,2:]
        iou = cal_iou_wh(bboxes_group[b][:,2:],anchors_concat)
        best_anchor = torch.argmax(iou, dim=-1)
        for l, scale in enumerate([32., 16., 8.]):
            cur_pred_boxes = pred_bboxes_group[l][b].view(-1,4)
            gt_boxes = bboxes_group[b]
            ious = cal_ious_xcycwh(cur_pred_boxes,gt_boxes)
            max_ious,_ = torch.max(ious,1)
            idx = max_ious > conf.ignore_thresh
            idx = idx.view(pred_bboxes_group[l][b].shape[:-1])
            conf_weight[l][b][idx] = 0.
        
        for na,best_anchor_idx in enumerate(best_anchor):
            for l,(scale, anchors_idx_per_layer) in \
            enumerate([(32, [0,1,2]),(16, [3,4,5]),(8, [6,7,8])]):
                if best_anchor_idx in anchors_idx_per_layer:
                    anchor_matched = anchors_concat[0,best_anchor_idx]
                    best_anchor_idx_inGroup = best_anchor_idx - anchors_idx_per_layer[0]
                    bbox_xy = bboxes_xy[na]
                    bbox_wh = bboxes_wh[na]
                    gt_ij = (bbox_xy/scale).to(torch.long)
                    true_xy = bbox_xy/scale - gt_ij.to(torch.float)
                    
                    true_w = torch.log(bbox_wh[0]/anchor_matched[0])
                    true_h = torch.log(bbox_wh[1]/anchor_matched[1])

                    targets[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0],0] = true_xy[0]
                    targets[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0],1] = true_xy[1]
                    targets[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0],2] = true_w
                    targets[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0],3] = true_h
                    gt_mask[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0]] = 1
#                     targets[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0],4] = 1
                    conf_weight[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0]] = conf.object_scale

                    targets[l][b,best_anchor_idx_inGroup,gt_ij[1],gt_ij[0],4] = labels_group[b][na]
    
    if not warm_up : coord_mask =[mask.clone() for mask in gt_mask]
    
    return targets,gt_mask,conf_weight,coord_mask    

def yolo_loss(conf,loss_feats, targets, gt_mask, conf_weight, coord_mask):
    loss_xy = 0
    loss_wh = 0
    loss_conf = 0
    loss_cls = 0
    losses = namedtuple('losses',['loss_total','loss_xy','loss_wh','loss_conf','loss_cls'])
    for l in range(3):
        nB = len(loss_feats[l])
        if not torch.sum(coord_mask[l] == 1).item() == 0: 
            loss_xy += conf.coord_scale_xy * conf.bce_loss(loss_feats[l][coord_mask[l] == 1][..., :2],
                                     targets[l][coord_mask[l] == 1][..., :2]) / nB
            loss_wh += conf.coord_scale_wh * conf.mse_loss(loss_feats[l][coord_mask[l] == 1][..., 2:4],
                                     targets[l][coord_mask[l] == 1][..., 2:4]) / (2 * nB)
        if not torch.sum(gt_mask[l] == 1).item() == 0: 
            loss_conf += BCEWithLogitsLoss(
                weight=conf_weight[l], size_average=False)(loss_feats[l][..., 4],
                                                           gt_mask[l]) / nB
            loss_cls += conf.class_scale * CrossEntropyLoss(size_average=False)(
                loss_feats[l][..., 5:][gt_mask[l] == 1],
                targets[l][..., 4][gt_mask[l] == 1].to(torch.long)) / nB
    loss_total = loss_xy + loss_wh + loss_conf + loss_cls    
    return losses(loss_total,loss_xy.item(),loss_wh.item(),loss_conf.item(),loss_cls.item())