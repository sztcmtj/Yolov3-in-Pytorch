import torch.nn.functional as F
from torch import nn
import torch
from utils.vis_utils import *
from utils.box_utils import *
from utils.dataset_tools import *
from utils.utils import *
import pdb

def build_targets(device,pred_bbox,bboxes_group,
                  labels_group,
                  best_anchors_idx_group,
                  anchors,
                  scale,nA,nH,nW,nC,
                  noobject_scale=0.5,
                  object_scale=5.,
                  thresh=0.5,warm_up=False):
    
    assert len(pred_bbox) == len(bboxes_group) == len(labels_group)
    nB = len(pred_bbox)
    
    confidence_weight  = torch.ones([nB, nA, nH, nW],dtype=torch.float32,device=device) * noobject_scale
    coord_mask = torch.zeros([nB, nA, nH, nW],device=device)
    cls_mask   = torch.zeros([nB, nA, nH, nW],dtype=torch.long,device=device)
    txc = torch.zeros([nB, nA, nH, nW],device=device) 
    tyc = torch.zeros([nB, nA, nH, nW],device=device) 
    tw = torch.zeros([nB, nA, nH, nW],device=device) 
    th = torch.zeros([nB, nA, nH, nW],device=device) 
    tconf = torch.zeros([nB, nA, nH, nW],device=device)
    tcls = torch.zeros([nB, nA, nH, nW, nC],device=device)
    
    for b in range(nB):
        cur_pred_boxes = pred_bbox[b].view(-1,4)
        gt_boxes = bboxes_group[b]/scale
        ious = cal_ious(xcycwh_2_xywh(cur_pred_boxes),xcycwh_2_xywh(gt_boxes))
        max_ious,_ = torch.max(ious,1)
        idx = max_ious > thresh
        idx = idx.view(pred_bbox[b].shape[:-1])
        confidence_weight[b][idx] = 0.
        """
        If the bounding box prior is not the best but does overlap a ground truth object 
        by more than some threshold we ignore the prediction, 
        following [15]. We use the threshold of .5. 
        Unlike [15] our system only assigns one bounding box prior for each ground truth object. 
        If a bounding box prior is not assigned to a ground truth object,
        it incurs no loss for coordinate or class predic- tions, only objectness.
        """
    if warm_up:
        txc.fill_(0.5)
        tyc.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)
        """
        in the training beginning, all the coordination destination goes to (0.5,0.5),
        which is the center of the square.
        I think this could help training to convert, because in the begining actually there is nothing except missed guess.
        notice the coord_mask are filled with all ones
        """
    nGT = 0
    nCorrect = 0
    
    for b in range(nB):
        
        gt_boxes = bboxes_group[b]/scale
        nGT += len(gt_boxes)
        gt_ij = gt_boxes[:,:2].to(torch.int64)
        
        muitl_labels = generate_multi_label(device,gt_ij,labels_group[b],nC)
        
        best_anchors_idx = best_anchors_idx_group[b]
        
        """
        first,just find out the right square which identified by gt_ij
        then only use the anchors and ground truth width and height to pick the right anchor,
        regardless of the center.
        Once we have the best fit anchor,we just only took this anchor seriously
        """
        filterd_pred_bbox = pred_bbox[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]]
        coord_mask[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = 1
        cls_mask[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = 1
        confidence_weight[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = object_scale
        txc[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = gt_boxes[:,0] - gt_ij[:,0].to(torch.float32)
        tyc[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = gt_boxes[:,1] - gt_ij[:,1].to(torch.float32)
        tw[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = torch.log(gt_boxes[:,2]/anchors[:,0][best_anchors_idx])
        th[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = torch.log(gt_boxes[:,3]/anchors[:,1][best_anchors_idx])
        ious = cal_iou_1on1(filterd_pred_bbox,gt_boxes)
        nCorrect += torch.sum(ious > 0.5).item()
        """
        YOLOv3 predicts an objectness score for each bounding box using logistic regression. 
        This should be 1 if the bounding box prior overlaps a ground truth object 
        by more than any other bounding box prior. 
        """
        tconf[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = 1
        
        tcls[b][best_anchors_idx,gt_ij[:,1],gt_ij[:,0]] = muitl_labels
    return nGT, nCorrect, coord_mask, confidence_weight, cls_mask, txc, tyc, tw, th, tconf, tcls    

class Yolo_loss(nn.Module):
    def __init__(self, conf,anchors,feature_size,num_classes=80):
        super(Yolo_loss, self).__init__()
        self.conf = conf
        self.num_classes = num_classes
        self.scale = conf.input_size / feature_size
        self.anchors = torch.tensor(anchors,dtype=torch.float32,device=conf.device) / self.scale
        self.num_anchors = len(anchors)
        self.coord_scale = conf.coord_scale
        self.noobject_scale = conf.noobject_scale
        self.object_scale = conf.object_scale
        self.class_scale = conf.class_scale
        self.thresh = conf.thresh
        self.nW = feature_size
        self.nH = feature_size
        self.shift_x,self.shift_y = enumerate_shifted_anchor(1,feature_size,feature_size) #generate grid
        self.shift_x,self.shift_y = self.shift_x.to(conf.device),self.shift_y.to(conf.device)
        
    def forward(self, output, bboxes_group,labels_group,best_anchors_idx_group,warm_up,debug=False):
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = self.nH
        nW = self.nW
        output = output.view(nB, nA, (5+nC), nH, nW) # [32, 5, 25, 13, 13]
        xc = F.sigmoid(output[:,:,0,:,:])
        yc = F.sigmoid(output[:,:,1,:,:])
        w = output[:,:,2,:,:]
        h = output[:,:,3,:,:]
        confidence = output[:,:,4,:,:]
        cls = output[:,:,5:,:,:]
        cls = cls.view(nB*nA,nC,nH*nW).transpose(1,2).contiguous().view(nB,nA,nH,nW,nC)
        xc_refine = (xc.clone().detach() + self.shift_x).unsqueeze(-1) 
        # refine means map x,y to each square in the 13*13 or 26*26 or 52*52 grid
        # the graph stops here to collect loss metric
        yc_refine = (yc.clone().detach() + self.shift_y).unsqueeze(-1)
        w_refine = (torch.exp(w.clone().detach()) * self.anchors[:,0].view(1,self.num_anchors,1,1)).unsqueeze(-1)
        h_refine = (torch.exp(h.clone().detach()) * self.anchors[:,1].view(1,self.num_anchors,1,1)).unsqueeze(-1)
        pred_bbox = torch.cat([xc_refine,yc_refine,w_refine,h_refine],dim=-1)
        """
        normally it's x,y,w,h format, but when we do indexing, it's first y,then x
        just notice here
        """
        nGT, nCorrect, \
        coord_mask, confidence_weight,\
        cls_mask, \
        txc, tyc, tw, th, \
        tconf, tcls = build_targets(self.conf.device,pred_bbox,bboxes_group,labels_group,best_anchors_idx_group,self.anchors,self.scale,
                                    nA,nH,nW,nC,
                                    noobject_scale = self.noobject_scale, 
                                    object_scale = self.object_scale,
                                    thresh=self.thresh,
                                    warm_up=warm_up)
        if debug:
            return confidence,cls,pred_bbox,nGT, nCorrect, \
                    coord_mask, confidence_weight,\
                    cls_mask, \
                    txc, tyc, tw, th, \
                    tconf, tcls
        loss_xc = self.coord_scale * self.conf.mse_loss(xc*coord_mask,txc*coord_mask)/2.0/nB
        loss_yc = self.coord_scale * self.conf.mse_loss(yc*coord_mask,tyc*coord_mask)/2.0/nB
        loss_w = self.coord_scale * self.conf.mse_loss(w*coord_mask,tw*coord_mask)/2.0/nB
        loss_h = self.coord_scale * self.conf.mse_loss(h*coord_mask,th*coord_mask)/2.0/nB
        loss_conf = self.conf.bce_loss(confidence_weight,size_average=False)(confidence,tconf)/nB
#         tcls_onehot = torch.zeros([nGT,nC]).scatter_(1,tcls[cls_mask==1].unsqueeze(-1),1)     
        
        loss_cls = self.class_scale * self.conf.bce_loss(size_average=False)(cls[cls_mask==1],tcls[cls_mask==1])/nB
        """
        Each box predicts the classes the bounding box may contain using multilabel classification. 
        We do not use a softmax as we have found it is unnecessary for good performance, 
        instead we simply use independent logistic classifiers. 
        During training we use binary cross-entropy loss for the class predictions. 
        This formulation helps when we move to more complex domains like the Open Images Dataset [5]. 
        In this dataset there are many overlapping labels (i.e. Woman and Person). 
        Using a softmax imposes the assumption that each box 
        has exactly one class which is often not the case. A multilabel approach better models the data.
        """
        loss = loss_xc + loss_yc + loss_w + loss_h + loss_conf + loss_cls
        return loss,nGT,nCorrect,loss_xc.item(),loss_yc.item(),loss_w.item(),loss_h.item(),loss_conf.item(),loss_cls.item()