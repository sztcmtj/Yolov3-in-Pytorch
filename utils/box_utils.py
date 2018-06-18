import torch
import numpy as np
import pdb

def xywh_2_x1y1x2y2(bbox):
    return [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]

def xywh_2_xcycwh(bbox):
    new_bbox = bbox.clone()
    new_bbox[:,0] += new_bbox[:,2] / 2.
    new_bbox[:,1] += new_bbox[:,3] / 2.
    return new_bbox

def xcycwh_2_xywh(bbox):
    new_bbox = bbox.clone()
    new_bbox[:,0] -= new_bbox[:,2] / 2.
    new_bbox[:,1] -= new_bbox[:,3] / 2.
    return new_bbox

def adjust_bbox(img_size,input_size,bboxes):
    ratio = torch.Tensor([input_size[0]/img_size[0],input_size[1]/img_size[1]])
    ratio = ratio.repeat(bboxes.shape[0],2)
    bboxes = bboxes * ratio
    return bboxes

def horizontal_flip_boxes(bboxes,size):
    bboxes[:,0] = size[0] - bboxes[:,0] - bboxes[:,2]
    return bboxes

def cal_ious(bbox_a,bbox_b):
    bbox_a = bbox_a.cpu().numpy().copy()
    bbox_b = bbox_b.cpu().numpy().copy()
    bbox_a[:,2:] += bbox_a[:,:2]
    bbox_b[:,2:] += bbox_b[:,:2]
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    result = area_i / (area_a[:, None] + area_b - area_i)
    return torch.tensor(result,dtype=torch.float32)

def cal_iou(bbox_a,bbox_b):
    bbox_a = bbox_a.cpu().numpy().copy()
    bbox_b = bbox_b.cpu().numpy().copy()
    area_a = np.prod(bbox_a[2:],)
    area_b = np.prod(bbox_b[2:])
    bbox_a[2:] += bbox_a[:2]
    bbox_b[2:] += bbox_b[:2]
    tl = np.maximum(bbox_a[:2], bbox_b[:2])
    br = np.minimum(bbox_a[2:], bbox_b[2:])
    area_i = np.prod(br - tl) * (tl < br).all()
    result = area_i/(area_a + area_b - area_i)
    return result

def cal_iou_1on1(bbox_a,bbox_b):
    bbox_a = bbox_a.cpu().numpy().copy()
    bbox_b = bbox_b.cpu().numpy().copy()
    area_a = np.prod(bbox_a[:,2:],axis=1)
    area_b = np.prod(bbox_b[:,2:],axis=1)
    bbox_a[:,2:] += bbox_a[:,:2]
    bbox_b[:,2:] += bbox_b[:,:2]
    tl = np.maximum(bbox_a[:,:2], bbox_b[:,:2])
    br = np.minimum(bbox_a[:,2:], bbox_b[:,2:])
    area_i = np.prod(br - tl,axis=1) * (tl < br).all()
    result = area_i/(area_a + area_b - area_i)
    return torch.tensor(result,dtype=torch.float32)