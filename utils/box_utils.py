import torch
import numpy as np
import pdb

def xcycwh_2_x1y1x2y2(bbox):
    new_bbox = bbox.clone()
    new_bbox[:,:2] -= 0.5 * bbox[:,2:]
    new_bbox[:,2:] = 0.5 * new_bbox[:,2:] + bbox[:,:2]
    return new_bbox

def x1y1x2y2_2_xcycwh(bbox):
    new_bbox = bbox.clone()
    new_bbox[:,:2] = 0.5 * (bbox[:,:2] + bbox[:,2:])
    new_bbox[:,2:] = bbox[:,2:] - bbox[:,:2]
    return new_bbox    

def xywh_2_xcycwh(bbox):
    new_bbox = bbox.clone()
    new_bbox[:,:2] += new_bbox[:,2:] / 2.
    return new_bbox

def xywh_2_x1y1x2y2(bbox):
    if bbox.ndimension() == 1:
        return [bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]]
    else:
        new_bbox = bbox.clone()
        new_bbox[:,2:] += new_bbox[:,:2]
        return new_bbox

def xcycwh_2_xywh(bbox):
    new_bbox = bbox.clone()
    new_bbox[:,:2] -= new_bbox[:,2:] / 2.
    return new_bbox

def trim_pred_bboxes(bboxes, size):
    """
    bboxes : torch.tensor, shape = [n,4], format = xcycwh
    """
    return x1y1x2y2_2_xcycwh(torch.clamp(xcycwh_2_x1y1x2y2(bboxes), 0, size))

def cal_iou_wh(bboxes_wh,anchors):
    """
    my simplified function of computing iou between true boxes's w and h only, and anchors
    bboxes_wh : tensor,shape = [n,2]
    anchors : tensor,shape = [nA,2]
    return : iou,shape = [n,nA]
    """
    wh = bboxes_wh.unsqueeze(1)
    min_w_min_h = torch.min(wh,anchors)
    intersect_area = min_w_min_h[..., 0] * min_w_min_h[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    return iou

def cal_iou_wh_orig(bboxes_wh,anchors):
    '''
    got it from https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py
    '''
    wh = bboxes_wh[:,:2]
    wh.unsqueeze_(1)
    box_maxes = wh / 2.
    box_mins = -box_maxes
    intersect_mins = torch.max(box_mins, anchor_mins)
    intersect_maxes = torch.min(box_maxes, anchor_maxes)
    intersect_wh = torch.clamp(intersect_maxes - intersect_mins, min = 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    return iou

def adjust_bbox(img_size, input_size, bboxes, detect=False):
    ratio = torch.Tensor([input_size/img_size[0],input_size/img_size[1]])
    ratio = ratio.repeat(bboxes.shape[0],2)
    if not detect:
        bboxes = bboxes * ratio
    else:
        bboxes = bboxes / ratio
    return bboxes

def horizontal_flip_boxes(bboxes,size):
    bboxes[:,0] = size - bboxes[:,0] - bboxes[:,2]
    return bboxes

def cal_ious(bbox_a,bbox_b):
    """
    bbox : torch.tensors
    bbox : shape [n,4], 
    format : xywh
    """
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

def cal_ious_xcycwh(bbox_a_,bbox_b_):
    """
    bbox : torch.tensors
    bbox : shape [n,4], 
    format : xcycwh
    """
    bbox_a = bbox_a_.cpu().numpy().copy()
    bbox_b = bbox_b_.cpu().numpy().copy()
    area_a = np.prod(bbox_a[:, 2:], axis=1)
    area_b = np.prod(bbox_b[:, 2:], axis=1)
    
    bbox_a[:,:2] -= bbox_a[:,2:]/2
    bbox_b[:,:2] -= bbox_b[:,2:]/2
    bbox_a[:,2:] += bbox_a[:,:2] 
    bbox_b[:,2:] += bbox_b[:,:2]
    
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)    
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