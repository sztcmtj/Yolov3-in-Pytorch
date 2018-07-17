import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from datetime import datetime
import random
from torch.utils.data import DataLoader
from torchvision import transforms as trans
import struct # get_image_size
import imghdr # get_image_size
from skvideo import io
import cv2
from tqdm import tqdm

def detect_video(conf, video_file, out_path, yolo, level=0):
    """Use yolo v3 to detect video.
    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        level : on which resolution to run detection, 
        range[1 - 7], default is 416
        the resolution list is in conf.resolutions
    """
    videogen = io.vreader(video_file)
    metdata = io.ffprobe(video_file)
    frame_rate = int(int(metdata['video']['@avg_frame_rate'].split('/')[0]) / int(metdata['video']['@avg_frame_rate'].split('/')[1]))
    frame = next(videogen)
    shape = (frame.shape[1], frame.shape[0])
    video_writer = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'XVID'), frame_rate, shape)
    for frame in tqdm(videogen, total = int(metdata['video']['@nb_frames'])):
        detected_frame = np.array(yolo.detect_on_img(conf, Image.fromarray(frame), level=level))[...,::-1]
#         pdb.set_trace()
        video_writer.write(detected_frame)
    video_writer.release()
    videogen.close()
    
def coco_collate_fn(batch):
    imgs_group = []
    bboxes_group = []
    labels_group = []
    for item in batch:
        if item == None:
            continue
        else:
            imgs_group.append(item.imgs.unsqueeze(0))
            bboxes_group.append(item.bboxes)
            labels_group.append(item.labels)
    return torch.cat(imgs_group), bboxes_group, labels_group

def scaling_model(conf, new_res_idx, yolo, train_ds):
    print('switching to resolution {}*{}'.format(conf.idx_2_res[str(new_res_idx)],
                                                 conf.idx_2_res[str(new_res_idx)]))
    conf.input_size = conf.resolutions[new_res_idx]
    conf.transform_test.transforms[0] = trans.Resize([conf.input_size, conf.input_size])
    train_ds.conf = conf
    yolo.train_loader = DataLoader(
        train_ds,
        batch_size=conf.batch_sizes[new_res_idx],
        shuffle=True,
        collate_fn=coco_collate_fn,
        pin_memory=False,
        num_workers=conf.num_workers[new_res_idx])
    conf.board_loss_every = len(yolo.train_loader) // 100
    conf.evaluate_every = len(yolo.train_loader) // 10
    conf.board_pred_image_every = len(yolo.train_loader) // 2
    conf.save_every = len(yolo.train_loader) // 2
    conf.board_grad_norm = len(yolo.train_loader) // 10
    yolo.val_loader.transform = trans.Compose([
        trans.Resize([conf.input_size, conf.input_size]),
        trans.ToTensor(),
        trans.Normalize(conf.mean, conf.std)
    ])
    yolo.val_loader.batch_size = conf.batch_sizes[new_res_idx]
    yolo.model.update_input_size(conf)
    yolo.res_idx = new_res_idx

def random_scaling(conf, yolo, train_ds):
    new_res_idx = random.randint(1, len(conf.resolutions)-1)
    scaling_model(conf, new_res_idx, yolo, train_ds)
    
def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

def clip_grad_norm_log_(conf,parameters, max_norm, writer,step,norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor]): an iterable of Tensors that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    conf.running_norm = conf.running_norm*0.98 + total_norm*0.02
    if step  % conf.board_grad_norm == 0:
        writer.add_scalar('running_norm',conf.running_norm,step)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def enumerate_shifted_anchor(feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return torch.cat([torch.FloatTensor(shift_x).unsqueeze(-1), torch.FloatTensor(shift_y).unsqueeze(-1)],-1)

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def soft_nms(boxes, scores, sigma = 0.5, Nt = 0.3, threshold = 0.001, method = 2):
    """Performs soft non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (xc, yc, w, h)].
    scores: 1-D array of box scores. All elements of scores should >= 0
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    area = boxes[:,2] * boxes[:,3]
    # transfer to x1y1x2y2 format
    boxes[:,0] -= boxes[:,2]/2
    boxes[:,1] -= boxes[:,3]/2
    boxes[:,2] += boxes[:,0]
    boxes[:,3] += boxes[:,1]
    
    pick = []
    
    score_idx = np.arange(len(scores))
    
    
    while np.any(scores != -1):
        # Get indicies of boxes sorted by scores (highest first)
        max_idx = scores.argmax()
        pick.append(max_idx)
        scores[max_idx] = -1

        # Compute IoU of the picked box with the rest
        score_idx = score_idx[score_idx != max_idx]
        iou = compute_iou(boxes[max_idx], boxes[score_idx], area[max_idx], area[score_idx])

        if method == 1: # linear
            weight = np.where(iou > Nt, 1-iou, 1)
        elif method == 2: # gaussian
            weight = np.exp(-(iou)/sigma)
        else: # original NMS
            weight = np.where(iou > Nt, 0, 1)

        scores[score_idx] *= weight

        scores[score_idx[scores[score_idx] < threshold]] = -1
        score_idx = np.delete(score_idx, np.where(scores[score_idx] < threshold)[0])
    return torch.tensor(pick, dtype=torch.long) 

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (x, y, w, h)].
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    
    area = boxes[:,2] * boxes[:,3]
    boxes[:,2] += boxes[:,0]
    boxes[:,3] += boxes[:,1]
    # transfer to x1y1x2y2 format
#     # Compute box areas
#     y1 = boxes[:, 1]
#     x1 = boxes[:, 0]
#     y2 = boxes[:, 3] 
#     x2 = boxes[:, 2]
#     area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return torch.tensor(pick, dtype=torch.long)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, x2, y2]
    boxes: [boxes_count, (x1, y1, x2, y2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[3], boxes[:, 3])
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[2], boxes[:, 2])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compare_tensors(a,b):
    return torch.sum(torch.abs(a - b))
