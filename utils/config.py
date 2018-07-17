from easydict import EasyDict as edict
from torchvision import transforms as trans
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from utils.vis_utils import *
from utils.box_utils import *
from utils.dataset_tools import *
from utils.utils import *
from torch.utils.data import DataLoader
from models.Yolo_model import Yolo_model, build_targets, yolo_loss
import json

def get_conf_loader(mode = 'train'):
    conf = edict()
    conf.coco_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                        [59, 119], [116, 90], [156, 198], [373, 326]]
    conf.train_path = Path('data/coco2017/train2017/')
    conf.train_anno_path = Path(
        'data/coco2017/annotations/instances_train2017.json')
    conf.val_path = Path('data/coco2017/val2017/')
    conf.val_anno_path = Path(
        'data/coco2017/annotations/instances_val2017.json')
    conf.log_path = Path('work_space/log/')
    conf.model_path = Path('work_space/model')
    conf.save_path = Path('work_space/save')
    conf.ids_path = 'data/ids.npy'
    conf.correct_id_2_class = json.load(open('data/correct_id_2_class.json','r'))
    conf.class_num = len(conf.correct_id_2_class)

    conf.font_size = 12
    conf.num_anchors = 3
    conf.resolutions = [416,224,288,352,416,480,544,608]
    conf.batch_sizes = [20,40,28,21,16,11,8,6]
    conf.res_2_idx = edict({'ft':0, '224':1, '288':2, '352':3, '416':4, '480':5, '544':6, '608':7})
    conf.idx_2_res = edict()
    for k,v in conf.res_2_idx.items():
        conf.idx_2_res[str(v)] = k
    conf.input_size = 416
    conf.scales = [32, 16, 8]
    conf.running_norm = 0.
    # conf.gdclip = 3000.
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.num_workers = [4,4,4,4,4,3,3,2]
    conf.batch_size = 16
    conf.gdclip = None
    conf.coord_scale_xy = 2.
    conf.coord_scale_wh = 20
    conf.noobject_scale = 0.5
    conf.object_scale = 5
    conf.class_scale = 2.
    conf.ignore_thresh = 0.5
    conf.evaluate_iou_threshold = 0.5
    conf.predict_confidence_threshold = 0.5
    conf.pred_nms_iou_threshold = 0.3
    conf.object_only_on_predict = False
    conf.warm_up_img_num = 12800
    conf.use_softnms = False
    conf.softnms_sigma = 0.2
    conf.softnms_Nt = 0.3
    conf.softnms_threshold = 0.1
    conf.softnms_method = 2 # 1:linear, 2:gaussian, else: hard NMS
    conf.mse_loss = nn.MSELoss(size_average=False)
    conf.bce_loss = nn.BCEWithLogitsLoss(size_average=False)

    model = Yolo_model(conf)
    model.to(conf.device)
    conf.mean = model.res50_pyramid.model.mean
    conf.std = model.res50_pyramid.model.std
    
    conf.transform_test = trans.Compose([
            trans.Resize([conf.input_size, conf.input_size]),
            trans.ToTensor(),
            trans.Normalize(conf.mean, conf.std)
        ])
    
    if mode in ['eval','train']:
        maps,_ = get_id_maps(conf)
        val_dataset = datasets.CocoDetection(conf.val_path, conf.val_anno_path)
        val_dataset.maps = maps
        
        val_loader = Coco_loader(
            conf,
            val_dataset,
            conf.transform_test,
            batch_size=conf.batch_size,
            hflip=False,
            shuffle=False)
        
        if mode == 'train':
            train_ds = Coco_dataset(conf, conf.train_path, conf.train_anno_path, maps)
            train_loader = DataLoader(
                train_ds,
                batch_size=conf.batch_size,
                shuffle=True,
                collate_fn=coco_collate_fn,
                pin_memory=False,
                num_workers=conf.num_workers[0])

            conf.eva_seen = 1000
            conf.board_loss_every = len(train_loader) // 100
            conf.evaluate_every = len(train_loader) // 10
            conf.board_pred_image_every = len(train_loader) // 2
            conf.save_every = len(train_loader) // 2
            conf.board_grad_norm = len(train_loader) // 10

            return conf, model, train_ds, train_loader, val_dataset, val_loader, maps

        else:
            return conf, model, val_dataset, val_loader, maps
    
    else:
        return conf, model    