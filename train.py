from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torchvision import transforms as trans
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
# import torch.nn.functional as F
from utils.vis_utils import *
from utils.box_utils import *
from utils.dataset_tools import *
from utils.utils import *
from tensorboardX import SummaryWriter
from tqdm import tqdm_notebook as tqdm
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader
from models.Yolo_model import Yolo_model, build_targets, yolo_loss
from Yolo_learner_V2 import Yolo
import json

conf = edict()

conf.coco_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                     [59, 119], [116, 90], [156, 198], [373, 326]]
conf.train_path = Path('data/coco2017/train2017/')
conf.train_anno_path = Path(
    'data/coco2017/annotations/instances_train2017.json')
conf.val_path = Path('data/coco2017/val2017/')
conf.val_anno_path = Path(
    'data/coco2017/annotations/instances_val2017.json')
conf.log_path = Path('work_space/log')
conf.model_path = Path('work_space/model')
conf.save_path = Path('work_space/save')
conf.ids_path = 'data/ids.npy'

maps,_ = get_id_maps(conf)
conf.correct_id_2_class = json.load(open('data/correct_id_2_class.json','r'))
conf.class_num = len(conf.correct_id_2_class)

conf.num_anchors = 3
conf.resolutions = [416,224,288,352,416,480,544,608]
conf.batch_sizes = [16,42,27,19,16,11,8,5]
conf.res_2_idx = edict({'ft':0, '224':1, '288':2, '352':3, '416':4, '480':5, '544':6, '608':7})
conf.idx_2_res = edict()
for k,v in conf.res_2_idx.items():
    conf.idx_2_res[str(v)] = k
conf.input_size = 416
conf.scales = [32, 16, 8]

conf.running_norm = 0.
# conf.gdclip = 3000.
conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conf.num_workers = [2,4,4,4,2,2,2,2]
conf.batch_size = 16
conf.gdclip = None
conf.coord_scale_xy = 2.
conf.coord_scale_wh = 20
conf.noobject_scale = 0.5
conf.object_scale = 5
conf.class_scale = 5.
conf.ignore_thresh = 0.5
conf.evaluate_iou_threshold = 0.5
conf.predict_confidence_threshold = 0.5
conf.pred_nms_iou_threshold = 0.4
conf.object_only_on_predict = False
conf.warm_up_img_num = 12800

model = Yolo_model(conf)
model.to(conf.device)
conf.mean = model.res50_pyramid.model.mean
conf.std = model.res50_pyramid.model.std

conf.mse_loss = nn.MSELoss(size_average=False)
conf.bce_loss = nn.BCEWithLogitsLoss(size_average=False)

train_ds = Coco_dataset(conf, conf.train_path, conf.train_anno_path, maps)
train_loader = DataLoader(
    train_ds,
    batch_size=conf.batch_size,
    shuffle=True,
    collate_fn=coco_collate_fn,
    pin_memory=False,
    num_workers=conf.num_workers[0])
conf.eva_batches = 100
conf.board_loss_every = len(train_loader) // 100
conf.evaluate_every = len(train_loader) // 10
conf.board_pred_image_every = len(train_loader) // 2
conf.save_every = len(train_loader) // 2
conf.board_grad_norm = len(train_loader) // 10
val_dataset = datasets.CocoDetection(conf.val_path, conf.val_anno_path)
val_dataset.maps = maps
conf.transform_test = trans.Compose([
    trans.Resize([conf.input_size, conf.input_size]),
    trans.ToTensor(),
    trans.Normalize(conf.mean, conf.std)
])
val_loader = Coco_loader(
    conf,
    val_dataset,
    conf.transform_test,
    batch_size=conf.batch_size,
    hflip=False,
    shuffle=False)

paras_ft = [*model.parameters()][159:]
optimizer_ft = optim.SGD(paras_ft,lr=1e-5,momentum=0.9,weight_decay=1e-4)
yolo = Yolo(conf,model,train_loader,val_loader,optimizer_ft)
yolo.train(conf,3)

paras = model.parameters()
yolo.optimizer = optim.SGD(paras,lr=1e-5,momentum=0.9,weight_decay=1e-4)
for idx in [1, 7, 2, 6, 3, 5, 4]:
    scaling_model(conf, idx, yolo, train_ds)
    yolo.train(conf, 3)

for e in range(60):
    if e % 20 ==0 and e != 0:        
        for param_group in yolo.optimizer.param_groups:
            param_group['lr'] /= 10.
        print('learning rate scaled to {}'.format(yolo.optimizer.param_groups[0]['lr']))
    random_scaling(conf, yolo, train_ds)
    yolo.train(conf, 1)

yolo.save_state(conf, 'final', conf.idx_2_res[str(yolo.res_idx)])