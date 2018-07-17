from utils.config import get_conf_loader
from torchvision import transforms as trans
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from utils.vis_utils import *
from utils.box_utils import *
from utils.dataset_tools import *
from utils.utils import *
from Yolo_learner_V2 import Yolo

conf, model, train_ds, train_loader, val_dataset, val_loader, maps = get_conf_loader(mode = 'train')

yolo = Yolo(conf,model,train_loader,val_loader,None)

yolo.optimizer = optim.Adam([*model.parameters()][159:],lr=1e-4)
for para in [*yolo.model.parameters()][:159]:
    para.requires_grad = False
yolo.train(conf,3)
yolo.save_state(conf, 'ft_finish', conf.idx_2_res[str(yolo.res_idx)])

for para in [*model.parameters()][9:159]:
    para.requires_grad = True
yolo.optimizer = optim.Adam([*model.parameters()][9:],lr=1e-4)

for e in range(3):
    for idx in [1, 7, 2, 6, 3, 5, 4]:
        scaling_model(conf, idx, yolo, train_ds)
        yolo.train(conf, 3)
    yolo.save_state(conf, 'round_{}'.format(e), conf.idx_2_res[str(yolo.res_idx)])

yolo.train_loader.dataset.final_round = True
for idx in [1, 7, 2, 6, 3, 5, 4]:
    scaling_model(conf, idx, yolo, train_ds)   
    yolo.train(conf, 1)
yolo.save_state(conf, 'final_round', conf.idx_2_res[str(yolo.res_idx)])
