from easydict import EasyDict as edict
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils.vis_utils import *
from utils.box_utils import *
# from utils.dataset_tools import *
from utils.utils import *
from models.Yolo_model import Yolo_model, build_targets, yolo_loss
from Yolo_learner_V2 import Yolo
from tqdm import tqdm_notebook as tqdm
import json
import argparse

conf = edict()

conf.coco_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                     [59, 119], [116, 90], [156, 198], [373, 326]]

conf.correct_id_2_class = json.load(open('data/correct_id_2_class.json','r'))
conf.class_num = len(conf.correct_id_2_class)
conf.log_path = Path('work_space/log')
conf.model_path = Path('work_space/model')
conf.save_path = Path('work_space/save')

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
conf.predict_confidence_threshold = 0.7
conf.pred_nms_iou_threshold = 0.3
conf.object_only_on_predict = True
conf.warm_up_img_num = 12800
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

yolo = Yolo(conf,model,None,None,None,init_writers = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect over image')
    parser.add_argument("-c", "--cls_conf", help="whether use class confidence", action="store_true")
    parser.add_argument('-f','--file_path', help='image url to detect', default='data/person.jpg', type=str)
    parser.add_argument('-o','--output_path', help='detection output path', default='data/output.jpg', type=str)
    parser.add_argument('-m','--model_name', help='trained model path', default='model_trained.pth', type=str)
    parser.add_argument('-t','--threshold', help='prediction threshold', default=0.7, type=float)
    parser.add_argument('-l','--level', help='on which resolution level to run detection [1-7]', default=0, type=int)
    args = parser.parse_args()
    yolo.model.load_state_dict(
    torch.load(
        conf.model_path /
        args.model_name
    ))
    if args.cls_conf:
        print('use cls confidence')
        conf.object_only_on_predict = False
    conf.predict_confidence_threshold = args.threshold
    detected_img = yolo.detect_on_img(conf,Image.open(args.file_path),level=args.level)
    detected_img.save(args.output_path)