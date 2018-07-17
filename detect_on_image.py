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
import argparse

conf, model = get_conf_loader('detect')

yolo = Yolo(conf,model,None,None,None,init_writers = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect over image')
    parser.add_argument("-c", "--cls_conf", help="whether use class confidence", action="store_true")
    parser.add_argument('-f','--file_path', help='image url to detect', default='data/person.jpg', type=str)
    parser.add_argument('-o','--output_path', help='detection output path', default='data/output.jpg', type=str)
    parser.add_argument('-m','--model_name', help='trained model path', default='yolo_model_final.pth', type=str)
    parser.add_argument('-t','--threshold', help='prediction threshold', default=0.7, type=float)
    parser.add_argument('-l','--level', help='on which resolution level to run detection [1-7]', default=0, type=int)
    parser.add_argument('-fs','--font_size', help='displayed font size', default=15, type=int)
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
    conf.font_size = args.font_size
    detected_img = yolo.detect_on_img(conf,Image.open(args.file_path),level=args.level)
    detected_img.save(args.output_path)