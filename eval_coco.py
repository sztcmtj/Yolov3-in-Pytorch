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

conf, model, val_dataset, val_loader, maps = get_conf_loader('eval')

yolo = Yolo(conf,model,None,val_loader,None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect over image')
    parser.add_argument('-m','--model_name', help='trained model path', default='yolo_model_final.pth', type=str)
    parser.add_argument('-l','--level', help='on which resolution level to run detection [1-7]', default=0, type=int)
    args = parser.parse_args()

    yolo.model.load_state_dict(
        torch.load(
            conf.model_path /
            args.model_name))

    cocoeval = yolo.eva_coco(conf, len(val_dataset), args.level)