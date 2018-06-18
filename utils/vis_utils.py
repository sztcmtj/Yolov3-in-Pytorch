import colorsys
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from utils.box_utils import *
from torchvision import transforms as trans

truefont = ImageFont.truetype('/root/Notebooks/fonts/arial.ttf',size=20)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    colors = (np.array(colors) * 255).astype(int)
    return colors

def draw_bbox_class(img,labels,bboxes,id_2_class):
    classes_list = labels.numpy().tolist()
    filtered_classes = set(classes_list)
    N = len(filtered_classes)
    colors = random_colors(N)
    class_2_color = {}
    for i,c in enumerate(filtered_classes):
        class_2_color[c] = (colors[i][0],colors[i][1],colors[i][2])
        draw = ImageDraw.Draw(img)
    bboxes_xywh = xcycwh_2_xywh(bboxes)
    for i in range(bboxes_xywh.shape[0]):
        draw.rectangle(xywh_2_x1y1x2y2(bboxes_xywh[i]),outline=class_2_color[labels[i].item()])
        draw.text((bboxes_xywh[i][0],bboxes_xywh[i][1]),text=id_2_class[labels[i].item()],font=truefont)
    return img

def de_preprocess(tensor,mean,std,cuda=False):
    std = torch.Tensor(std).view(3,1,1)
    mean = torch.Tensor(mean).view(3,1,1)
    if cuda:
        std = std.cuda()
        mean = mean.cuda()
    return tensor * std + mean

def to_img(tensor,conf):
    return trans.ToPILImage()(de_preprocess(tensor,conf.mean,conf.std))