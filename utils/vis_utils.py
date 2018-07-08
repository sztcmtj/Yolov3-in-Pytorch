import colorsys
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from utils.box_utils import *
from torchvision import transforms as trans

truefont = ImageFont.truetype('data/fonts/arial.ttf',size=12)

def get_class_colors(conf):
    colors = random_colors(conf.class_num)
    class_2_color = {}
    for i,c in enumerate([*conf.correct_id_2_class.keys()]):
        class_2_color[c] = (colors[i][0], colors[i][1], colors[i][2])
    return class_2_color

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

def draw_bbox_class(img, labels, bboxes, id_2_class, class_2_color):
    """
    img : PIL Image
    """
    x_max, y_max = img.size
#     classes_list = labels.numpy().tolist()
#     filtered_classes = set(classes_list)
#     N = len(filtered_classes)
#     colors = random_colors(N)
#     class_2_color = {}
#     for i,c in enumerate(filtered_classes):
#         class_2_color[c] = (colors[i][0],colors[i][1],colors[i][2])
    draw = ImageDraw.Draw(img)
    bboxes_xywh = xcycwh_2_xywh(bboxes)
    for i in range(bboxes_xywh.shape[0]):
        x1y1x2y2 = xywh_2_x1y1x2y2(bboxes_xywh[i])
        x_corner,y1 = x1y1x2y2[0].item(),x1y1x2y2[1].item()
        
        text=id_2_class[str(labels[i].item())]
        text_w, text_h = draw.textsize(text,font=truefont)
        
        if y1 < text_h:
            y_corner = 0
        else:
            y_corner = y1 - text_h
            
        draw.rectangle([(x_corner,y_corner),(x_corner + text_w, y_corner + text_h)],fill = class_2_color[str(labels[i].item())])
        draw.rectangle(x1y1x2y2,outline=class_2_color[str(labels[i].item())])
        draw.text((x_corner,y_corner),text=text, fill='black', font=truefont)
    return img

def show_util(conf,idx,imgs, labels_group, bboxes_group, correct_id_2_class, class_2_color):
    return draw_bbox_class(
        trans.ToPILImage()(de_preprocess(conf, imgs[idx].cpu())),
        labels_group[idx].cpu(), bboxes_group[idx].cpu(), correct_id_2_class, class_2_color)

def de_preprocess(conf,tensor,cuda=False):
    std = torch.Tensor(conf.std).view(3,1,1)
    mean = torch.Tensor(conf.mean).view(3,1,1)
    if cuda:
        std = std.cuda()
        mean = mean.cuda()
    return tensor * std + mean

def to_img(tensor,conf):
    return trans.ToPILImage()(de_preprocess(tensor,conf.mean,conf.std))