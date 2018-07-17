from torchvision import datasets
from torchvision import transforms as trans
import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.vis_utils import *
from utils.box_utils import *
# from utils.dataset_tools import *
from utils.utils import *
from models.Yolo_model import Yolo_model, build_targets, yolo_loss
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from collections import namedtuple
from tensorboardX import SummaryWriter
import time
from matplotlib import pyplot as plt
from pycocotools.cocoeval import COCOeval
import json

def calc_preds(conf, preds, return_confidence=False, training = True):
    '''
    calculate the prediction(bboxes and labels) from the output preds
    inputs : 
    preds : namedtuple, output of model forward
    return :
    bboxes_group, labels_group : list, each img's predicted bboxes and labels
    '''
    bboxes_group = []
    labels_group = []
    if return_confidence: 
        confidences_group = []
        cls_conf_group = []
    nB = len(preds.loss_feats[0])
    with torch.no_grad():
        for nb in range(nB):
            bboxes_predicted = []
            cls_predicted = []
            conf_predicted = []
            cls_conf_predicted = []
            for l in range(3):
                confidences = F.sigmoid(preds.loss_feats[l][nb][..., 4])
                cls_conf_preds, classes = torch.max(
                    F.softmax(preds.loss_feats[l][nb][..., 5:],dim=-1), dim=-1)
                bboxes = preds.pred_bboxes_group[l][nb]
                if conf.object_only_on_predict:
                    final_conf = confidences
                else:
                    final_conf = confidences * cls_conf_preds
#                 if not training:
#                     final_conf[cls_conf_preds < 0.8] = 0
                    
                predicted_mask = final_conf > conf.predict_confidence_threshold
                    
                bboxes_predicted.append(bboxes[predicted_mask])
                cls_predicted.append(classes[predicted_mask])
                conf_predicted.append(confidences[predicted_mask])
                cls_conf_predicted.append(cls_conf_preds[predicted_mask])
            bboxes_predicted = torch.cat(bboxes_predicted)
            cls_predicted = torch.cat(cls_predicted)
            conf_predicted = torch.cat(conf_predicted)
            cls_conf_predicted = torch.cat(cls_conf_predicted)
            if len(bboxes_predicted) != 0:
                if not conf.use_softnms:
                    picked_boxes = non_max_suppression(
                        xcycwh_2_xywh(bboxes_predicted).cpu().numpy(),
                        conf_predicted.cpu().numpy(),
                        conf.pred_nms_iou_threshold)
                else:
                    picked_boxes = soft_nms(
                        bboxes_predicted.cpu().numpy(),
                        conf_predicted.cpu().numpy(), 
                        sigma = conf.softnms_sigma,
                        Nt = conf.softnms_Nt,
                        threshold = conf.softnms_threshold,
                        method = conf.softnms_method)

                bboxes_group.append(trim_pred_bboxes(bboxes_predicted[picked_boxes], conf.input_size))
                labels_group.append(cls_predicted[picked_boxes])
                if return_confidence: 
                    confidences_group.append(conf_predicted[picked_boxes])
                    cls_conf_group.append(cls_conf_predicted[picked_boxes])
            else:
                bboxes_group.append(
                    torch.tensor([0., 0., 0.,
                                  0.]).unsqueeze(0).to(conf.device))
                labels_group.append(torch.tensor([0]).to(conf.device))
                if return_confidence:
                    confidences_group.append(torch.tensor([0]).to(conf.device))
                    cls_conf_group.append(torch.tensor([0]).to(conf.device))
    if return_confidence:
        return bboxes_group, labels_group, confidences_group, cls_conf_group
    else:
        return bboxes_group, labels_group

class Yolo(object):
    def __init__(self,
                 conf,
                 model=None,
                 train_loader=None,
                 val_loader=None,
                 optimizer=None,
                 init_writers = True):
        # for multi scale training
        self.steps = [0, 0, 0, 0, 0, 0, 0, 0]
        if init_writers:
            self.writers = [SummaryWriter(conf.log_path / 'writer_ft')] + \
            [
                SummaryWriter(conf.log_path / 'writer_{}'.format(resolution))
                for resolution in conf.resolutions[1:]
            ]
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seen = 0
        self.optimizer = optimizer
        self.resolution_num = len(conf.resolutions)
        self.res_idx = 0
        self.class_2_color = get_class_colors(conf)

    def save_state(self, conf, val_loss, extra=None, model_only=False):
        torch.save(
            self.model.state_dict(), conf.model_path /
            ('{}_val_loss:{}_model_seen:{}_step:{}_{}.pth'.format(
                get_time(), val_loss, self.seen, self.steps, extra)))
        if not model_only:
            torch.save(
                self.optimizer.state_dict(), conf.model_path /
                ('{}_val_loss:{}_optimizer_seen:{}_step:{}_{}.pth'.format(
                    get_time(), val_loss, self.seen, self.steps, extra)))

    def predict(self, conf, imgs, return_img=False, return_confidence=False):
        '''
        inputs :
        imgs : input tensor : shape [nB,3,input_size,input_size]
        return : PIL Image or bboxes_group and labels_group
        '''
        imgs = imgs.to(conf.device)
#         size = imgs.shape[-1]
        nB = len(imgs)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(imgs)
            if not return_confidence:
                bboxes_group, labels_group = calc_preds(
                    conf, preds, return_confidence = return_confidence)
            else:
                bboxes_group, labels_group, confidences_group, cls_conf_group \
                = calc_preds(conf, preds, return_confidence = return_confidence, training = self.model.training)
        self.model.train()

        if return_confidence:
            if return_img:
                return show_util_with_conf(conf, 0, imgs, labels_group, bboxes_group,\
                                           confidences_group, cls_conf_group,\
                                           conf.correct_id_2_class, self.class_2_color)
            else:
                return bboxes_group, labels_group, confidences_group, cls_conf_group
        else:
            if return_img:
                return show_util(conf, 0, imgs, labels_group, bboxes_group,
                             conf.correct_id_2_class, self.class_2_color)
            else:
                return bboxes_group, labels_group
    
    def detect_on_img(self, conf, img, level = 0, show_img = True, show_conf = False):
        '''
        detect with original img size
        img : PIL Image
        level : on which resolution to run detection, range[1 - 7], default is 416
        the resolution list is in conf.resolutions
        '''
        conf.input_size = conf.resolutions[level]
        conf.transform_test.transforms[0] = trans.Resize([conf.resolutions[level], conf.resolutions[level]])
        self.model.update_input_size(conf)
        input_img = conf.transform_test(img).unsqueeze(0)

        bboxes_group, labels_group, confidences_group, cls_conf_group = self.predict(conf, input_img, False, True)
        
        confidences, cls_conf = confidences_group[0].cpu(), cls_conf_group[0].cpu()      
        bboxes, labels = bboxes_group[0].cpu(), labels_group[0].cpu()
        
        bboxes_adjusted = adjust_bbox(img.size, conf.input_size, bboxes, detect=True)
        
        if show_img:        
            if show_conf:
                return_img = draw_bbox_class_with_conf(conf, img, labels, bboxes_adjusted,\
                                             conf.correct_id_2_class, self.class_2_color,\
                                             confidences, cls_conf, True)
            else:
                return_img = draw_bbox_class(conf, img, labels, bboxes_adjusted, conf.correct_id_2_class, self.class_2_color)        
            return return_img

        else:
            return bboxes_adjusted, labels, confidences, cls_conf


    def evaluate(self, conf, verbose=False):
        self.val_loader.current = 0
        self.model.eval()
        if verbose:
            loader = tqdm(iter(self.val_loader), total=conf.eva_batches)
        else:
            loader = iter(self.val_loader)

        running_loss = 0.
        running_loss_xy = 0.
        running_loss_wh = 0.
        running_loss_conf = 0.
        running_loss_cls = 0.
        n_correct = 0
        n_gt = 0
        n_pred = 0
        cls_correct_num = 0
        batch_count = 0
        warm_up = self.seen < conf.warm_up_img_num
        with torch.no_grad():
            for imgs, labels_group, bboxes_group in loader:
                if batch_count * self.val_loader.batch_size < conf.eva_seen:
                    imgs = imgs.to(conf.device)
                    for i, label in enumerate(labels_group):
                        labels_group[i] = label.to(conf.device)
                    for i, bboxes in enumerate(bboxes_group):
                        bboxes_group[i] = bboxes.to(conf.device)
                    preds = self.model(imgs)

                    targets, gt_mask, conf_weight, coord_mask = build_targets(
                        conf, preds.pred_bboxes_group, bboxes_group,
                        labels_group, self.model.head.anchors_group, warm_up)

                    losses = yolo_loss(conf, preds.loss_feats, targets,
                                       gt_mask, conf_weight, coord_mask)
                    running_loss += losses.loss_total.item()
                    running_loss_xy += losses.loss_xy
                    running_loss_wh += losses.loss_wh
                    running_loss_conf += losses.loss_conf
                    running_loss_cls += losses.loss_cls

                    bboxes_group_pred, labels_group_pred = calc_preds(conf, preds)

                    for nb in range(len(imgs)):
                        pred_bboxes = bboxes_group_pred[nb]
                        gt_bboxes = bboxes_group[nb]
                        ious = cal_ious_xcycwh(pred_bboxes,
                                               gt_bboxes).to(conf.device)
                        max_matched_iou_gt, max_matched_box_idx_gt = torch.max(
                            ious, dim=1)
                        matched_mask = max_matched_iou_gt > conf.evaluate_iou_threshold
                        matched_classes = labels_group_pred[nb][matched_mask]
                        cls_correct_num += torch.sum(
                            matched_classes == labels_group[nb][
                                max_matched_box_idx_gt][matched_mask]).item()
                        n_pred += len(pred_bboxes)
                        n_gt += len(gt_bboxes)
                        n_correct += torch.sum(matched_mask).item()
                    batch_count += 1
                else:
                    break

        precision = n_correct / n_gt
        recall = n_correct / n_pred
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        cls_acc = cls_correct_num / n_gt

        self.model.train()
        return running_loss/batch_count,\
                running_loss_xy/batch_count,\
                running_loss_wh/batch_count,\
                running_loss_conf/batch_count,\
                running_loss_cls/batch_count,\
                precision,recall,f1,cls_acc
    
    def eva_coco(self, conf, limit=0, level = 4):
        total = limit if limit else len(self.val_loader.dataset)
        image_ids = []
        results = []
        for i in tqdm(np.random.choice(np.random.permutation(len(self.val_loader.dataset)), total, replace=False)):
            bboxes, labels, confidences, cls_conf = self.detect_on_img(conf,self.val_loader.dataset[i][0], level = level, show_img=False)
            bboxes = xcycwh_2_xywh(bboxes)
            ids = set([item['image_id'] for item in self.val_loader.dataset[i][1]])
    #         assert len(ids) == 1 or len(ids) == 0, 'more than 1 image_id in one coco instance'
            if len(ids) == 0: continue
            image_id = ids.pop()
            image_ids.append(image_id)
            for k in range(len(labels)):
                result = {
                    "image_id": image_id,
                    "category_id": self.val_loader.dataset.maps[1][labels[k].item()],
                    "bbox": bboxes[k].numpy().tolist(),
                    "score": cls_conf[k].item(),
                }
                results.append(result)

        with open("results.json",'w',encoding='utf-8') as json_file:
            json.dump(results,json_file,ensure_ascii=False)

        coco_dt = self.val_loader.dataset.coco.loadRes("results.json")
        cocoEval = COCOeval(self.val_loader.dataset.coco, coco_dt, "bbox")
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        print('-------------------------------------------------------------------------------')
        cocoEval.summarize()

        return cocoEval


    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                num=None):
        if not num:
            num = len(self.train_loader) // 5
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for imgs, bboxes_group, labels_group in tqdm(
                iter(self.train_loader), total=num):
            batch_num += 1
            warm_up = True if self.seen < 12800 else False
            imgs = imgs.to(conf.device)
            for i, label in enumerate(labels_group):
                labels_group[i] = label.to(conf.device)
            for i, bboxes in enumerate(bboxes_group):
                bboxes_group[i] = bboxes.to(conf.device)

            self.optimizer.zero_grad()

            preds = self.model(imgs)

            targets, gt_mask, conf_weight, coord_mask = build_targets(
                conf, preds.pred_bboxes_group, bboxes_group, labels_group,
                self.model.head.anchors_group, warm_up)

            yolo_losses = yolo_loss(conf, preds.loss_feats, targets, gt_mask,
                                    conf_weight, coord_mask)

            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (
                1 - beta) * yolo_losses.loss_total.item()
            self.writers[0].add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writers[0].add_scalar('smoothed_loss', smoothed_loss,
                                       batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writers[0].add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            yolo_losses.loss_total.backward()
            self.optimizer.step()
            #Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
            if batch_num > num:
                return log_lrs, losses

    def train(self, conf, epochs):
        running_loss = 0.
        running_loss_xy = 0.
        running_loss_wh = 0.
        running_loss_conf = 0.
        running_loss_cls = 0.

        for e in range(epochs):
            time.sleep(2)
            for imgs, bboxes_group, labels_group in tqdm(
                    iter(self.train_loader)):

                warm_up = True if self.seen < 12800 else False

                imgs = imgs.to(conf.device)
                for i, label in enumerate(labels_group):
                    labels_group[i] = label.to(conf.device)
                for i, bboxes in enumerate(bboxes_group):
                    bboxes_group[i] = bboxes.to(conf.device)

                self.optimizer.zero_grad()

                preds = self.model(imgs)

                targets, gt_mask, conf_weight, coord_mask = build_targets(
                    conf, preds.pred_bboxes_group, bboxes_group, labels_group,
                    self.model.head.anchors_group, warm_up)

                losses = yolo_loss(conf, preds.loss_feats, targets, gt_mask,
                                   conf_weight, coord_mask)

                losses.loss_total.backward()

                if conf.gdclip:
                    clip_grad_norm_log_(
                        conf, self.optimizer.param_groups[0]['params'],
                        conf.gdclip, self.writer, self.steps[self.res_idx])

                self.optimizer.step()
                self.steps[self.res_idx] += 1
                self.seen += len(imgs)

                running_loss += losses.loss_total.item()
                running_loss_xy += losses.loss_xy
                running_loss_wh += losses.loss_wh
                running_loss_conf += losses.loss_conf
                running_loss_cls += losses.loss_cls

                if self.steps[self.res_idx] % conf.board_loss_every == 0:
                    if warm_up:
                        self.writers[0].add_scalar(
                            'loss_warm_up',
                            running_loss / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[0].add_scalar(
                            'loss_xy_warm_up',
                            running_loss_xy / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[0].add_scalar(
                            'loss_wh_warm_up',
                            running_loss_wh / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[0].add_scalar(
                            'loss_conf_warm_up',
                            running_loss_conf / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[0].add_scalar(
                            'loss_cls_warm_up',
                            running_loss_cls / conf.board_loss_every,
                            self.steps[self.res_idx])
                    else:
                        self.writers[self.res_idx].add_scalar(
                            'loss', running_loss / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[self.res_idx].add_scalar(
                            'loss_xy', running_loss_xy / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[self.res_idx].add_scalar(
                            'loss_wh', running_loss_wh / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[self.res_idx].add_scalar(
                            'loss_conf',
                            running_loss_conf / conf.board_loss_every,
                            self.steps[self.res_idx])
                        self.writers[self.res_idx].add_scalar(
                            'loss_cls',
                            running_loss_cls / conf.board_loss_every,
                            self.steps[self.res_idx])

                    running_loss = 0.
                    running_loss_xy = 0.
                    running_loss_wh = 0.
                    running_loss_conf = 0.
                    running_loss_cls = 0.

                if self.steps[self.res_idx] % conf.evaluate_every == 0:
                    val_loss,\
                    val_loss_xy,\
                    val_loss_wh,\
                    val_loss_conf,\
                    val_loss_cls,\
                    precision, recall, f1, cls_acc = self.evaluate(conf)

                    self.writers[self.res_idx].add_scalar(
                        'val_loss', val_loss, self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_loss_xy', val_loss_xy, self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_loss_wh', val_loss_wh, self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_loss_conf', val_loss_conf,
                        self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_loss_cls', val_loss_cls, self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_precision', precision, self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_recall', recall, self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_f1', f1, self.steps[self.res_idx])
                    self.writers[self.res_idx].add_scalar(
                        'val_cls_acc', cls_acc, self.steps[self.res_idx])

                if self.steps[self.res_idx] % conf.board_pred_image_every == 0:
                    imgs_board = []
                    for i in range(20):
                        img, _ = self.val_loader.dataset[i]
                        imgs_board.append(
                            self.val_loader.transform(img).unsqueeze(0))
                    imgs_board = torch.cat(imgs_board)
                    bboxes_group, labels_group, confidences_group, cls_conf_group = self.predict(
                        conf, imgs_board, return_img=False, return_confidence=True)

                    for i in range(20):
                        img = show_util_with_conf(conf, i, imgs_board, labels_group, bboxes_group,\
                                           confidences_group, cls_conf_group,\
                                           conf.correct_id_2_class, self.class_2_color)
                        self.writers[self.res_idx].add_image(
                            'pred_image_{}'.format(i),
                            trans.ToTensor()(img),
                            global_step=self.steps[self.res_idx])

                if self.steps[self.res_idx] % conf.save_every == 0:
                    self.save_state(
                        conf,
                        val_loss,
                        extra=conf.idx_2_res[str(self.res_idx)])