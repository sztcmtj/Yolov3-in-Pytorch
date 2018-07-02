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
from utils.dataset_tools import *
from utils.utils import *
from models.Yolo_head import Yolo_loss
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt

class Yolo(object):
    def __init__(self,conf,model=None,train_loader=None,val_loader=None,writer=None,optimizer=None,step=0,seen=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.step = step
        self.seen = seen
        self.optimizer = optimizer
#         self.yolo_loss_small = Yolo_loss(conf,conf.coco_anchors[:3],52)
#         self.yolo_loss_medium = Yolo_loss(conf,conf.coco_anchors[3:6],26)
#         self.yolo_loss_large = Yolo_loss(conf,conf.coco_anchors[6:],13)
    
    def save_state(self,conf,val_loss,extra=None):
        torch.save(self.model.state_dict(),conf.model_path/
                   ('{}_val_loss:{}_model_seen:{}_step:{}_{}.pth'.format(get_time(),val_loss,self.seen,self.step,extra)))
        torch.save(self.optimizer.state_dict(),conf.model_path/
                    ('{}_val_loss:{}_optimizer_seen:{}_step:{}_{}.pth'.format(get_time(),val_loss,self.seen,self.step,extra)))
        
    def predict(self,img,conf,threshold,only_objectness=True,return_img=False):
        self.model.eval()
        confidences_predicted = []
        bboxes_predicted = []
        classes_predicted = []
        with torch.no_grad():
            img = img.to(conf.device)
            y1_13x13,y2_26x26,y3_52x52 = self.model(img.unsqueeze(0))
            for output,feature_size,scale,anchors in [(y1_13x13,13,32.,self.yolo_loss_large.anchors),
                                                      (y2_26x26,26,16.,self.yolo_loss_medium.anchors),
                                                      (y3_52x52,52,8.,self.yolo_loss_small.anchors)]:
                output = output.view(1, 3, (5+conf.class_num), feature_size, feature_size) # [1, 255, 13, 13]
                xc = F.sigmoid(output[:,:,0,:,:])
                yc = F.sigmoid(output[:,:,1,:,:])
                w = output[:,:,2,:,:]
                h = output[:,:,3,:,:]
                confidence = torch.sigmoid(output[:,:,4,:,:])[0]
                cls = output[0,:,5:,:,:]
                cls = torch.sigmoid(cls)
                shift_x,shift_y = enumerate_shifted_anchor(1,feature_size,feature_size)
                shift_x,shift_y = shift_x.to(conf.device),shift_y.to(conf.device)
                xc_refine = (xc + shift_x).unsqueeze(-1) 
                #refine means map x,y to each square in the 13*13 or 26*26 or 52*52 grid
                yc_refine = (yc + shift_y).unsqueeze(-1)
                w_refine = (torch.exp(w) * anchors[:,0].view(1,3,1,1)).unsqueeze(-1)
                h_refine = (torch.exp(h) * anchors[:,1].view(1,3,1,1)).unsqueeze(-1)
                pred_bbox = torch.cat([xc_refine,yc_refine,w_refine,h_refine],dim=-1)[0] * scale
                max_cls_conf,class_predicted = torch.max(cls,dim=1)
                if only_objectness:
                    final_conf = confidence
                else:
                    final_conf = confidence*max_cls_conf
                mask = final_conf > threshold
                bbox_predicted = pred_bbox[mask]
                class_predicted = class_predicted[mask]
                confidence_predicted = final_conf[mask]
                bboxes_predicted.append(bbox_predicted)
                classes_predicted.append(class_predicted)
                confidences_predicted.append(confidence_predicted)
            bboxes = torch.cat(bboxes_predicted)
            classes = torch.cat(classes_predicted)
            confidences = torch.cat(confidences_predicted)
            self.model.train()
            if len(bboxes) != 0:
                picked_boxes = non_max_suppression(xcycwh_2_xywh(bboxes.detach()).clone().cpu().numpy(),
                                                   confidences.detach().clone().cpu().numpy(),
                                                   threshold)
                if return_img:
                    return trans.ToTensor()((draw_bbox_class(trans.ToPILImage()(de_preprocess(img.cpu(),conf.mean,conf.std)),
                                                              classes[picked_boxes].cpu(),
                                                              bboxes[picked_boxes].cpu(),
                                                              self.train_loader.dataset.maps[2])))
                else:
                    return bboxes[picked_boxes],classes[picked_boxes],confidences[picked_boxes]
            else:
                if return_img:
                    return de_preprocess(img.cpu(),conf.mean,conf.std)
                else:
                    return torch.tensor([0.,0.,0.,0.]).unsqueeze(0).to(conf.device),\
                torch.tensor([0]).to(conf.device),\
                torch.tensor([0.]).to(conf.device)            
    
    def evaluate(self,conf,value_batches=100,img_batches=50,verbose=False):
        self.val_loader.current = 0
        self.model.eval()
        if verbose:
            loader = tqdm(iter(self.val_loader),total = max(value_batches,img_batches))
        else:
            loader = iter(self.val_loader)
        
        running_loss = 0.
        running_nGT = 0.
        running_nCorrect = 0.
        running_loss_x = 0.
        running_loss_y = 0.
        running_loss_w = 0.
        running_loss_h = 0.
        running_loss_conf = 0.
        running_loss_cls = 0.
        n_correct = 0
        n_gt = 0
        n_pred = 0
        cls_correct_num = 0
        batch_count = 0    
        with torch.no_grad():         
            for imgs,labels_group,bboxes_group in loader:   
                if batch_count < value_batches:
                    imgs = imgs.to(conf.device)
                    
                    for i,label in enumerate(labels_group):
                        labels_group[i] = label.to(conf.device)
                    for i,bboxes in enumerate(bboxes_group):
                        bboxes_group[i] = bboxes.to(conf.device)

                    y1_13x13,y2_26x26,y3_52x52 = self.model(imgs)
                    
                    labels_group_small,labels_group_medium,labels_group_large,\
                    bboxes_group_small,bboxes_group_medium,bboxes_group_large,\
                    best_anchors_idx_group_small,\
                    best_anchors_idx_group_medium,\
                    best_anchors_idx_group_large = arrange_bbox_label(conf.device,conf.coco_anchors,labels_group,bboxes_group)

                    large_feature,large_bboxes,\
                    large_labels,\
                    large_anchors_idx = prepare_loss_input(conf.device, y1_13x13,
                                                           labels_group_large,
                                                           bboxes_group_large,
                                                           best_anchors_idx_group_large)

                    medium_feature,\
                    medium_bboxes,\
                    medium_labels,\
                    medium_anchors_idx = prepare_loss_input(conf.device ,y2_26x26,
                                                            labels_group_medium,
                                                            bboxes_group_medium,
                                                            best_anchors_idx_group_medium)

                    small_feature,\
                    small_bboxes,\
                    small_labels,\
                    small_anchors_idx = prepare_loss_input(conf.device, y3_52x52,
                                                           labels_group_small,
                                                           bboxes_group_small,
                                                           best_anchors_idx_group_small)

                    if len(large_feature) != 0:
                        loss_large,nGT_large,nCorrect_large,\
                        loss_x_large,loss_y_large,loss_w_large,loss_h_large,\
                        loss_conf_large,loss_cls_large = self.yolo_loss_large(large_feature,
                                                                         large_bboxes,
                                                                         large_labels,
                                                                         large_anchors_idx,
                                                                         warm_up=False)
                        running_loss += loss_large.item()
                        running_nGT += nGT_large 
                        running_nCorrect += nCorrect_large
                        running_loss_x += loss_x_large
                        running_loss_y += loss_y_large
                        running_loss_w += loss_w_large
                        running_loss_h += loss_h_large
                        running_loss_conf += loss_conf_large
                        running_loss_cls += loss_cls_large

                    if len(medium_feature) != 0:
                        loss_medium,nGT_medium,nCorrect_medium,\
                        loss_x_medium,loss_y_medium,loss_w_medium,loss_h_medium,\
                        loss_conf_medium,loss_cls_medium = self.yolo_loss_medium(medium_feature,
                                                                            medium_bboxes,
                                                                            medium_labels,
                                                                            medium_anchors_idx,
                                                                            warm_up=False)
                        running_loss += loss_medium.item()
                        running_nGT += nGT_medium 
                        running_nCorrect += nCorrect_medium
                        running_loss_x += loss_x_medium
                        running_loss_y += loss_y_medium
                        running_loss_w += loss_w_medium
                        running_loss_h += loss_h_medium
                        running_loss_conf += loss_conf_medium
                        running_loss_cls += loss_cls_medium

                    if len(small_feature) != 0:
                        loss_small,nGT_small,nCorrect_small,\
                        loss_x_small,loss_y_small,loss_w_small,loss_h_small,\
                        loss_conf_small,loss_cls_small = self.yolo_loss_small(small_feature,
                                                                         small_bboxes,
                                                                         small_labels,
                                                                         small_anchors_idx,
                                                                         warm_up=False)
                        running_loss += loss_small.item()
                        running_nGT += nGT_small 
                        running_nCorrect += nCorrect_small
                        running_loss_x += loss_x_small
                        running_loss_y += loss_y_small
                        running_loss_w += loss_w_small
                        running_loss_h += loss_h_small
                        running_loss_conf += loss_conf_small
                        running_loss_cls += loss_cls_small
                        
                if batch_count < img_batches:
                    for j in range(len(imgs)):
                        pred_bboxes,pred_classes,_ = self.predict(imgs[j],
                                                                  conf,
                                                                  conf.predict_confidence_threshold,
                                                                  only_objectness=True,
                                                                  return_img=False)
                        
                        ious = cal_ious(xcycwh_2_xywh(pred_bboxes),xcycwh_2_xywh(bboxes_group[j])).to(conf.device)
                        max_matched_iou_gt,max_matched_box_idx_gt = torch.max(ious,dim=0)
                        matched_mask = max_matched_iou_gt > conf.evaluate_iou_threshold
                        matched_classes = pred_classes[max_matched_box_idx_gt][matched_mask]
                        cls_correct_num += torch.sum(matched_classes == labels_group[j][matched_mask]).item()
                        n_pred += len(pred_bboxes)
                        n_gt += len(bboxes_group[j])
#                         print('inner loop have {} gt'.format(n_gt))
                        n_correct += torch.sum(matched_mask).item()
                
                batch_count += 1

                if batch_count >= max(value_batches,img_batches):
                    break
  
        precision = n_correct/n_gt
        recall = n_correct/n_pred
        f1 = 2*precision*recall / (precision + recall + 1e-8)
        cls_acc = cls_correct_num/n_gt
        
        self.model.train()
            
        return running_loss/value_batches,\
                running_nCorrect/running_nGT,\
                running_loss_x/value_batches,\
                running_loss_y/value_batches,\
                running_loss_w/value_batches,\
                running_loss_h/value_batches,\
                running_loss_conf/value_batches,\
                running_loss_cls/value_batches,\
                precision,recall,f1,cls_acc
    
    def find_lr(self,conf,init_value = 1e-8, final_value=10., beta = 0.98,num = None):
        if not num:
            num = len(self.train_loader) // 5
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for imgs,labels_group,bboxes_group in tqdm(iter(self.train_loader),total=num):
            batch_num += 1
            imgs = imgs.to(conf.device)
            for i,label in enumerate(labels_group):
                labels_group[i] = label.to(conf.device)
            for i,bboxes in enumerate(bboxes_group):
                bboxes_group[i] = bboxes.to(conf.device)            
                
            self.optimizer.zero_grad()
            
            y1_13x13,y2_26x26,y3_52x52 = self.model(imgs)
            
            labels_group_small,labels_group_medium,labels_group_large,\
            bboxes_group_small,bboxes_group_medium,bboxes_group_large,\
            best_anchors_idx_group_small,\
            best_anchors_idx_group_medium,\
            best_anchors_idx_group_large = arrange_bbox_label(conf.device,conf.coco_anchors,labels_group,bboxes_group)
            
            large_feature,large_bboxes,\
            large_labels,\
            large_anchors_idx = prepare_loss_input(conf.device,
                                                   y1_13x13,
                                                   labels_group_large,
                                                   bboxes_group_large,
                                                   best_anchors_idx_group_large)

            medium_feature,\
            medium_bboxes,\
            medium_labels,\
            medium_anchors_idx = prepare_loss_input(conf.device,
                                                    y2_26x26,
                                                    labels_group_medium,
                                                    bboxes_group_medium,
                                                    best_anchors_idx_group_medium)

            small_feature,\
            small_bboxes,\
            small_labels,\
            small_anchors_idx = prepare_loss_input(conf.device,
                                                   y3_52x52,
                                                   labels_group_small,
                                                   bboxes_group_small,
                                                   best_anchors_idx_group_small)
            
            warm_up = True if self.seen < 12800 else False
            
            if len(large_feature) != 0:
                loss_large, _, _,\
                 _, _, _, _,\
                 _, _ = self.yolo_loss_large(large_feature,
                                                                 large_bboxes,
                                                                 large_labels,
                                                                 large_anchors_idx,
                                                                 warm_up)
            else:
                loss_large = 0
                
            if len(medium_feature) != 0:
                loss_medium, _, _,\
                 _, _, _, _,\
                 _, _ = self.yolo_loss_medium(medium_feature,
                                                                    medium_bboxes,
                                                                    medium_labels,
                                                                    medium_anchors_idx,
                                                                    warm_up)
            else:
                loss_medium = 0
                
            if len(small_feature) != 0:
                loss_small, _, _,\
                 _, _, _, _,\
                 _, _ = self.yolo_loss_small(small_feature,
                                                                 small_bboxes,
                                                                 small_labels,
                                                                 small_anchors_idx,
                                                                 warm_up)
            else:
                loss_small = 0
            
            loss = loss_small + loss_medium + loss_large
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) *loss.item()
            self.writer.add_scalar('avg_loss',avg_loss,batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss',smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 10 * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5],losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr',math.log10(lr),batch_num)
            #Do the SGD step
            loss.backward()
            self.optimizer.step()
            #Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
            if batch_num > num:
                return log_lrs, losses

    def train(self,conf,epochs=1,log=None):
        stagnate = 0
        
        running_loss = 0.
        running_nGT = 0.
        running_nCorrect = 0.
        running_loss_x = 0.
        running_loss_y = 0.
        running_loss_w = 0.
        running_loss_h = 0.
        running_loss_conf = 0.
        running_loss_cls = 0.      
        
        for e in range(epochs):
            self.train_loader.current = 0
            for imgs,labels_group,bboxes_group in tqdm(iter(self.train_loader)):

                
                imgs = imgs.to(conf.device)
                for i,label in enumerate(labels_group):
                    labels_group[i] = label.to(conf.device)
                for i,bboxes in enumerate(bboxes_group):
                    bboxes_group[i] = bboxes.to(conf.device)

                self.optimizer.zero_grad()
                
                y1_13x13,y2_26x26,y3_52x52 = self.model(imgs)
#                 if torch.sum(torch.isnan(y1_13x13)).item() > 0 or torch.sum(torch.isnan(y2_26x26)).item() > 0 or torch.sum(torch.isnan(y3_52x52)).item() > 0:
#                     pdb.set_trace()          

                labels_group_small,labels_group_medium,labels_group_large,\
                bboxes_group_small,bboxes_group_medium,bboxes_group_large,\
                best_anchors_idx_group_small,\
                best_anchors_idx_group_medium,\
                best_anchors_idx_group_large = arrange_bbox_label(conf.device,conf.coco_anchors,labels_group,bboxes_group)

                large_feature,large_bboxes,\
                large_labels,\
                large_anchors_idx = prepare_loss_input(conf.device,
                                                       y1_13x13,
                                                       labels_group_large,
                                                       bboxes_group_large,
                                                       best_anchors_idx_group_large)

                medium_feature,\
                medium_bboxes,\
                medium_labels,\
                medium_anchors_idx = prepare_loss_input(conf.device,
                                                        y2_26x26,
                                                        labels_group_medium,
                                                        bboxes_group_medium,
                                                        best_anchors_idx_group_medium)

                small_feature,\
                small_bboxes,\
                small_labels,\
                small_anchors_idx = prepare_loss_input(conf.device,
                                                       y3_52x52,
                                                       labels_group_small,
                                                       bboxes_group_small,
                                                       best_anchors_idx_group_small)

                warm_up = True if self.seen < 12800 else False

                if len(large_feature) != 0:
                    loss_large,nGT_large,nCorrect_large,\
                    loss_x_large,loss_y_large,loss_w_large,loss_h_large,\
                    loss_conf_large,loss_cls_large = self.yolo_loss_large(large_feature,
                                                                     large_bboxes,
                                                                     large_labels,
                                                                     large_anchors_idx,
                                                                     warm_up)
    #                 loss += loss_large
                    running_nGT += nGT_large 
                    running_nCorrect += nCorrect_large
                    running_loss_x += loss_x_large
                    running_loss_y += loss_y_large
                    running_loss_w += loss_w_large
                    running_loss_h += loss_h_large
                    running_loss_conf += loss_conf_large
                    running_loss_cls += loss_cls_large
                    
#                     print('loss_large = {}'.format(loss_large))
                    
                else:
                    loss_large = 0
#                     print('no large loss counted')

                if len(medium_feature) != 0:
                    loss_medium,nGT_medium,nCorrect_medium,\
                    loss_x_medium,loss_y_medium,loss_w_medium,loss_h_medium,\
                    loss_conf_medium,loss_cls_medium = self.yolo_loss_medium(medium_feature,
                                                                        medium_bboxes,
                                                                        medium_labels,
                                                                        medium_anchors_idx,
                                                                        warm_up)
    #                 loss += loss_medium
                    running_nGT += nGT_medium 
                    running_nCorrect += nCorrect_medium
                    running_loss_x += loss_x_medium
                    running_loss_y += loss_y_medium
                    running_loss_w += loss_w_medium
                    running_loss_h += loss_h_medium
                    running_loss_conf += loss_conf_medium
                    running_loss_cls += loss_cls_medium
                    
#                     print('loss_medium = {}'.format(loss_medium))
                    
                else:
                    loss_medium = 0
#                     print('no medium loss counted')

                if len(small_feature) != 0:
                    loss_small,nGT_small,nCorrect_small,\
                    loss_x_small,loss_y_small,loss_w_small,loss_h_small,\
                    loss_conf_small,loss_cls_small = self.yolo_loss_small(small_feature,
                                                                     small_bboxes,
                                                                     small_labels,
                                                                     small_anchors_idx,
                                                                     warm_up)
                    
    #                 loss += loss_small
                    running_nGT += nGT_small 
                    running_nCorrect += nCorrect_small
                    running_loss_x += loss_x_small
                    running_loss_y += loss_y_small
                    running_loss_w += loss_w_small
                    running_loss_h += loss_h_small
                    running_loss_conf += loss_conf_small
                    running_loss_cls += loss_cls_small
                    
#                     print('loss_small = {}'.format(loss_small))                    
                    
                else:
                    loss_small = 0
#                     print('no small loss counted')

                loss = loss_small + loss_medium + loss_large
                loss.backward()
                if conf.gdclip:
                    clip_grad_norm_log_(conf,self.model.parameters(), conf.gdclip,self.writer,self.step)
                self.optimizer.step()
                self.step += 1
                self.seen += len(imgs)

                running_loss += loss.detach().item()

                if self.step % conf.board_loss_every == 0:
                    if warm_up:
                        self.writer.add_scalar('loss_warm_up',running_loss/conf.board_loss_every,self.step)
                        self.writer.add_scalar('pseudo_recall_warm_up',running_nCorrect/running_nGT,self.step)
                        self.writer.add_scalar('loss_x_warm_up',running_loss_x/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_y_warm_up',running_loss_y/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_w_warm_up',running_loss_w/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_h_warm_up',running_loss_h/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_conf_warm_up',running_loss_conf/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_cls_warm_up',running_loss_cls/conf.board_loss_every,self.step)
                    else:
                        self.writer.add_scalar('loss',running_loss/conf.board_loss_every,self.step)
                        self.writer.add_scalar('pseudo_recall',running_nCorrect/running_nGT,self.step)
                        self.writer.add_scalar('loss_x',running_loss_x/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_y',running_loss_y/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_w',running_loss_w/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_h',running_loss_h/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_conf',running_loss_conf/conf.board_loss_every,self.step)
                        self.writer.add_scalar('loss_cls',running_loss_cls/conf.board_loss_every,self.step)

                    running_loss = 0.
                    running_nGT = 0.
                    running_nCorrect = 0.
                    running_loss_x = 0.
                    running_loss_y = 0.
                    running_loss_w = 0.
                    running_loss_h = 0.
                    running_loss_conf = 0.
                    running_loss_cls = 0.

                if self.step % conf.evaluate_every == 0:
                    val_loss,\
                    pseudo_recall,\
                    val_loss_x,\
                    val_loss_y,\
                    val_loss_w,\
                    val_loss_h,\
                    val_loss_conf,\
                    val_loss_cls,\
                    precision,recall,f1,cls_acc = self.evaluate(conf,100,50)

                    self.writer.add_scalar('val_loss',val_loss,self.step)
                    self.writer.add_scalar('val_pseudo_recall',pseudo_recall,self.step)
                    self.writer.add_scalar('val_loss_x',val_loss_x,self.step)
                    self.writer.add_scalar('val_loss_y',val_loss_y,self.step)
                    self.writer.add_scalar('val_loss_w',val_loss_w,self.step)
                    self.writer.add_scalar('val_loss_h',val_loss_h,self.step)
                    self.writer.add_scalar('val_loss_conf',val_loss_conf,self.step)
                    self.writer.add_scalar('val_loss_cls',val_loss_cls,self.step)
                    self.writer.add_scalar('val_precision',precision,self.step)
                    self.writer.add_scalar('val_recall',recall,self.step)
                    self.writer.add_scalar('val_f1',f1,self.step)
                    self.writer.add_scalar('val_cls_acc',cls_acc,self.step)    

                if self.step % conf.board_pred_image_every == 0:
                    for i in range(20):
                        img,_ = self.val_loader.dataset[i]
                        img_tensor = self.predict(self.val_loader.transform(img),
                                                  conf,
                                                  conf.predict_confidence_threshold,
                                                  only_objectness=True,
                                                  return_img=True)
                        self.writer.add_image('pred_image_{}'.format(i),img_tensor,global_step=self.step)

                if self.step % conf.save_every == 0:
                    self.save_state(conf,val_loss,extra=log)  