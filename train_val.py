import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch_optimizer as optim
from torch.nn.modules.loss import _Loss

import cv2
import time
import os, sys
import pandas as pd
import albumentations as A
import numpy as np
from glob import glob
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
import time
from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

# import timm

from segmentation_models_pytorch.losses import DiceLoss

from model_utils import timm
from model_utils.model import GHDModel, load_model
from dataset import GHDDataset
from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--do_train', help='do_train', required=False, type=bool,default=False)
    parser.add_argument('--resume', help='resume', required=False, type=bool,default=False)
    parser.add_argument('--ckpt_path', help='ckpt_path', required=False, type=str,default='')
    args = parser.parse_args()
    return args

args = parse_args()



do_train = args.do_train
resume = args.resume
pretrained_weight_path = './ckpt/convnext_pico_ols_d1-611f0ca7.pth'

ckpt_path = args.ckpt_path 
save_model_dir = './ckpt'
os.makedirs(save_model_dir,exist_ok=True)


label_mapping_rvs = ['nag','p','ag','lgd','hgd','c'] 

device = torch.device("cuda")

batch_size = 4 
max_epoch = 500
bag_size = 4

soft_label = 0.95
pos_idx = 3
c_idx = 4

mask_size = 48 

eval_interval = 1


val_df = pd.read_csv('./data/datases/sample_data.csv')
train_df = pd.read_csv('./data/datases/sample_data.csv')

pkl_image_root = './data/patches'




data_val = GHDDataset(val_df,label_mapping_rvs=label_mapping_rvs, soft_label=soft_label,mask_size = mask_size,
                    section='val',pkl_image_root=pkl_image_root,bag_size=bag_size,c_idx=c_idx)
val_data_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True, pin_memory=False,num_workers=4)





if do_train:


    
    data_train = GHDDataset(train_df,label_mapping_rvs=label_mapping_rvs, soft_label=soft_label,mask_size = mask_size,
                        section='train',upsample={'hgd':10, 'c':2, 'lgd':5},pkl_image_root=pkl_image_root,bag_size=bag_size,c_idx=c_idx)
    train_data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=False,num_workers=10)



    if resume:
        model = load_model(C =len(label_mapping_rvs),weight_path=ckpt_path,strict=True,device=device)
    else:
        model = load_model(C =len(label_mapping_rvs),weight_path=pretrained_weight_path,device=device)
    
    # torch.save(model.state_dict(), save_model_dir+f'/ckpt_0000.pth')
   
    optimizer = optim.RAdam(model.parameters(), 0.00005)
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function_bce = torch.nn.BCEWithLogitsLoss()#(pos_weight=torch.Tensor([0.5,0.5,0.5,0.5,0.6,0.7,0.6,0.6]))
    loss_function_seg = DiceLoss(mode='multiclass',classes=[0,1,2,3],from_logits=False,ignore_index=-1,log_loss=True)
    # loss_function_seg = TverskyLoss(mode='multiclass',classes=[0,1,2,3],from_logits=False,ignore_index=-1,log_loss=True)
    
    loss_function = MULLOSS(loss_function_bce,loss_function_seg)
    # loss_function = FocalLoss(gamma=2)

    batches_per_epoch = len(train_data_loader)
    batches_val = len(val_data_loader)
    print('batches_per_epoch',batches_per_epoch)
    writer = SummaryWriter(log_dir=save_model_dir+'/logs_001',flush_secs=10)

    for epoch in range(max_epoch):
        logging.info('Start epoch {}'.format(epoch))
        model.train()
        epoch_loss = 0
        epoch_loss_seg = 0
        epoch_loss_bce = 0
        train_time_sp = time.time()
        pred_all = []
        label_all = []
        for batch_id, batch_data in enumerate(train_data_loader):
            images, masks, labels, _, _ = batch_data
            # t_n = np.random.choice([6,7,8],1)[0]
            # images = images[:,:t_n]
            # labels = labels[:,:t_n]
            images = images.float().to(device)
            label_all.extend(labels.numpy())
            labels = labels.float().to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            pred_labels,pred_mask = model(images) 

            bs, num_instances, w, h = masks.shape     
            masks = masks.view(bs*num_instances, w, h)      

            loss_value,bce,dice = loss_function(pred_labels, labels, pred_mask, masks)
            loss_value.backward()                
            optimizer.step()

            epoch_loss += loss_value.item()
            epoch_loss_seg += dice.item()
            epoch_loss_bce += bce.item()
            m = sigmoid(pred_labels.cpu().detach().numpy())
            pred_all.extend(m)

            used_time = (time.time() - train_time_sp) / (batch_id+1)
            writer.add_scalar('Loss/train', epoch_loss/(1+batch_id), batches_per_epoch *epoch + batch_id)
            writer.add_scalar('Loss/train_dice', epoch_loss_seg/(1+batch_id), batches_per_epoch *epoch + batch_id)
            writer.add_scalar('Loss/train_bce', epoch_loss_bce/(1+batch_id), batches_per_epoch *epoch + batch_id)
            logging.info(
                f'epoch {epoch} Batch: {batch_id}/{batches_per_epoch}, loss = {loss_value.item():.3f}, avg_batch_time = {used_time:.3f}')
          
        pred_all = np.array(pred_all)
        label_all = np.array(label_all)        
        pred_label = np.round(pred_all)

        f1s = []
        aucs = []
        for i in range(pred_all.shape[1]):
            tmp_y = np.round(label_all[:,i])
            tmp_p = pred_all[:,i]
            tmp_label = np.round(pred_label[:,i])
            f1_cls = metrics.f1_score(tmp_y,tmp_label)
            try:
                auc_cls = metrics.roc_auc_score(tmp_y,tmp_p)
            except Exception as e:
                auc_cls = 0
            f1s.append(f1_cls)
            aucs.append(auc_cls)
            writer.add_scalar(f'ACC_train/auc_{label_mapping_rvs[i]}', auc_cls, epoch)
            writer.add_scalar(f'ACC_train/f1-score_{label_mapping_rvs[i]}', f1_cls, epoch)
        f1_micro = np.mean(f1s)
        auc_micro = np.mean(aucs)
        writer.add_scalar(f'ACC_train/f1-score_micro', f1_micro, epoch)
        writer.add_scalar(f'ACC_train/auc_micro', auc_micro, epoch)

        highest_cls_pred = [get_h_label(i) for i in pred_all]
        highest_cls_true = [get_h_label(i) for i in label_all]

        cm = metrics.confusion_matrix(highest_cls_true,highest_cls_pred,labels=list(range(len(label_mapping_rvs))))
        sen_c = np.sum(cm[c_idx:,pos_idx:])/np.sum(cm[c_idx:])
        sen_pos = np.sum(cm[pos_idx:,pos_idx:])/np.sum(cm[pos_idx:])
        sp = np.sum(cm[:pos_idx,:pos_idx])/np.sum(cm[:pos_idx])
        writer.add_scalar('ACC_train/sen_pos', sen_pos, epoch)
        writer.add_scalar('ACC_train/sen_c', sen_c, epoch)
        writer.add_scalar('ACC_train/sp', sp, epoch)
        writer.add_scalar('ACC_train/spsen', (sp+sen_pos)/2, epoch)

        if epoch % eval_interval == 0:
            model.eval()  
            pred_all = []
            label_all = []
            filename_all = []
            slide_num_adj_all = []
            with torch.no_grad():
                val_loss = 0
                val_loss_seg = 0
                val_loss_bce = 0
                for batch_id, batch_data in tqdm(enumerate(val_data_loader)):    
                    images, masks, labels, filename, slide_num_adj = batch_data
                    images = images.float().to(device)            
                    label_all.extend(labels.numpy())
                    labels = labels.float().to(device)
                    masks = masks.to(device)
                    optimizer.zero_grad()
                    pred_labels,pred_mask = model(images)

                    bs, num_instances, w, h = masks.shape     
                    masks = masks.view(bs*num_instances, w, h)      

                    loss_value,bce,dice = loss_function(pred_labels, labels, pred_mask, masks)
                    val_loss += loss_value.item()
                    val_loss_seg += dice.item()
                    val_loss_bce += bce.item()

                    m = sigmoid(pred_labels.cpu().detach().numpy())
                    pred_all.extend(m)
                    filename_all.extend(filename)
                    slide_num_adj_all.extend(slide_num_adj)
                pred_all = np.array(pred_all)
                label_all = np.array(label_all)
                pred_label = np.round(pred_all)
                writer.add_scalar('Loss/val', val_loss/batches_val, epoch)
                writer.add_scalar('Loss/val_dice', val_loss_seg/batches_val, epoch)
                writer.add_scalar('Loss/val_bce', val_loss_bce/batches_val, epoch)


                f1s = []
                aucs = []
                for i in range(pred_all.shape[1]):
                    tmp_y = np.round(label_all[:,i])
                    tmp_p = pred_all[:,i]
                    tmp_label = np.round(pred_label[:,i])
                    f1_cls = metrics.f1_score(tmp_y,tmp_label)
                    try:
                        auc_cls = metrics.roc_auc_score(tmp_y,tmp_p)
                    except Exception as e:
                        auc_cls = 0
                    f1s.append(f1_cls)
                    aucs.append(auc_cls)
                    writer.add_scalar(f'AUC_val/auc_{label_mapping_rvs[i]}', auc_cls, epoch)
                    writer.add_scalar(f'ACC_val/f1-score_{label_mapping_rvs[i]}', f1_cls, epoch)
                

                f1_micro = np.mean(f1s)
                auc_micro = np.mean(aucs)
                writer.add_scalar(f'ACC_val/f1-score_micro', f1_micro, epoch)
                writer.add_scalar(f'AUC_val/auc_micro', auc_micro, epoch)

                highest_cls_pred = [get_h_label(i) for i in pred_all]
                highest_cls_true = [get_h_label(i) for i in label_all]

                
                cm = metrics.confusion_matrix(highest_cls_true,highest_cls_pred,labels=list(range(len(label_mapping_rvs))))
                sen_c = np.sum(cm[c_idx:,pos_idx:])/np.sum(cm[c_idx:])
                sen_pos = np.sum(cm[pos_idx:,pos_idx:])/np.sum(cm[pos_idx:])
                sp = np.sum(cm[:pos_idx,:pos_idx])/np.sum(cm[:pos_idx])
                writer.add_scalar('ACC_val/sen_pos', sen_pos, epoch)
                writer.add_scalar('ACC_val/sen_c', sen_c, epoch)
                writer.add_scalar('ACC_val/sp', sp, epoch)
                writer.add_scalar('ACC_val/spsen', (sp+sen_pos)/2, epoch)
                
                torch.save(model.state_dict(), save_model_dir+f'/ckpt_{epoch}_{sp:.4f}_{sen_pos:.4f}_{sen_c:.4f}.pth')
               
                used_time = time.time() - train_time_sp    
        logging.info(
            f'epoch {epoch}, epoch time = {used_time:.3f}')
else:
    model = load_model(C =len(label_mapping_rvs),weight_path=ckpt_path,strict=True,device=device)
    model.eval()  
    pred_all = []
    label_all = []
    filename_all = []
    slide_num_adj_all = []
    with torch.no_grad():
        val_loss = 0
        for batch_id, batch_data in tqdm(enumerate(val_data_loader)):    
            images, masks, labels, filename, slide_num_adj = batch_data
            images = images.float().to(device)
            label_all.extend(labels.numpy())
            labels = labels.float().to(device)
            masks = masks.to(device)
            pred_labels,pred_mask = model(images)

            bs, num_instances, w, h = masks.shape     
            masks = masks.view(bs*num_instances, w, h)      

            # dice = loss_function_seg(pred_mask, masks)
            # val_loss += dice.item()

            m = sigmoid(pred_labels.cpu().detach().numpy())
            pred_all.extend(m)
            filename_all.extend(filename)
            slide_num_adj_all.extend(slide_num_adj)
        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        pred_label = np.round(pred_all)
        label_all = np.round(label_all)

        f1s = []
        for i in range(pred_all.shape[1]):
            tmp_y = label_all[:,i]
            tmp_p = pred_all[:,i]
            tmp_label = pred_label[:,i]
            f1_cls = metrics.f1_score(tmp_y,tmp_label)
            print(f'{label_mapping_rvs[i]} f1-score = {f1_cls:.5f}')
            try:
                auc_cls = metrics.roc_auc_score(tmp_y,tmp_p)
            except Exception as e:
                auc_cls = 0
            print(f'{label_mapping_rvs[i]} auc-score = {auc_cls:.5f}')

        pred_label = get_final_label(pred_all,threshs=0.5)  
        
        highest_cls_pred = [get_h_label(i) for i in pred_label]
        highest_cls_true = [get_h_label(i) for i in label_all]

          
        cm_mul = metrics.multilabel_confusion_matrix(label_all,pred_label,labels=list(range(len(label_mapping_rvs))))
        cm = metrics.confusion_matrix(highest_cls_true,highest_cls_pred,labels=list(range(len(label_mapping_rvs))))
        # rpt = metrics.classification_report(label_all,pred_label,output_dict=False,digits=4)
        sen_c = np.sum(cm[c_idx:,pos_idx:])/np.sum(cm[c_idx:])
        sen_pos = np.sum(cm[pos_idx:,pos_idx:])/np.sum(cm[pos_idx:])
        sp = np.sum(cm[:pos_idx,:pos_idx])/np.sum(cm[:pos_idx])
        print(f'sp {sp}\nsen_pos {sen_pos}\nsen_c {sen_c}\n')
        print(cm_mul)
        print(cm)
                
        result_df = pd.DataFrame(pred_all)
        for i in range(pred_all.shape[1]):
            result_df[label_mapping_rvs[i] +'_pred'] = pred_label[:,i]
        for i in range(pred_all.shape[1]):
            result_df[label_mapping_rvs[i]+'_true'] = label_all[:,i]
        result_df['filename'] = filename_all
        result_df['pred_label'] = [label_mapping_rvs[a] for a in highest_cls_pred]
        result_df['gt'] = [label_mapping_rvs[a] for a in highest_cls_true]
        result_df['slide_num_adj_all'] = slide_num_adj_all
        result_df['mul_pred_6c'] = ['_'.join([label_mapping_rvs[j] for j in range(len(i)) if i[j]==1]) for i in pred_label]
        result_df['miss_highest_prob'] = [max([pred_all[i][ii] for ii in range(6) if label_all[i][ii]==0]+[-1]) for i in range(len(pred_all))]
      
        result_df.to_csv(save_result_csv_path, index=False)
 

