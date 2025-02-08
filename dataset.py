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



class GHDDataset(Dataset):
    def __init__(
            self,
            data_df,
            label_mapping_rvs,
            section='train',
            input_size = 768,
            pkl_image_root = '',
            upsample = {},
            bag_size = 4,
            soft_label = 0.95,
            c_idx = 4,
            pos_idx = 3,
            mask_size = 48
        ):
       
        self.section = section
        self.input_size = input_size
        self.pkl_image_root = pkl_image_root
        self.label_mapping_rvs = label_mapping_rvs
        self.soft_label = soft_label
        self.c_idx = c_idx
        self.pos_idx = pos_idx
        self.mask_size = mask_size
        self.upsample = upsample
        if section=='train':
            self.bag_size = bag_size
        else:
            self.bag_size = 8
            
        self.label_mapping = {'nag':0,'p':1,'ag':2,'lgd':3,'hgd':4,'c':5}

        self.data_df = self.filter_not_exists(data_df)
        self.aug_seq = self.create_aug_seq()

 

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, indx):
        filename = self.data_df.loc[indx,'filename_list']
        filename = np.random.choice(filename,1)[0]

        labels = self.data_df.loc[indx,'labels']
        if isinstance(labels,list):
            label = np.array(labels)
        else:
            raise NotImplementedError
        
        if (np.max(label[1:])>0.5) and (label[0]>0.5):  # 所有其他都不是的时候才是nag
            label[0] = 1-self.soft_label
        if (np.max(label[self.c_idx:])>0.5):  # 有高级别或以上，不报阴性类
            for i in range(3):
                label[i] = 1-self.soft_label


        all_patches_path = list(glob(f'{self.pkl_image_root}/{filename}/*_image.png'))


        if len(all_patches_path)>self.bag_size:
            chosen_patch_path = np.random.choice(all_patches_path,self.bag_size,replace=False)
        elif len(all_patches_path)>0:
            chosen_patch_path = np.random.choice(all_patches_path,len(all_patches_path)-1,replace=False)
        else:
            with open('./error.txt','a+') as f:
                f.write(f'{filename} no image \n\n')
        all_images = []
        all_mask = []
        if self.section=='train':
            np.random.shuffle(chosen_patch_path)
        for path in chosen_patch_path:
            sub_im = cv2.imread(path)
            mask_p1 = path.split('_')[:-2]
            mask_path = '_'.join(mask_p1) + '_mask.png'
            if os.path.exists(mask_path):
                mask_im = cv2.imread(mask_path)
                mask_im = cv2.resize(mask_im,(self.mask_size,self.mask_size),interpolation=cv2.INTER_NEAREST)[:,:,0].astype(int)
                if np.max(label[self.pos_idx:])<0.5: # 阴性
                    mask_im = self.get_empty_mask(mask_size) + 1

            else:
                mask_im = self.get_empty_mask(self.mask_size)
            

            mask_im = mask_im.astype(int)
            mask_im = mask_im-1
            
            all_mask.append(mask_im)
            if self.section=='train':
                sub_im = self.aug_seq(image=sub_im)['image']
            sub_im = cv2.resize(sub_im,(self.input_size,self.input_size))
            all_images.append(sub_im)
        
        
        for _ in range(self.bag_size-len(all_images)):
            all_images.append(self.get_empty_image())
            all_mask.append(self.get_empty_mask(mask_size))

        all_images = np.array(all_images)
        all_images = np.transpose(all_images,[0,3,1,2]) # bag,h,w,c  --> bag,c,h,w        
        all_mask = np.array(all_mask).astype(int)   # bag,h,w  

        all_mask[np.where(all_mask==3)] = -1
        all_mask[np.where(all_mask==4)] = 3

        return all_images, all_mask, label, filename, self.data_df.loc[indx,'slide_num_adj']

    def get_empty_image(self):
        im = np.zeros((self.input_size,self.input_size,3),dtype=np.int)
        return im  

    def get_empty_mask(self,size):
        return np.zeros((size,size),dtype=int)

    def create_aug_seq(self):
        seq = A.Compose([
            A.RandomShadow(p=0.2,shadow_roi=(0, 0, 1, 1)),
            A.OneOf([
                A.Downscale(p=0.5,scale_min=0.8,scale_max=0.9999),
                A.JpegCompression(p=0.5,quality_lower=70,quality_upper=95),
            ]),
            A.OneOf([
                A.GaussNoise(p=0.8), 
                A.ISONoise(p=0.8),    
                A.MultiplicativeNoise(p=0.8)
            ]),
            A.PixelDropout(p=0.3,),
            # A.Flip(p=0.7),
            A.OneOf([
                A.ColorJitter(brightness=0.2,hue=0.18,contrast=(0.7,1.8),saturation=0.4,p=0.9),
                A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40,  p=0.9),
                A.HueSaturationValue(p=0.9,hue_shift_limit=[-10,10],sat_shift_limit=[-45,25],val_shift_limit=20),
            ]),
            A.OneOf([
                A.RandomGamma(gamma_limit=(45,120),p=0.9),       
                A.RandomToneCurve(p=0.9,scale=0.5),
                A.Sharpen(p=0.5,alpha=(0.3, 0.7), lightness=(0.5, 1.0),),
                A.Emboss(p=0.6,),
                A.CLAHE(p=0.5,clip_limit=4.0),
            ]),
            A.Spatter(p=0.3,intensity=0.4),
            A.RandomBrightnessContrast(p=0.8),
        ])

        return seq


 
    

    def filter_not_exists(self,data_df):
        scanner_types = ['sdpc','tmap','mrxs','zyp','kfb'] if 'tmap' in data_df.columns else ['one_of']
        new_data_df = pd.DataFrame()
        for i,row in tqdm(data_df.iterrows()):
            slide_num_adj = row['slide_num_adj']   
            matched_20c = row['matched_20c']   
          
            if 'c' in data_df.columns:
                labels = [self.soft_label if row[a]==1 else 1-self.soft_label for a in self.label_mapping_rvs]
            else:            
                raise NotImplementedError           

            wsi_save_dir_list = []
            image_path_list = []
            filename_list = []
            for scn in scanner_types:
                if row[scn] =='not_exist' or (row[scn] =='bug') or (not isinstance(row[scn],str)):
                    continue
                image_path = row[scn]
                filename = os.path.basename(image_path)
                wsi_save_dir = f'{self.pkl_image_root}/{filename}'  

                if os.path.exists(wsi_save_dir+'/finished'):
                   
                    wsi_save_dir_list.append(wsi_save_dir)
                    image_path_list.append(image_path)
                    filename_list.append(filename)
                    
                        
            if len(filename_list)==0: continue
            if self.section=='train':
                data_dict = {
                    'slide_num_adj':slide_num_adj,
                    'labels':labels,
                    'filename_list':filename_list,
                    'wsi_save_dir_list':wsi_save_dir_list,
                    'image_path_list':image_path_list,
                    # 'oneof': oneof,
                }
                new_data_df = new_data_df.append(data_dict,ignore_index=True)
            else:
                for ii in range(len(wsi_save_dir_list)):
                    data_dict = {
                        'slide_num_adj':slide_num_adj,
                        'labels':labels,
                        'filename_list':filename_list[ii:ii+1],
                        'wsi_save_dir_list':wsi_save_dir_list[ii:ii+1],
                        'image_path_list':image_path_list[ii:ii+1],
                        # 'oneof': oneof,
                    }  
            

                    new_data_df = new_data_df.append(data_dict,ignore_index=True)

        print('--------')
        for k,v in self.upsample.items():
            new_data_df = pd.concat([new_data_df] + [new_data_df[new_data_df['labels'].apply(lambda x: x[self.label_mapping[k]]>0.5)]]*v)
            print(len(new_data_df),k,v)
        new_data_df = new_data_df.reset_index()
        return new_data_df