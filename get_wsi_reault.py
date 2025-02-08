# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch_optimizer as optim
from torch.nn.modules.loss import _Loss

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import timm
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt_path', help='ckpt_path', required=False, type=str,default='./ckpt/example.pth')
    args = parser.parse_args()
    return args

args = parse_args()

device = torch.device("cpu")

batch_size = 1
label_mapping = {'nag':0,'p':1,'ag':2,'lgd':3,'hgd':4,'c':5}

label_mapping_rvs = ['nag','p','ag','lgd','hgd','c'] 

pkl_root = './data/pkl'


ckpt_path = args.ckpt_path


save_csv_path = './result_test_tmp.csv'

val_df = pd.read_csv('./data/datases/sample_data.csv')

thresh = 0.39 

one_of = True

def process_csv(data_df):
    if isinstance(data_df,list):
        new_data_df = pd.DataFrame()
        for i in data_df:
            pkl_path = f'{pkl_root}/{i}_dir/{i}.ghdmi'
            if os.path.exists(pkl_path):           
                data_dict = {
                    'slide_num_adj':i,
                    'labels':[1 for _ in label_mapping_rvs],
                    'filename':i,
                    'pkl_path':pkl_path,
                    # 'oneof': oneof,
                }         
                new_data_df = new_data_df.append(data_dict,ignore_index=True)     
    elif 'tmap' not in data_df.columns:
        new_data_df = pd.DataFrame()
        for i,row in tqdm(data_df.iterrows()):
            filename = row['filename']
            pkl_path = f'{pkl_root}/{filename}_dir/{filename}.ghdmi'
            if os.path.exists(pkl_path):           
                data_dict = {
                    'slide_num_adj':filename,
                    'labels':[1 for _ in label_mapping_rvs],
                    'filename':filename,
                    'pkl_path':pkl_path,
                    # 'oneof': oneof,
                }           
                new_data_df = new_data_df.append(data_dict,ignore_index=True)             
    else:
        scanner_types = ['mrxs','sdpc','tmap','kfb','zyp'] if 'tmap' in data_df.columns else ['one_of']

        new_data_df = pd.DataFrame()
        for i,row in tqdm(data_df.iterrows()):
            slide_num_adj = row['slide_num_adj']  if 'slide_num_adj' in row.keys() else row['filename'] 

            matched_20c = row['matched_20c'] 


            if 'lgd' in data_df.columns:
                labels = [1 if row[a]==1 else 0 for a in label_mapping_rvs]
            else:            
                labels = [1 for _ in label_mapping_rvs]          
           
            for scn in scanner_types:
                if not isinstance(row[scn],str):
                    continue

                image_path = row[scn]
                filename = os.path.basename(image_path)
                pkl_path = f'{pkl_root}/{filename}_dir/{filename}.ghdmi'
                print(pkl_path,os.path.exists(pkl_path))

                if os.path.exists(pkl_path):           
                    data_dict = {
                        'slide_num_adj':slide_num_adj,
                        'labels':labels,
                        'filename':filename,
                        'pkl_path':pkl_path,
                        'matched_20c': matched_20c,
                    }            
                    new_data_df = new_data_df.append(data_dict,ignore_index=True)           
                if one_of:
                    break

    new_data_df = new_data_df.reset_index()
    return new_data_df


class GHDModel(nn.Module):   
    def __init__(self,):
        super(GHDModel, self).__init__()

        self.L = 2048 # 2048 # 1408 # 512 # 2048 #2048 # 256  # num_output_feature of the encoder_2
        self.D = 224 # 256
        self.K = 1

        self.C = len(label_mapping_rvs)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.C)

        self.classifier_linears = nn.ModuleList([nn.Linear(self.L*self.K, 1) for _ in range(self.C)])

       
    def weight_average(self, x, w,num_instances):
        w = w.view(-1,1)
        M = torch.multiply(x,w)
        M = M.view(-1, num_instances, self.L)  # bs x N x L
        M = torch.sum(M,dim=1)  # bs x L

        return M


    def forward(self, x):
        # x: bs x N x C
        bs, num_instances, ch = x.shape

        x = x.view(bs*num_instances, ch) # x: N bs x C x W x W

        A_V = self.attention_V(x)  # N bs x self.D
        A_U = self.attention_U(x)  # N bs x self.D

        A = self.attention_weights(A_V * A_U) # element wise multiplication # N bs x self.C

        A = A.view(bs, num_instances, self.C)  # bs x N x  C
        A = F.softmax(A, dim=1).view(-1,self.C) 

        x = [self.weight_average(x,i,num_instances) for i in torch.split(A, split_size_or_sections=1, dim=1)]
        x = [self.classifier_linears[i](x[i]) for i in range(self.C)]  # [] bs x 1
        Y_prob = torch.cat(x,1)

        return Y_prob
 
    # def forward(self, x):
    #     # x: bs x N x C
    #     bs, num_instances, ch = x.shape

    #     x = x.view(bs*num_instances, ch) # x: N bs x C x W x W

    #     A_V = self.attention_V(x)  # N bs x self.D
    #     A_U = self.attention_U(x)  # N bs x self.D

    #     A = self.attention_weights(A_V * A_U) # element wise multiplication # N bs x self.C

    #     A = A.view(bs, num_instances, self.C)  # bs x N x  C
    #     if num_instances>8:
    #         mul_a = []
    #         for ii in range(self.C):
    #             cond_value = torch.sort(A[0,:,ii],descending=True).values[8]
    #             mul_a.append(torch.where(A[0,:,ii]>cond_value,0.,-float('inf')))
    #         A = A + torch.stack(mul_a).T.unsqueeze(0)
    #     A = F.softmax(A, dim=1).view(-1,self.C) 

    #     x = [self.weight_average(x,i,num_instances) for i in torch.split(A, split_size_or_sections=1, dim=1)]
    #     x = [self.classifier_linears[i](x[i]) for i in range(self.C)]  # [] bs x 1
    #     Y_prob = torch.cat(x,1)

    #     return Y_prob
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def load_model(weight_path=None,strict=False):
    print('loading model')
    model = GHDModel()
    # load pretrain
    if weight_path is not None:
        print ('loading pretrained model {}'.format(weight_path))
        stdict = torch.load(weight_path,map_location=device)#['state_dict']
        new_stdict = {}
        prefix = ''
        for i,(k,v) in enumerate(stdict.items()):
            if 'encoder' in k or ('seg' in k): continue
            if 'module.' in k:
                new_stdict[prefix + k[7:]] = stdict[k]
            else:
                new_stdict[prefix + k] = stdict[k]
        missing_keys = [i for i in model.state_dict().keys() if i not in new_stdict.keys()]
        print('missing_keys',missing_keys)
        model.load_state_dict(
            new_stdict, strict=strict
        )
    
    model = model.to(device)
    model = nn.DataParallel(model)
    # print(model)
    return model

def get_final_label(x,threshs=0.5):  
    if threshs==None:
        x1 = np.round(x)
    elif isinstance(threshs,float):
        x1 = (x>threshs).astype(np.int)
    else:
        x1 = np.zeros_like(x)
        for i,th in enumerate(threshs):
            x1[:,i] = [1 if p>th else 0 for p in x[:,i]]
    for r in range(len(x1)):
        if np.max(x1[r])==0:
            # x1[r][np.argmax(x[r])] = 1
            x1[r,0] = 1
    return x1 

def get_h_label(x):
    x1 = np.round(x)
    if max(x1)==0:
        return np.argmax(x)
    else:
        for i in list(range(len(x1)))[::-1]:
            if x1[i]==1:
                return i




val_df = process_csv(val_df)
print(len(val_df))

pos_c_id = 4
pos_id = 3


model = load_model(weight_path=ckpt_path,strict=True)
model = model.eval()

all_y_true = []
all_y_true_onehot = []
all_y_pred = []
all_pred = []

for i,row in val_df.iterrows():
    pkl_path = row['pkl_path']
    with open(pkl_path,'rb') as f:
        pkl_content = pickle.load(f)
    feature = pkl_content['feature_map']
    feature = torch.Tensor(feature).squeeze(1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(feature)
        pred = sigmoid(pred.cpu().detach().numpy())

        pred_1 = 1-pred
        pred_1 = pred_1/(1-thresh)*0.5
        pred = 1 - pred_1

    all_pred.append(pred[0])
    pred_label = get_final_label(pred)
    

    all_y_pred.append(pred_label[0])
    all_y_true.append(row['labels'])
 
    all_y_true_onehot.append(row['labels'])


    if i%300 == 0 :
        highest_cls_pred = [get_h_label(i) for i in all_y_pred]
        highest_cls_true = [get_h_label(i) for i in all_y_true]

        cm = confusion_matrix(highest_cls_true,highest_cls_pred,labels=np.arange(0,len(label_mapping_rvs)))
        sen_c = np.sum(cm[pos_c_id:,pos_id:])/np.sum(cm[pos_c_id:])
        sen_pos = np.sum(cm[pos_id:,pos_id:])/np.sum(cm[pos_id:])
        sp = np.sum(cm[:pos_id,:pos_id])/np.sum(cm[:pos_id])
        print(f'sp {sp}\nsen_pos {sen_pos}\nsen_c {sen_c}\n')
        print(cm)
        print(np.sum(cm))
        print(classification_report(highest_cls_true,highest_cls_pred))

        c_e = coverage_error(all_y_true_onehot, all_pred)
        lraps = label_ranking_average_precision_score(all_y_true_onehot, all_pred)
        print(f'coverage_error: {c_e}')
        print(f'label_ranking_average_precision_score: {lraps}')



c_e = coverage_error(all_y_true_onehot, all_pred)
lraps = label_ranking_average_precision_score(all_y_true_onehot, all_pred)
print(f'coverage_error: {c_e}')
print(f'label_ranking_average_precision_score: {lraps}')

highest_cls_pred = [get_h_label(i) for i in all_y_pred]
highest_cls_true = [get_h_label(i) for i in all_y_true]

cm = confusion_matrix(highest_cls_true,highest_cls_pred,labels=np.arange(0,len(label_mapping_rvs)))
sen_c = np.sum(cm[pos_c_id:,pos_id:])/np.sum(cm[pos_c_id:])
sen_pos = np.sum(cm[pos_id:,pos_id:])/np.sum(cm[pos_id:])
sp = np.sum(cm[:pos_id,:pos_id])/np.sum(cm[:pos_id])
print(f'sp {sp}\nsen_pos {sen_pos}\nsen_c {sen_c}\n')
print(cm)
print(np.sum(cm))

print(classification_report(highest_cls_true,highest_cls_pred,labels=np.arange(0,len(label_mapping_rvs)),target_names=label_mapping_rvs,digits=4))
all_y_pred = np.array(all_y_pred)
all_y_true = np.array(all_y_true)

result_df = pd.DataFrame(all_pred)
result_df.columns = [i+'_prob' for i in label_mapping_rvs]
for i in range(all_y_pred.shape[1]):
    result_df[label_mapping_rvs[i] +'_pred'] = all_y_pred[:,i]
for i in range(all_y_true.shape[1]):
    result_df[label_mapping_rvs[i]+'_true'] = all_y_true[:,i]
result_df['filename'] = val_df['filename']
result_df['pred_label'] = [label_mapping_rvs[a] for a in highest_cls_pred]
result_df['gt'] = [label_mapping_rvs[a] for a in highest_cls_true]
try:
    result_df['slide_num_adj'] = val_df['slide_num_adj']
    result_df['matched_20c'] = val_df['matched_20c']
except Exception as e:
    print(e)
result_df['miss_highest_prob'] = [max([all_pred[i][ii] for ii in range(len(label_mapping_rvs)) if all_y_true[i][ii]==0]+[-1]) for i in range(len(all_y_true))]
result_df['mul_pred'] = ['_'.join([label_mapping_rvs[j] for j in range(len(i)) if i[j]==1]) for i in all_y_pred]
result_df['mul_gt'] = ['_'.join([label_mapping_rvs[j] for j in range(len(i)) if i[j]==1]) for i in all_y_true]

result_df.to_csv(save_csv_path,index=False)


