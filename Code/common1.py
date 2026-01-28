import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
import pandas as pd
from tool import EarlyStopping
from sklearn.metrics import roc_auc_score,mean_squared_error

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')


# save dict    
def save_obj(obj,name):
    with open(name + '.pkl','wb') as f:
        pickle.dump(obj,f)
                  
#load dict        
def load_obj(name):
    with open(name +'.pkl','rb') as f:
        return pickle.load(f)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def interp(v, q, num):
    f = interpolate.interp1d(v,q,kind='linear')
    v_new = np.linspace(v[0],v[-1],num)
    q_new = f(v_new)
    vq_new = np.concatenate((v_new.reshape(-1,1),q_new.reshape(-1,1)),axis=1)
    return q_new

def get_xl(name, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor):

    A = load_obj(f'./data/our_data/{name}')[name]
    A_rul = A['rul']
    A_dq = A['dq']
    A_df = A['data']
    
    all_idx = list(A_dq.keys())[9:]
    all_fea, rul_lbl, cap_lbl = [], [], []
    for cyc in all_idx:
        if cyc > 131:
            break
        tmp = A_df[cyc]
        tmp = tmp.loc[tmp['Status'].apply(lambda x: not 'discharge' in x)]
        
        left = (tmp['Current (mA)']<5000).argmax() + 1
        right = (tmp['Current (mA)']<1090).argmax() - 2
    
        tmp = tmp.iloc[left:right]
    
        tmp_v = tmp['Voltage (V)'].values
        tmp_q = tmp['Capacity (mAh)'].values
        tmp_t = tmp['Time (s)'].values
        v_fea = interp(tmp_t, tmp_v, fea_num)
        q_fea = interp(tmp_t, tmp_q, fea_num)
    
        tmp_fea = np.hstack((v_fea.reshape(-1,1), q_fea.reshape(-1,1)))
        
        all_fea.append(np.expand_dims(tmp_fea,axis=0))
    
    for cyc in all_idx:    
        rul_lbl.append(A_rul[cyc])
        cap_lbl.append(A_dq[cyc])   
        
    all_fea = np.vstack(all_fea)
    rul_lbl = np.array(rul_lbl)
    cap_lbl = np.array(cap_lbl)
    
    cap_lbl_c = cap_lbl.copy()
     
    
    all_fea_c = all_fea.copy()
    all_fea_c[:,:,0] = (all_fea_c[:,:,0]-v_low)/(v_upp-v_low)
    all_fea_c[:,:,1] = (all_fea_c[:,:,1]-q_low)/(q_upp-q_low)
    dif_fea = all_fea_c - all_fea_c[0:1,:,:]
    all_fea = np.concatenate((all_fea,dif_fea),axis=2)
    
    all_fea = all_fea[:100]
    cap_lbl = cap_lbl[:100]
    rul_lbl = np.array([rul_lbl[100]])
    
    
    
    
    all_fea_new = np.zeros(all_fea.shape)
    all_fea_new[:,:,0] = (all_fea[:,:,0]-v_low)/(v_upp-v_low)
    all_fea_new[:,:,1] = (all_fea[:,:,1]-q_low)/(q_upp-q_low)
    all_fea_new[:,:,2] = all_fea[:,:,2]
    all_fea_new[:,:,3] = all_fea[:,:,3]
    
    print(f'{name} length is {all_fea_new.shape[0]}', 
          'v_max:', '%.4f'%all_fea_new[:,:,0].max(),
          'q_max:', '%.4f'%all_fea_new[:,:,1].max(),
          'dv_max:', '%.4f'%all_fea_new[:,:,2].max(), 
          'dq_max:', '%.4f'%all_fea_new[:,:,3].max())
    
    rul_lbl = rul_lbl / rul_factor
    cap_lbl = cap_lbl / cap_factor
    

    

    return all_fea_new, cap_lbl, rul_lbl.reshape(-1,1), cap_lbl_c/cap_factor

