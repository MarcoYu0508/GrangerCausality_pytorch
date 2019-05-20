# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:17:39 2019

@author: marco
"""

import numpy as np
import scipy.io as sio
import torch
import ujson

from inspect import isfunction
from sklearn import preprocessing as skp
from torch.utils.data import DataLoader, Dataset


def get_json_data(file_name):
    return ujson.load(open(file_name, 'r'))

def get_mat_data(file_name, var_name):
    data_dict = sio.loadmat(file_name)
    return data_dict[var_name]

def get_txt_data(file_name, delimiter=',', dtype=np.float32):
    data = np.loadtxt(file_name, delimiter=delimiter, dtype=dtype)
    return data
 
def train_test_split1(data: np.ndarray, split: float = 0.7):
    split_point = int(np.ceil(data.shape[0] * split))
    return data[:split_point, :], data[split_point:, :]

def train_test_split2(inputs: np.ndarray, targets: np.ndarray, split: float = 0.7):
    split_point = int(np.ceil(inputs.shape[0] * split))
    return inputs[:split_point, :], inputs[split_point:, :], targets[:split_point, :], targets[split_point:, :]

def normalize(train_data, test_data, scaler_type: str= 'MinMaxScaler'):
    if scaler_type in ['MinMaxScaler', 'StandardScaler']:
        scaler = getattr(skp, scaler_type)().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    elif isfunction(scaler_type):
        train_data = scaler_type(train_data)
        test_data = scaler_type(test_data)
    else:
        raise ValueError("""An invalid option was supplied, options are ['MinMaxScaler', 'StandardScaler', None] or lambda function.""")
    return train_data, test_data   

def series2xy(series_data: np.ndarray, idx_x=None, idx_y=None, seq_length: int=20, num_shift: int=1):
    num_point, _ = series_data.shape
    inputs, targets = [], []
    
    for idx in range(0, num_point - seq_length - num_shift + 1, num_shift):
        if idx_x is None and idx_y is None:
            inputs.append(series_data[idx:(idx + seq_length), :])
            targets.append(series_data[idx + seq_length, :])
        elif idx_x is None and idx_y is not None:
            inputs.append(series_data[idx:(idx + seq_length), :])
            targets.append(series_data[idx + seq_length, idx_y])
        elif idx_y is None and idx_x is not None:
            inputs.append(series_data[idx:(idx + seq_length), idx_x])
            targets.append(series_data[idx + seq_length, :])
        else:
            inputs.append(series_data[idx:(idx + seq_length), idx_x])
            targets.append(series_data[idx + seq_length, idx_y])
    return np.array(inputs), np.array(targets)

class MakeSeqData(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        super(MakeSeqData, self).__init__()
        self.fill_dim = lambda a: a.unsqueeze_(1) if a.ndimension() == 1 else a
        self.data = self.fill_dim(torch.from_numpy(inputs))
        self.target = self.fill_dim(torch.from_numpy(targets))
        
    def get_tensor_data(self):
        return self.data, self.target
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return self.data.shape[0]
        
def make_loader(seq_data: np.ndarray, idx_x=None, idx_y=None, tt_split=0.7, tv_split=0.7, seq_len=20, num_shift=1, bt_sz=32, s_type='MinMaxScaler'):
    train_subseq, test_subseq = train_test_split1(seq_data, split=tt_split)
    train_subseq, test_subseq = normalize(train_subseq, test_subseq, scaler_type=s_type)
    X_train, y_train = series2xy(train_subseq, idx_x=idx_x, idx_y=idx_y, seq_length=seq_len, num_shift=num_shift)
    X_test, y_test = series2xy(test_subseq,  idx_x=idx_x, idx_y=idx_y, seq_length=seq_len, num_shift=num_shift)
    X_train, X_valid, y_train, y_valid = train_test_split2(X_train, y_train, split=tv_split)
    sub = [train_subseq, valid_subseq, test_subseq] = [MakeSeqData(x,y) for x,y in zip([X_train, X_valid, X_test], [y_train, y_valid, y_test])]
    [train_loader, valid_loader, test_loader] = [DataLoader(t, batch_size=bt_sz, shuffle=sf, drop_last=False, pin_memory=True) for t, sf in zip(sub, [False, False, False])]
    return train_loader, valid_loader, test_loader