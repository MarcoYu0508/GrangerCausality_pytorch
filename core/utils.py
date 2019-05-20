# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:18:17 2019

@author: marco
"""

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .loader import get_txt_data



class Timer():
    def start(self):
        self.start_dt = dt.datetime.now()
    
    def stop(self):
        end_dt = dt.datetime.now()
        print(f'Time taken: {(end_dt-self.start_dt).total_seconds():.2F}s')
        
def set_device():
    return torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)
                
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
def get_Granger_Causality(err_cond, err_all):
    if isinstance(err_cond, np.ndarray) and isinstance(err_all, np.ndarray):
        gc_matrix = np.double(err_cond).var(1) / np.double(err_all).var(0)
        gc_matrix = np.log(gc_matrix.clip(min=1.))
    elif isinstance(err_cond, torch.Tensor) and isinstance(err_all, torch.Tensor):
        gc_matrix = err_cond.double().var(1) / err_all.double().var(0)
        gc_matrix = gc_matrix.clamp(min=1.).log().cpu().numpy()
    else:
        raise ValueError('input variables should have the same type(numpy.ndarray or torch.tensor).')
    
    np.fill_diagonal(gc_matrix, 0.)
    return gc_matrix

def get_gc_precent(gc_matrix):
    deno = np.sum(gc_matrix, axis=0)
    deno[deno == np.zeros(1)] = np.nan
    gc_precent = gc_matrix / deno
    gc_precent[np.isnan(gc_precent)] = 0.
    return gc_precent

def plot_save_gc_precent(txt_path: str, save_png_path: str, png_title: str, save_txt_path: str):
    data = get_txt_data(txt_path, delimiter=' ')
    gc_precent = get_gc_precent(data)
    plt.matshow(gc_precent)
    plt.title(png_title)
    plt.savefig(save_png_path)
    np.savetxt(save_txt_path, gc_precent)
    
def matshow(data: np.ndarray, xlabel: str, ylabel: str, title: str, png_name: str):
    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap="YlGn")
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_yticks(np.arange(len(ylabel)))
    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar = ax.figure.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(title, rotation=-90, va="bottom")
    if data.shape[0] < 5:
        for i in range(len(xlabel)):
            for j in range(len(ylabel)):
                ax.text(j, i, round(data[i, j], 4) if not abs(data[i, j]) < 1e-8 else '', ha="center", va="center", color="k")
    fig.tight_layout()
    plt.savefig(png_name)
    plt.show()