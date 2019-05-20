# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:55:08 2019

@author: marco
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torch

from torch import nn, optim
from core import (Timer, get_Granger_Causality, get_json_data, get_mat_data, make_loader, matshow, set_device)
from Models import Modeler, RNN_Net

def train_valid(in_dim, hidden_dim, out_dim, ckpt, test_data, loaders):
    
    net = RNN_Net(in_dim, hidden_dim, out_dim, rnn_type=cfg['rnn_type'], 
                  num_layers=cfg['num_layers'], dropout=cfg['dropout'])
    opt = optim.RMSprop(net.parameters(), lr=cfg['lr_rate'],
                        momentum=cfg['momentum'],
                        weight_decay=cfg['weight_decay'])
    lr_decay2 = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    criterion = nn.MSELoss()
    model = Modeler(net, opt, criterion, device)
    
    t_loss = []
    v_loss = []
    
    for epoch in range(cfg['num_epoch']):
        train_loss = model.train_model(loaders['train'])
        t_loss.append(train_loss)
        valid_loss = model.evaluate_model(loaders['valid'])
        v_loss.append(valid_loss)
        lr_decay2.step(valid_loss)
        print(f"[{epoch+1}/{cfg['num_epoch']}] ===>> train_loss: {train_loss: .4f} | valid_loss: {valid_loss: .4f}")
     
    # 畫loss的趨勢圖
    t_loss = np.array(t_loss)
    v_loss = np.array(v_loss)
    x = np.linspace(1,cfg['num_epoch'],cfg['num_epoch'])
    plt.figure(dpi=200)
    plt.plot(x, t_loss, color = "blue")
    plt.plot(x, v_loss, color = "orange")
    plt.legend(["train_loss","valid_loss"])
    plt.grid()
    plt.savefig(ckpt+'.png')
    plt.close()    
        
    model.save_trained_model(ckpt+'.pth')
    model.load_model(ckpt+'.pth')
    prediction, err = model.predict(*test_data)
    
    #預測結果視覺化
    for ch in range(prediction.shape[-1]):
        plt.figure(dpi=200)
        plt.plot(np.c_[test_data[1].cpu().numpy()[:,ch]], color = "blue")
        plt.plot(np.c_[prediction.cpu().numpy()[:,ch]], color = "red")
        plt.ylim([0,1])
        plt.legend([f'label channel{ch+1}',f'prediction channel{ch+1}'])
        plt.grid()
        plt.savefig(f'images/{signal_type}/prediction/{signal_type}{(i+1)*10}_err{ch}'+time.strftime('%Y_%m_%d_%H_%M_%S')+'.png')
        plt.close()
        plt.figure(dpi=200)
        plt.plot(np.c_[err.cpu().numpy()[:,ch]], '.', color = "green")
        plt.ylim([-1,1])
        plt.legend([f'err{ch+1}, var={err[:,ch].double().cpu().numpy().var(): .4f}'])
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'images/{signal_type}/error/{signal_type}{(i+1)*10}_err{ch}'+time.strftime('%Y_%m_%d_%H_%M_%S')+'.png')
        plt.close()
    
    return err

def main(s_number):
    if signal_type in all_signal_type[:3]:
        seq_data_all = get_mat_data(f'Data/{signal_type}_noise1.mat', f'{signal_type}')
    else:
        seq_data_all = get_mat_data(f'Data/{signal_type}/{signal_type}{s_number}.mat', f'{signal_type}')
        
    model_id = 1
    print(f'model_id: {model_id}')
    train_loader, valid_loader, test_loader = make_loader(
            seq_data_all, tt_split=cfg['tt_split'], tv_split=cfg['tv_split'],
            seq_len=cfg['seq_len'], bt_sz=cfg['bt_sz'])
    loaders = {'train':train_loader, 'valid': valid_loader}
    err_all = train_valid(cfg['in_dim'], cfg['num_hidden'], cfg['out_dim'],
                          f'checkpoints/{signal_type}/{signal_type}{s_number}_model_weights',
                          test_loader.dataset.get_tensor_data(), loaders)
    
    temp = []
    for ch in range(cfg['num_channel']):
        model_id += 1
        print(f'model_id: {model_id}')
        idx = list(set(range(cfg['num_channel'])) - {ch})  
        seq_data = seq_data_all[:, idx]
        train_loader, valid_loader, test_loader = make_loader(
            seq_data, tt_split=cfg['tt_split'], tv_split=cfg['tv_split'],
            seq_len=cfg['seq_len'], bt_sz=cfg['bt_sz'])
        loaders = {'train':train_loader, 'valid': valid_loader}
        err = train_valid(cfg['in_dim']-1, cfg['num_hidden'], cfg['out_dim']-1,
                          f'checkpoints/{signal_type}/{signal_type}{s_number}_model_weights',
                          test_loader.dataset.get_tensor_data(), loaders) 
        temp += [err]
    temp = torch.stack(temp)
    
    err_cond = temp.new_zeros(temp.size(0), temp.size(1), cfg['num_channel'])
    for idx in range(cfg['num_channel']):
        col = list(set(range(cfg['num_channel'])) - {idx})
        err_cond[idx, :, col] = temp[idx]
    return get_Granger_Causality(err_cond, err_all)
        
flag = 1

if __name__ == '__main__':
    
    for i in range(720,721):
        
        timer = Timer()
        timer.start()
        config = get_json_data('configs/cfg.json')
        device = set_device()
        all_signal_type = ['linear_signals', 'nonlinear_signals', 'longlag_nonlinear_signals', 'IMS']
        
        avg_gc_matrix = 0
        
        signal_type = all_signal_type[3]
        print(f'signal type: {signal_type}')
        cfg = config[signal_type]
        for _ in range(cfg['num_trial']):
            avg_gc_matrix += main((i+1)*10)
            
        label = ['ch' + str(t + 1) for t in range(cfg['num_channel'])]
        
        if flag == 1:
             matshow(avg_gc_matrix, label, label, f'{signal_type} Granger Causality Matrix', f'images/{signal_type}/result/{signal_type}{(i+1)*10}_Granger_Matrix'+time.strftime('%Y_%m_%d_%H_%M_%S')+'.png')
             np.savetxt(f'checkpoints/{signal_type}/{signal_type}{(i+1)*10}_granger_matrix.txt', avg_gc_matrix)
        else:
             matshow(avg_gc_matrix, label, label, f'{signal_type} Granger Causality Matrix', f'images/{signal_type}/result/{signal_type}_Granger_Matrix'+time.strftime('%Y_%m_%d_%H_%M_%S')+'.png')
             np.savetxt(f'checkpoints/{signal_type}/{signal_type}_granger_matrix.txt', avg_gc_matrix)

        # 計時结束
        timer.stop() 