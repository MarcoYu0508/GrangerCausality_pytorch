# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:51:52 2019

@author: marco
"""

import torch 
from tensorboardX import SummaryWriter
from torch import nn

class Modeler:
    def __init__(self, network, opt, criterion, device, visualization=False):
        self.model = nn.DataParallel(network).to(device) if torch.cuda.device_count() > 1 else network.to(device)
        self.opt = opt
        self.criterion = criterion
        self.dev = device
        self.tsfm = lambda a: a.to(self.dev).float()
        self.visualization = visualization
        if self.visualization:
            self.write = SummaryWriter('log')
            
    def __del__(self):
        if self.visualization:
            self.write.close()
            
    def train_model(self, loaders, epoch=None):
        self.model.train()
        
        for data, target in loaders:
            data, target = self.tsfm(data), self.tsfm(target)
            
            out = self.model(data)
            loss = self.criterion(out, target)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
        if epoch is not None:
            self.write.add_scalar('train loss', loss.item(), epoch)
        return loss.item()
    
    @torch.no_grad()
    def evaluate_model(self, loaders, epoch=None):
        self.model.eval()
        
        for data, target in loaders:
            data, target = self.tsfm(data), self.tsfm(target)
            out = self.model(data)
            loss = self.criterion(out, target)
            
        if epoch is not None:
            self.write.add_scalar('evaluate loss', loss.item(), epoch)
        return loss.item()
    
    @torch.no_grad()
    def predict(self, x, y):
        x, y =  self.tsfm(x), self.tsfm(y)
        out = self.model(x).detach()
        return out, out -y
    
    def save_trained_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))