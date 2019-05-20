# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:29:11 2019

@author: marco
"""

import torch
from torch import nn

class RNN_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, ouput_dim, rnn_type='LSTN', num_layers=1, dropout=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        dropout = 0. if num_layers == 1 else dropout
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("""An invalid option was supplied, options are ['LSTM', 'GRU']""")
        self.fc = nn.Linear(hidden_dim, ouput_dim)
        
    def forward(self, x):
        hidden = self.initHidden(x.size(0))
        y, _ = self.rnn(x, hidden)
        return self.fc(y[:, -1, :])
    
    def initHidden(self, batchsize):
        weight = next(self.parameters())
        h0 = weight.new_zeros(self.num_layers, batchsize, self.hidden_dim).requires_grad_(False)
        if self.rnn_type == 'LSTM':
            return (h0,h0)
        else:
            return h0

    def repackage_hidden(self, hn):
        if isinstance(hn, torch.Tensor):
            return hn.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in hn)