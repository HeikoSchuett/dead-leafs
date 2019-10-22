#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:19:03 2019

@author: heiko
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights_layer_conv(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

def init_weights_layer_linear(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.1*nn.init.calculate_gain('relu'))
        layer.bias.data.fill_(0)

class model_recurrent(nn.Module):
    def __init__(self,n_rep=20,n_neurons=100):
        super(model_recurrent, self).__init__()
        self.fc1 = nn.Linear(n_neurons+(3*5*5), n_neurons)
        self.fc2 = nn.Linear(n_neurons, 1)
        self.n_rep = n_rep
        self.n_neurons = n_neurons
        self.norm = nn.LayerNorm(n_neurons)
    def forward(self, x):
        x = x.view(-1,3*5*5)
        siz = list(x.shape)
        siz[1] = self.n_neurons
        h1 = torch.ones(siz,device=x.device)
        for i in range(self.n_rep):
            #inp1 = torch.cat((x,(h1-torch.mean(h1,dim=1).view(-1,1))/(eps+h1.std())),dim=1)
            inp1 = torch.cat((x,self.norm(h1)),dim=1)
            h1 = F.relu(self.fc1(inp1))
        x = self.fc2(h1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)
        
test_input = torch.Tensor(np.random.rand(200,3,5,5))

model = model_recurrent()

model(test_input)

model_cuda = model.to('cuda')
test_input_cuda = test_input.to('cuda')

model_cuda(test_input_cuda)