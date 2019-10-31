#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:53:57 2019

@author: heiko
"""


import DeadLeaf as dl
import numpy as np


sizes = np.arange(2,6,dtype=np.float)
imSize = np.array((5,5))
exponents = np.arange(0,2)

## first: exactly as in experiment
point_p = np.load('p_same_5.npy')
point_p = np.exp(point_p)
dl.save_training_data('/Users/heiko/tinytinydeadrects/training',100000,imSize=imSize,distances=None,sizes=sizes,exponents=exponents,point_probabilities=point_p)
dl.save_training_data('/Users/heiko/tinytinydeadrects/validation',10000,imSize=imSize,distances=None,sizes=sizes,exponents=exponents,point_probabilities=point_p)

## separate by exponent
for i in range(1,6):
    dl.save_training_data('/Users/heiko/tinytinydeadrects/training%d' % i,100000,imSize=np.array((5,5)),distances=None,sizes=np.arange(1,6),exponents=np.array([i]))
    dl.save_training_data('/Users/heiko/tinytinydeadrects/validation%d' % i,10000,imSize=np.array((5,5)),distances=None,sizes=np.arange(1,6),exponents=np.array([i]))
    
