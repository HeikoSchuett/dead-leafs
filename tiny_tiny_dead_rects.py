#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:53:57 2019

@author: heiko
"""


import DeadLeaf as dl
import numpy as np

## first: exactly as in experiment
dl.save_training_data('/Users/heiko/tinytinydeadrects/training',100000,imSize=np.array((5,5)),distances=None,sizes=np.arange(1,6))
dl.save_training_data('/Users/heiko/tinytinydeadrects/validation',10000,imSize=np.array((5,5)),distances=None,sizes=np.arange(1,6))

## separate by exponent
for i in range(1,6):
    dl.save_training_data('/Users/heiko/tinytinydeadrects/training%d' % i,100000,imSize=np.array((5,5)),distances=None,sizes=np.arange(1,6),exponents=np.array([i]))
    dl.save_training_data('/Users/heiko/tinytinydeadrects/validation%d' % i,10000,imSize=np.array((5,5)),distances=None,sizes=np.arange(1,6),exponents=np.array([i]))
    