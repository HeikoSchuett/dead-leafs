#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:46:47 2018

@author: heiko
"""


import numpy as np
import DeadLeaf as dl

distances = np.arange(0,100)

sizes = 5*np.array([np.arange(1,150,dtype='float'),np.arange(1,150,dtype='float')])
prob = (sizes[0]/np.min(sizes[0])) **-3
sizes = sizes.transpose()

p1 = [dl.calc_prob_one(sizes=sizes,prob=prob,dx=k,dy=0) for k in distances]