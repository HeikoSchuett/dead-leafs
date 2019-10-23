#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:33:07 2019

@author: heiko
"""
import numpy as np
import pandas as pd
from skimage import io 
import os
import tqdm

import DeadLeaf as dl


nCols = 9
imSize=np.array((5,5))
distances=None
sizes=np.arange(1,6)

cols = (255*np.linspace(0,1,nCols)).astype(np.uint8)

# Analysis of the very first image
idx = 0
root_dir = '/Users/heiko/tinytinydeadrects/training'
solutions_df = pd.read_csv(os.path.join(root_dir,'solution.csv'),index_col=0)
img_name = os.path.join(root_dir,solutions_df['im_name'].iloc[idx])
test_image = io.imread(img_name).astype(np.float32)
test_image = np.array(test_image.transpose([2,0,1]))

g = dl.graph(test_image[2],sizes=sizes,colors=cols)

points = (test_image[0]==255) & (test_image[1]==0)
points = np.where(points)
g.get_exact_prob(points)
"""
nSamp = 5000

l = []

for iSamp in tqdm.trange(nSamp):
    samp = [0,None]
    while samp[1] is None:
        samp = g.get_decomposition_explained_bias(points,silent=True)
    l.append(samp)
    
estimate = [s[1] for s in l]
logPPos = [s[2] for s in l]
logPVis = [s[3] for s in l]
logPCorrection = [s[4] for s in l]

lik = np.array(logPPos)-np.array(logPVis)+np.array(logPCorrection)
lik = np.exp(lik-np.max(lik))
lik = lik/np.sum(lik)

prob = np.sum(lik*np.array(estimate))
"""