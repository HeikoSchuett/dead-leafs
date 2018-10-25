# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:03:12 2018

@author: heiko
"""

import numpy as np

def gen_rect_leaf(imSize = [255,255],sizes = [5,10,15],colors=[0,0.5,1],grid = 1,noise = 0,noiseType='norm',prob=None):
    if prob is None:
        prob = np.ones(len(sizes))
    assert (imSize[0] % grid) ==0,'Image Size not compatible with grid'
    assert (imSize[1] % grid) ==0,'Image Size not compatible with grid'
    assert np.all(np.array(sizes) % grid ==0),'Patch sizes not compatible with grid'
    assert noise>=0, 'noise is the standard deviation and thus should be >=0'
    assert np.all(prob>0), 'probabilities for shapes must be >0'
    assert len(prob) == len(sizes), 'probabilities and sizes should have equal length'
    prob = prob/np.sum(prob)
    probc = prob.cumsum()
    image = np.nan*np.zeros(imSize)
    while np.any(np.isnan(image)):
        if len(np.array(sizes).shape) == 1:
            idx_sizex = np.searchsorted(probc,np.random.rand())
            idx_sizey = np.searchsorted(probc,np.random.rand())
            sizx = sizes[idx_sizex]
            sizy = sizes[idx_sizey]
        elif len(np.array(sizes).shape) == 2:
            idx_size = np.searchsorted(probc,np.random.rand())
            sizx = sizes[idx_size][0]
            sizy = sizes[idx_size][1]
        idx_color = np.random.randint(len(colors))
        c = colors[idx_color]
        sizx = sizx/grid
        sizy = sizy/grid
        idx_x = np.random.randint(1-sizx,imSize[0]/grid)
        idx_y = np.random.randint(1-sizy,imSize[1]/grid)
        image[int(grid*max(idx_x,0)):int(grid*(idx_x+sizx)),int(grid*max(idx_y,0)):int(grid*(idx_y+sizy))] = c
    if noiseType=='norm':
        image = image+noise*np.random.randn(imSize[0],imSize[1])
    elif noiseType == 'uniform':
        image = image+noise*2*(np.random.rand(imSize[0],imSize[1])-.5)
    image[image<0] = 0
    image[image>1] = 1
    return image
    
    
    