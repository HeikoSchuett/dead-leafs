#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:12:49 2019

@author: heiko
"""

import numpy as np
import torch

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    From: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def calc_prob_one(sizes = [5,10,15],grid=None,prob=None,dx = 1,dy = 1):
    sizes = np.array(sizes)
    if grid is not None:
        sizes = sizes/grid
        dx = dx/grid
        dy = dy/grid
    if prob is None:
        prob = np.ones(len(sizes))
    if len(sizes.shape) == 1:
        sizes = cartesian([sizes,sizes])
        prob = np.outer(prob,prob).flatten()
    p1 = sum(prob*np.array([max(0,(sizes[k,0]-dx))*max(0,(sizes[k,1]-dy)) for k in range(len(sizes))]))
    p2 = sum(prob*np.array([2*sizes[k,0]*sizes[k,1]-2*max(0,(sizes[k,0]-dx))*max(0,(sizes[k,1]-dy)) for k in range(len(sizes))]))
    p = p1/(p1+p2)
    return p
    #return (p,p1,p2)

def calc_prob_one_grid(sizes = [5,10,15],grid=None,prob = None,dx = 1,dy = 1):
    ps = np.zeros((len(dx),len(dy)))
    kx = 0
    for idx in dx:
        ky=0
        for idy in dy:
            ps[kx,ky] = calc_prob_one(sizes = sizes, grid = grid, prob = prob, dx = idx,dy = idy)
            ky += 1
        kx += 1
    return ps

def calc_distance_distribution(ps):
    p_diff = torch.zeros_like(ps)
    x = np.arange(ps.shape[0],dtype=np.int)
    y = np.arange(ps.shape[1],dtype=np.int)
    yy, xx = np.meshgrid(y,x)
    xx = xx.flatten()
    yy = yy.flatten()
    ps = ps.flatten()
    for i in range(len(ps)):
        xx_not_i = np.concatenate((xx[:i],xx[(i+1):]))
        yy_not_i = np.concatenate((yy[:i],yy[(i+1):]))
        ps_not_i = torch.cat((ps[:i],ps[(i+1):]))
        ps_not_i = ps_not_i/torch.sum(ps_not_i)
        x_diff = np.abs(xx_not_i - xx[i])
        y_diff = np.abs(yy_not_i - yy[i])
        for j in range(len(ps_not_i)):
            p_diff[x_diff[j],y_diff[j]] += ps[i]*ps_not_i[j]
    return p_diff



ps = torch.ones((5,5))/25
ps = torch.log(ps)
ps.requires_grad=True

sizes = np.arange(2,6)
prob = sizes ** 0
prob = prob/np.sum(prob)

p_same = torch.Tensor(calc_prob_one_grid(sizes = sizes, prob = prob, grid = None, dx = np.arange(ps.shape[0]), dy = np.arange(ps.shape[1])))

optimizer = torch.optim.SGD([ps], lr = 1, momentum = 0)

for iUpdate in range(50):
    p_diff = calc_distance_distribution(torch.exp(ps)/torch.sum(torch.exp(ps)))
    p_same_sum = torch.sum(p_diff*p_same)
    
    loss = 10*(torch.abs(p_same_sum-0.5)) + torch.sum(torch.exp(ps) * ps)
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    ps.data = ps - torch.logsumexp(ps.flatten(),dim=0)
    
    optimizer.param_groups[0]['lr'] = 0.9 * optimizer.param_groups[0]['lr']
