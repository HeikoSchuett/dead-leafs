#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:32:34 2019

@author: heiko
"""

import DeadLeaf as dl
import numpy as np
import tqdm

nCols = 5
imSize= [3,3]
sizes = [1,3]
exponent = 1
distance = 1
angle = 0
abs_angle = 0

test_image = dl.generate_image(exponent,0,sizes,distance,angle,abs_angle,imSize=np.array(imSize),num_colors=nCols,mark_points=False)
points = test_image[2]
same_rect = test_image[3]

g = dl.graph(test_image[0],sizes=sizes,colors=np.linspace(0,1,nCols))


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

# The absolute minimal image imSize = [1,2]
# it works!
nCols = 4
sizes = [1,2]
minimal_image = np.array([[0.,0.]])
points = [[0,0],[0,1]]
g = dl.graph(minimal_image,sizes=sizes,colors=np.linspace(0,1,nCols))


nSamp = 10000

l = []

for iSamp in tqdm.trange(nSamp):
    samp = [0,None]
    while samp[1] is None:
        samp = g.get_decomposition(points,silent=True)
    l.append(samp)
    
estimate = [s[1] for s in l]
logPPos = [s[2] for s in l]
logPVis = [s[3] for s in l]
logPCorrection = [s[4] for s in l]

lik = np.array(logPPos)-np.array(logPVis)
lik = np.exp(lik-np.max(lik))
lik = lik/np.sum(lik)

prob = np.sum(lik*np.array(estimate))



# The absolute minimal image imSize = [1,2]
# biased sampling works!

import DeadLeaf as dl
import numpy as np
import tqdm
nCols = 4
sizes = [1,2]
minimal_image = np.array([[0.,0.]])
points = [[0,0],[0,1]]
g = dl.graph(minimal_image,sizes=sizes,colors=np.linspace(0,1,nCols))


nSamp = 1000

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


# test that [1,3] gives same result
# it does not, but it should not either?

import DeadLeaf as dl
import numpy as np
import tqdm
nCols = 4
sizes = [1,2]
minimal_image = np.array([[0.,0.,0.]])
points = [[0,0],[0,1]]
g = dl.graph(minimal_image,sizes=sizes,colors=np.linspace(0,1,nCols))


nSamp = 1000

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