# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:03:12 2018

@author: heiko
"""

import numpy as np
import scipy.signal as signal

def gen_rect_leaf(imSize = [255,255],sizes = [5,10,15],colors=[0,0.5,1],grid = 1,noise = 0,noiseType='norm',prob=None,fixedC=0,fixedIdx=[]):
    if prob is None:
        prob = np.ones(len(sizes))
    assert (imSize[0] % grid) ==0,'Image Size not compatible with grid'
    assert (imSize[1] % grid) ==0,'Image Size not compatible with grid'
    assert np.all(np.array(sizes) % grid ==0),'Patch sizes not compatible with grid'
    assert noise>=0, 'noise is the standard deviation and thus should be >=0'
    assert np.all(prob>0), 'probabilities for shapes must be >0'
    assert len(prob) == len(sizes), 'probabilities and sizes should have equal length'
    fixedIdx = np.array(fixedIdx)
    sizes = np.array(sizes)
    # correction for the different size of the possible area
    if len(np.array(sizes).shape) == 1:
        probx = prob * (sizes+imSize[0])/(np.max(sizes)+imSize[0])
        proby = prob * (sizes+imSize[1])/(np.max(sizes)+imSize[1])
        probx = probx/np.sum(probx)
        proby = proby/np.sum(proby)
        probcx = probx.cumsum()
        probcy = proby.cumsum()
    else:
        prob = prob * (sizes[:,0]+imSize[0])/(np.max(sizes[:,0])+imSize[0])
        prob = prob * (sizes[:,1]+imSize[1])/(np.max(sizes[:,1])+imSize[1])
        prob = prob/np.sum(prob)
        probc = prob.cumsum()
    image = np.nan*np.zeros(imSize,dtype='float')
    rectList = list()
    while np.any(np.isnan(image)):
        if len(np.array(sizes).shape) == 1:
            idx_sizex = np.searchsorted(probcx,np.random.rand())
            idx_sizey = np.searchsorted(probcy,np.random.rand())
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
        rectList.append([grid*idx_x,grid*idx_y,grid*sizx,grid*sizy,idx_color])
        image[int(grid*max(idx_x,0)):int(grid*max(0,idx_x+sizx)),int(grid*max(idx_y,0)):int(grid*max(0,idx_y+sizy))] = c
    rectList=np.array(rectList,dtype=np.int16)
    oneObject = False
    # if we want to fix some point in the image to a color
    # find last rectangle put in these places and redraw all rectangles up to this point
    idxStart = -1
    while len(fixedIdx)>0:
        idxStart = idxStart+1
        R = rectList[idxStart]
        delete = []
        for i in range(len(fixedIdx)):
            #print((R[0]<=fixedIdx[i][0]),((R[0]+R[2])>fixedIdx[i][0]),(R[1]<=fixedIdx[i][1]),((R[1]+R[3])>fixedIdx[i][1]))
            if (R[0]<=fixedIdx[i][0]) and ((R[0]+R[2])>fixedIdx[i][0]) and (R[1]<=fixedIdx[i][1]) and ((R[1]+R[3])>fixedIdx[i][1]):
                rectList[idxStart,-1]=fixedC
                delete.append(i)
                #print('found one')
        if len(delete)>1:
            oneObject = True
        fixedIdx = np.delete(fixedIdx,delete,axis=0)
    for i in range(len(rectList)):
        image[int(max(rectList[len(rectList)-i-1,0],0)):int(max(0,rectList[len(rectList)-i-1,0]+rectList[len(rectList)-i-1,2])),
              int(max(rectList[len(rectList)-i-1,1],0)):int(max(0,rectList[len(rectList)-i-1,1]+rectList[len(rectList)-i-1,3]))] = colors[rectList[len(rectList)-i-1,-1]]
    if noiseType=='norm':
        image = image+noise*np.random.randn(imSize[0],imSize[1])
    elif noiseType == 'uniform':
        image = image+noise*2*(np.random.rand(imSize[0],imSize[1])-.5)
    image[image<0] = 0
    image[image>1] = 1
    return (image,rectList,oneObject)
    

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

   
            
class dlMovie:
    def __init__(self,imSize = [255,255],sizes = [5,10,15],colors=[0,0.5,1],grid = 1,noise = 0,noiseType='norm',prob=None):    
        if prob is None:
            self.prob = np.ones(len(sizes))
        else:
            self.prob = prob
        assert (imSize[0] % grid) ==0,'Image Size not compatible with grid'
        assert (imSize[1] % grid) ==0,'Image Size not compatible with grid'
        assert np.all(np.array(sizes) % grid ==0),'Patch sizes not compatible with grid'
        assert noise>=0, 'noise is the standard deviation and thus should be >=0'
        assert np.all(self.prob>0), 'probabilities for shapes must be >0'
        assert len(self.prob) == len(sizes), 'probabilities and sizes should have equal length'
        self.imSize = imSize
        self.sizes = np.array(sizes)
        self.colors = colors
        self.grid = grid
        self.noise = noise
        self.noiseType = noiseType
        self.image = np.zeros(imSize,dtype='float')
        self.rectList=np.zeros((0,5),dtype=np.int16)
        # correction for the different size of the possible area
        if len(np.array(sizes).shape) == 1:
            probx = self.prob * (self.sizes+imSize[0])/(np.max(self.sizes)+imSize[0])
            proby = self.prob * (self.sizes+imSize[1])/(np.max(self.sizes)+imSize[1])
            probx = probx/np.sum(probx)
            proby = proby/np.sum(proby)
            self.probcx = probx.cumsum()
            self.probcy = proby.cumsum()
        else:
            prob = prob * (sizes[:,0]+imSize[0])/(np.max(sizes[:,0])+imSize[0])
            prob = prob * (sizes[:,1]+imSize[1])/(np.max(sizes[:,1])+imSize[1])
            prob = prob/np.sum(prob)
            self.probc = prob.cumsum()
    def add_leaf(self):
        if len(self.sizes.shape) == 1:
            idx_sizex = np.searchsorted(self.probcx,np.random.rand())
            idx_sizey = np.searchsorted(self.probcy,np.random.rand())
            sizx = self.sizes[idx_sizex]
            sizy = self.sizes[idx_sizey]
        elif len(self.sizes.shape) == 2:
            idx_size = np.searchsorted(self.probc,np.random.rand())
            sizx = self.sizes[idx_size][0]
            sizy = self.sizes[idx_size][1]
        idx_color = np.random.randint(len(self.colors))
        c = self.colors[idx_color]
        sizx = sizx/self.grid
        sizy = sizy/self.grid
        idx_x = np.random.randint(1-sizx,self.imSize[0]/self.grid)
        idx_y = np.random.randint(1-sizy,self.imSize[1]/self.grid)
        self.rectList=np.append(self.rectList,[[self.grid*idx_x,self.grid*idx_y,self.grid*sizx,self.grid*sizy,idx_color]], axis=0)
        self.image[int(self.grid*max(idx_x,0)):int(self.grid*max(0,idx_x+sizx)),int(self.grid*max(idx_y,0)):int(self.grid*max(0,idx_y+sizy))] = c
    def get_image(self):
        return self.image

class node:
    def __init__(self,rectList=None):
        self.children = None
        self.probChild = None
        if rectList is None:
            self.rectList=np.zeros((0,5),dtype=np.int16)
        else:
            self.rectList = rectList

    def add_children(self,image,sizes,colors,prob):
        self.children = list()
        self.probChild = list()
        kP = 0
        for iSize in sizes:
            rect = np.ones(iSize)
            for iC in colors:
                im = (image-iC)**2
                im[np.isnan(im)]=0
                imTest = signal.convolve2d(im,rect,'full')
                im2 = ~np.isnan(image)
                imTest2 = signal.convolve2d(im2,rect,'full')
                locations = np.where(np.logical_and(imTest==0,imTest2!=0))
                for t in np.array(locations).T:
                    self.children.append(node(
                            np.concatenate((self.rectList,[[t[0]-iSize[0]+1,t[1]-iSize[1]+1,iSize[0],iSize[1],iC]]),0)))
                    self.probChild.append(prob[kP])
            kP = kP+1
    def get_sample_child(self,image,sizes,colors,prob):
        # NOTE: This changes the image although it is not returned!
        if self.children is None:
            self.add_children(image,sizes,colors,prob)
        pc = np.cumsum(self.probChild)
        pc = pc/pc[-1]
        ran = np.random.rand()
        idx = np.argmax(ran<pc)
        child = self.children[idx]
        idx_x = child.rectList[-1,0]
        idx_y = child.rectList[-1,1]
        sizx = child.rectList[-1,2]
        sizy = child.rectList[-1,3]
        image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(max(idx_y,0)):int(max(0,idx_y+sizy))] = np.nan
        return child
        
class graph: 
    def __init__(self,image,sizes,colors,prob):
        self.image = np.array(image)
        if prob is None:
            self.prob = np.ones(len(sizes))
        else:
            self.prob = prob
        imSize = self.image.shape
        assert np.all(self.prob>0), 'probabilities for shapes must be >0'
        assert len(self.prob) == len(sizes), 'probabilities and sizes should have equal length'
        if len(np.array(sizes).shape) == 1:
            self.sizes = np.reshape(np.concatenate(np.meshgrid(sizes,sizes),axis=0),[2,9]).transpose()
            self.prob = np.outer(prob,prob).flatten()
        else: 
            self.sizes = np.array(sizes)
        self.colors = colors
        prob = self.prob * (self.sizes[:,0]+imSize[0])/(np.max(self.sizes[:,0])+imSize[0])
        prob = self.prob * (self.sizes[:,1]+imSize[1])/(np.max(self.sizes[:,1])+imSize[1])
        self.prob = prob/np.sum(prob)
        self.probc = prob.cumsum()
    def get_decomposition(self,points):
        points = np.array(points)
        n0 = node()
        self.im = np.copy(self.image)
        n = n0
        all_contained = None
        k = 0
        while np.any(~np.isnan(self.im)):
            n = n.get_sample_child(self.im,self.sizes,self.colors,self.prob)
            k = k+1
            print(k)
            print(np.sum(~np.isnan(self.im)))
            if all_contained is None:
                if np.all(np.logical_and(
                        np.logical_and(points[:,0]>=n.rectList[-1,0],points[:,0]<(n.rectList[-1,0]+n.rectList[-1,2])),
                        np.logical_and(points[:,1]>=n.rectList[-1,1],points[:,1]<(n.rectList[-1,1]+n.rectList[-1,3])))):
                    all_contained = True
                elif np.any(np.logical_and(
                        np.logical_and(points[:,0]>=n.rectList[-1,0],points[:,0]<(n.rectList[-1,0]+n.rectList[-1,2])),
                        np.logical_and(points[:,1]>=n.rectList[-1,1],points[:,1]<(n.rectList[-1,1]+n.rectList[-1,3])))):
                    all_contained = False
        return (n.rectList,all_contained)
        