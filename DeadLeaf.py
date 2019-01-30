# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:03:12 2018

@author: heiko
"""

import numpy as np
import scipy.signal as signal
import tqdm

def gen_rect_leaf(imSize = [255,255],sizes = [5,10,15],colors=[0,0.5,1],grid = 1,noise = 0,noiseType='norm',prob=None,fixedC=0,fixedIdx=[],border=False):
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
        if border:
            idx_x = rectList[len(rectList)-i-1,0]
            idx_y = rectList[len(rectList)-i-1,1]
            sizx = rectList[len(rectList)-i-1,2]
            sizy = rectList[len(rectList)-i-1,3]
            if idx_x >= 0:
                image[int(idx_x),int(max(idx_y,0)):int(max(0,idx_y+sizy))] = 5
            if (idx_x+sizx) <= imSize[0]:
                image[int((idx_x+sizx)-1),int(max(idx_y,0)):int(idx_y+sizy)] = 5
            if idx_y >= 0:
                image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(idx_y)] = 5
            if (idx_y+sizy) <= imSize[1]:
                image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(idx_y+sizy)-1] = 5
                
    if border:
        b = image==5
    if noiseType=='norm':
        image = image+noise*np.random.randn(imSize[0],imSize[1])
    elif noiseType == 'uniform':
        image = image+noise*2*(np.random.rand(imSize[0],imSize[1])-.5)
    image[image<0] = 0
    if border:
        image[image>1] = 1
        image[b] = 5
    else:
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
    def __init__(self,imSize = [255,255],sizes = [5,10,15],colors=[0,0.5,1],grid = 1,noise = 0,noiseType='norm',prob=None,border=False):    
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
        self.border = border
        self.image = np.nan*np.zeros(imSize,dtype='float')
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
        if self.border:
            if idx_x >= 0:
                self.image[int(idx_x),int(max(idx_y,0)):int(max(0,idx_y+sizy))] = 5
            if (idx_x+sizx) <= self.imSize[0]:
                self.image[int((idx_x+sizx)-1),int(max(idx_y,0)):int(idx_y+sizy)] = 5
            if idx_y >= 0:
                self.image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(idx_y)] = 5
            if (idx_y+sizy) <= self.imSize[1]:
                self.image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(idx_y+sizy)-1] = 5
    def get_image(self):
        return self.image

class node:
    def __init__(self):
        self.children = None
        self.probChild = None
    def add_children(self,image,sizes,colors,prob):
        self.children = list()
        self.probChild = list()
        self.probInvisibleChild = list()
        kP = 0
        im2 = ~np.isnan(image)
        sizes = np.int32(sizes)
        for iSize in tqdm.tqdm(sizes):
            recty = np.ones((iSize[0],1))
            rectx = np.ones((1,iSize[1]))
            fieldSize = np.prod(image.shape+iSize-1)
            # Try better convolution? -> only beginning and End needed?
            imTest2 = signal.convolve2d(im2,rectx,'full')
            imTest2 = signal.convolve2d(imTest2,recty,'full')
            locationsInvisible = np.where(imTest2==0)
            for t in np.array(locationsInvisible).T:
                self.probInvisibleChild.append(prob[kP]/fieldSize)
            for iC in colors:
                im = (image-iC)**2
                im[np.isnan(im)]=0
                imTest = signal.convolve2d(im,rectx,'full')
                imTest = signal.convolve2d(imTest,recty,'full')
                locations = np.where(np.logical_and(imTest==0,imTest2!=0))
                for t in np.array(locations).T:
                    self.children.append([t[0]-iSize[0]+1,t[1]-iSize[1]+1,iSize[0],iSize[1],iC,imTest2[t[0],t[1]]])
                    self.probChild.append(prob[kP]/fieldSize)
            kP = kP+1
        self.probInvisible = np.sum(np.array(self.probInvisibleChild))
        #print(self.probInvisible)
        self.probPossible = np.sum(np.array(self.probChild))
    def get_sample_child(self,image,sizes,colors,prob):
        # NOTE: This changes the image although it is not returned!
        if self.children is None:
            self.add_children(image,sizes,colors,prob)
        pc = np.cumsum(self.probChild)
        pc = pc/pc[-1]
        ran = np.random.rand()
        idx = np.argmax(ran<pc)
        child = self.children[idx]
        #print(self.children[idx])
        idx_x = child[0]
        idx_y = child[1]
        sizx = child[2]
        sizy = child[3]
        image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(max(idx_y,0)):int(max(0,idx_y+sizy))] = np.nan
        return (child,self.probPossible,self.probInvisible)
    def get_sample_child_explained_bias(self,image,sizes,colors,prob):
        # NOTE: This changes the image although it is not returned!
        if self.children is None:
            self.add_children(image,sizes,colors,prob)
        pCorrection = np.array(self.children)[:,5]+ np.log(self.probChild)
        pCorrection = pCorrection-np.max(pCorrection)
        p = np.exp(pCorrection)
        pc = np.cumsum(p)
        pc = pc/pc[-1]
        ran = np.random.rand()
        idx = np.argmax(ran<pc)
        child = self.children[idx]
        #print(self.children[idx])
        idx_x = child[0]
        idx_y = child[1]
        sizx = child[2]
        sizy = child[3]
        image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(max(idx_y,0)):int(max(0,idx_y+sizy))] = np.nan
        logpCorrection = np.log(p[idx]/self.probChild[idx])
        return (child,self.probPossible,self.probInvisible,logpCorrection)
        
class graph: 
    def __init__(self,image,sizes,colors,prob=None):
        self.image = np.array(image)
        if prob is None:
            self.prob = np.ones(len(sizes))
        else:
            self.prob = prob
        imSize = self.image.shape
        assert np.all(self.prob>0), 'probabilities for shapes must be >0'
        assert len(self.prob) == len(sizes), 'probabilities and sizes should have equal length'
        if len(np.array(sizes).shape) == 1:
            self.sizes = np.reshape(np.concatenate(np.meshgrid(sizes,sizes),axis=0),[2,len(sizes)**2]).transpose()
            self.prob = np.outer(self.prob,self.prob).flatten()
        else:
            self.sizes = np.array(sizes)
        self.colors = colors
        prob = self.prob * (self.sizes[:,0]+imSize[0])/(np.max(self.sizes[:,0])+imSize[0])
        prob = self.prob * (self.sizes[:,1]+imSize[1])/(np.max(self.sizes[:,1])+imSize[1])
        self.prob = prob/np.sum(prob)
        self.probc = prob.cumsum()
    def get_decomposition(self,points=None):
        logPPos = 0
        logPVis = 0
        rectList = np.zeros((0,6),dtype=np.int16)
        if points is not None:
            points = np.array(points)
        n0 = node()
        im = np.copy(self.image)
        n = n0
        all_contained = None
        k = 0
        while np.any(~np.isnan(im)):
            n = node()
            (rect,pPos,pInVis) = n.get_sample_child(im,self.sizes,self.colors,self.prob)
            logPPos = logPPos + np.log(pPos)
            logPVis = logPVis + np.log(1-pInVis)
            k = k+1
            rectList = np.append(rectList,[rect],axis=0)
            print(k)
            print(np.sum(~np.isnan(im)))
            if all_contained is None and points is not None:
                if np.all(np.logical_and(
                        np.logical_and(points[:,0]>=rectList[-1,0],points[:,0]<(rectList[-1,0]+rectList[-1,2])),
                        np.logical_and(points[:,1]>=rectList[-1,1],points[:,1]<(rectList[-1,1]+rectList[-1,3])))):
                    all_contained = True
                elif np.any(np.logical_and(
                        np.logical_and(points[:,0]>=rectList[-1,0],points[:,0]<(rectList[-1,0]+rectList[-1,2])),
                        np.logical_and(points[:,1]>=rectList[-1,1],points[:,1]<(rectList[-1,1]+rectList[-1,3])))):
                    all_contained = False
        logPCorrection = - rectList.shape[0]* np.log(len(self.colors))
        return (rectList,all_contained,logPPos,logPVis,logPCorrection)
    def get_decomposition_explained_bias(self,points=None):
        logPPos = 0
        logPVis = 0
        logPCorrection = 0
        rectList = np.zeros((0,6),dtype=np.int16)
        if points is not None:
            points = np.array(points)
        n0 = node()
        im = np.copy(self.image)
        n = n0
        all_contained = None
        k = 0
        while np.any(~np.isnan(im)):
            n = node()
            (rect,pPos,pInVis,correction) = n.get_sample_child_explained_bias(im,self.sizes,self.colors,self.prob)
            logPPos = logPPos + np.log(pPos)
            logPVis = logPVis + np.log(1-pInVis)
            logPCorrection = logPCorrection-correction
            k = k+1
            rectList = np.append(rectList,[rect],axis=0)
            print(k)
            print(np.sum(~np.isnan(im)))
            if all_contained is None and points is not None:
                if np.all(np.logical_and(
                        np.logical_and(points[:,0]>=n.rectList[-1,0],points[:,0]<(rectList[-1,0]+rectList[-1,2])),
                        np.logical_and(points[:,1]>=n.rectList[-1,1],points[:,1]<(rectList[-1,1]+rectList[-1,3])))):
                    all_contained = True
                elif np.any(np.logical_and(
                        np.logical_and(points[:,0]>=n.rectList[-1,0],points[:,0]<(rectList[-1,0]+rectList[-1,2])),
                        np.logical_and(points[:,1]>=n.rectList[-1,1],points[:,1]<(rectList[-1,1]+rectList[-1,3])))):
                    all_contained = False
        logPCorrection = logPCorrection- rectList.shape[0]* np.log(len(self.colors))
        return (rectList,all_contained,logPPos,logPVis,logPCorrection)
  
def generate_image(exponent,border,distance,angle,abs_angle,sizes,imSize=np.array([300,300]),num_colors=9):
    prob = (sizes/np.min(sizes)) ** -(exponent/2)
    
    if angle and not abs_angle:
        pos = [[-distance/2,-distance/2],[distance/2,distance/2]]
    elif angle and abs_angle:
        pos = [[-distance/2,distance/2],[distance/2,-distance/2]]
    elif not angle and not abs_angle:
        pos = [[-distance/2,0],[distance/2,0]]
    elif not angle and abs_angle: 
        pos = [[0,-distance/2],[0,distance/2]]
    pos = np.floor(np.array(pos))
    
    positions = pos
    positions = np.floor(positions)
    positions_im = np.zeros_like(positions)
    positions_im[:,1] = np.floor(imSize/2)+positions[:,0]
    positions_im[:,0] = np.floor(imSize/2)-positions[:,1]-1
    col = np.random.randint(num_colors)
    im = gen_rect_leaf(imSize,
          sizes=sizes,
          prob = prob,
          grid=1,
          colors=np.linspace(0,1,num_colors),
          fixedIdx = positions_im,
          fixedC=col,
          border=border)
    image = im[0]
    image = np.repeat(np.expand_dims(image,axis=-1),3,axis=-1)
    image[im[0]==5,:] = [.5,.5,1]
    image[np.asarray(positions_im,dtype=np.int)[:,0],
          np.asarray(positions_im,dtype=np.int)[:,1],:] = [1,0,0]
    return (image,im[1],positions_im,im[2],col)


def generate_image_from_rects(imSize,rectList,border=False,colors=None):
    if colors is None:
        colors = np.arange(np.max(rectList[:,4])+1)/np.max(rectList[:,4])
    image = np.zeros(imSize)
    for i in range(len(rectList)):
        image[int(max(rectList[len(rectList)-i-1,0],0)):int(max(0,rectList[len(rectList)-i-1,0]+rectList[len(rectList)-i-1,2])),
              int(max(rectList[len(rectList)-i-1,1],0)):int(max(0,rectList[len(rectList)-i-1,1]+rectList[len(rectList)-i-1,3]))] = colors[rectList[len(rectList)-i-1,-1]]
        if border:
            idx_x = rectList[len(rectList)-i-1,0]
            idx_y = rectList[len(rectList)-i-1,1]
            sizx = rectList[len(rectList)-i-1,2]
            sizy = rectList[len(rectList)-i-1,3]
            if idx_x >= 0:
                image[int(idx_x),int(max(idx_y,0)):int(max(0,idx_y+sizy))] = 5
            if (idx_x+sizx) <= imSize[0]:
                image[int((idx_x+sizx)-1),int(max(idx_y,0)):int(idx_y+sizy)] = 5
            if idx_y >= 0:
                image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(idx_y)] = 5
            if (idx_y+sizy) <= imSize[1]:
                image[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(idx_y+sizy)-1] = 5
    image3 = np.repeat(np.expand_dims(image,axis=-1),3,axis=-1)
    image3[image==5,:] = [.5,.5,1]
    return image3

def test_positions(rectList,fixedIdx):
    # find whether two points in fixedIdx are on one object or not
    oneObject = False
    idxStart = -1
    while True:
        idxStart = idxStart+1
        R = rectList[idxStart]
        delete = []
        for i in range(len(fixedIdx)):
            if (R[0]<=fixedIdx[i][0]) and ((R[0]+R[2])>fixedIdx[i][0]) and (R[1]<=fixedIdx[i][1]) and ((R[1]+R[3])>fixedIdx[i][1]):
                delete.append(i)
        if len(delete)>1:
            oneObject = True
        if len(delete)>0:
            break
    return oneObject