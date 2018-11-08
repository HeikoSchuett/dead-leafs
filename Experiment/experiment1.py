#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:58:28 2018

@author: Heiko
"""


from psychopy import visual,core,event
import numpy as np
import PIL
import pickle
import os

# This is the only way I found to do this. It looks like a HORRIBLE IDEA
# but unfortunately it works so far.
import sys
sys.path.append('..')
import DeadLeaf as dl
sys.path.pop(-1)

import datetime

im_folder = './images/'
res_folder = './results/'

if not os.path.exists(im_folder):
    os.mkdir(im_folder)
if not os.path.exists(res_folder):
    os.mkdir(res_folder)

n_trials = 1 # per condition
n_cols = [2,3,4,6,8]
distances = [5,10,25,50,100]

core.checkPygletDuringWait = True

#Setup for Screen and 
# Create the window object.
window = visual.Window(fullscr=True,monitor='Experiment1',units='pix',wintype='pyglet',size=[1000,1000])
clock = window.frameClock
event.clearEvents()
#
#core.wait(1,1)
#

#CenterImage.setImage('../sizeInvariantSquares.png')
#CenterImage.draw(window)
#window.flip()
#core.wait(1,1)
    

sizes = 5*np.arange(1,80,dtype='float')
prob = (sizes/np.min(sizes)) **-1.5
imSize = np.array([800,800])



def Trial(window,clock,num_colors=8,positions=[]):
    positions = np.floor(positions)
    positions_im = np.zeros_like(positions)
    positions_im[:,1] = np.floor(imSize/2)+positions[:,0]
    positions_im[:,0] = np.floor(imSize/2)-positions[:,1]-1
    im = dl.gen_rect_leaf(imSize,
          sizes=sizes,
          prob = prob,
          grid=1,
          colors=np.linspace(0,1,num_colors),
          fixedIdx = positions_im,
          fixedC=num_colors-1)
    now = datetime.datetime.now()
    im_name = now.strftime(im_folder+"image%Y_%m_%d_%H_%M_%S.png")
    im_pil = PIL.Image.fromarray(255*im[0]).convert('RGB')
    im_pil.save(im_name,'PNG')
    rect_name = now.strftime(im_folder+"rect%Y_%m_%d_%H_%M_%S.npy")
    np.save(rect_name,im[1])
    same_object = False
    t0 = window.flip()
    CenterImage = visual.ImageStim(window)
    CenterImage.setImage(im_pil)
    CenterImage.draw(window)
    if len(positions)>0 and angle:
        draw_pos_marker(window,positions[0])
        draw_pos_marker(window,positions[1])
    elif len(positions)>0 and not angle:
        draw_pos_marker_diagonal(window,positions[0])
        draw_pos_marker_diagonal(window,positions[1])
    t1 = window.flip()
    #core.wait(5,5)
    keypresses = event.waitKeys(maxWait=5,timeStamped=clock)
    t2 = window.flip()
    #keypresses = event.getKeys(None,False,clock)
    same_object = im[2]
    return im_name,rect_name,keypresses,same_object,(t0,t1,t2),positions,positions_im

def draw_pos_marker(window,pos):
    l1 = visual.Line(window,start=pos+[-15.5,0.5],end=pos+[-3.5,0.5],lineColor='red', lineWidth=1)
    l2 = visual.Line(window,start=pos+[ 15.5,0.5],end=pos+[ 3.5,0.5],lineColor='red', lineWidth=1)
    l3 = visual.Line(window,start=pos+[0.5,-15.5],end=pos+[0.5,-3.5],lineColor='red', lineWidth=1)
    l4 = visual.Line(window,start=pos+[0.5, 15.5],end=pos+[0.5, 3.5],lineColor='red', lineWidth=1)
    l1.draw(window)
    l2.draw(window)
    l3.draw(window)
    l4.draw(window)
    
def draw_pos_marker_diagonal(window,pos):
    l1 = visual.Line(window,start=pos-14.5,end=pos-2.5,lineColor='red', lineWidth=1)
    l2 = visual.Line(window,start=pos+15.5,end=pos+3.5,lineColor='red', lineWidth=1)
    l3 = visual.Line(window,start=pos+[15.5,-14.5],end=pos+[3.5,-2.5],lineColor='red', lineWidth=1)
    l4 = visual.Line(window,start=pos+[-14.5,15.5],end=pos+[-2.5,3.5],lineColor='red', lineWidth=1)
    l1.draw(window)
    l2.draw(window)
    l3.draw(window)
    l4.draw(window)
    
    
#im = dl.gen_rect_leaf([1,1],
#          sizes=sizes,
#          prob = prob,
#          grid=1,
#          colors=np.linspace(0,1,3))
#im = [np.array([[1.0, 0] * 3] * 20).transpose()]
#
#im_pil = PIL.Image.fromarray(255*im[0]).convert('RGB')
#CenterImage = visual.ImageStim(window)
#CenterImage.setImage(im_pil)
#CenterImage.draw(window)
#draw_pos_marker(window,np.array([0,0]))
#draw_pos_marker(window,np.array([0,3]))
#window.flip()
#core.wait(500,500)    



## Main Experiment Script
# results = [n_colours,distance,angle,abs_angle,truth,response]
results = np.zeros((2*n_trials*len(n_cols)*len(distances),6))*np.nan
mesh = np.meshgrid(n_cols,distances,[0,1])

results[:,0] = np.repeat(mesh[0].flatten(),n_trials)
results[:,1] = np.repeat(mesh[1].flatten(),n_trials)
results[:,2] = np.repeat(mesh[2].flatten(),n_trials)
results[:,3] = np.random.randint(2,None,(2*n_trials*len(n_cols)*len(distances)))

np.random.shuffle(results)
resList = []

# Display Introtext
introText = visual.TextStim(window, text='Dear Participant,\n\n' +
                            'you will be shown pictures formed of rectangles as you just saw.\n' +
                            'Two points will be marked with red markers. Please report, whether you believe they fall on the same rectangle or not.\n\n'+
                            'If they fall on the same rectangle, press m,\n'+
                            'if they do not press z.\n\n'+
                            'Only the first press counts and you will get 5 seconds for each image.\n\n'+
                            'The number of different colours will vary.\n'+
                            'However the two points we ask about will always be white\n\n'+
                            '    Press any key to continue', 
                            antialias=False)
introText.wrapWidth=700
introText.draw()
window.flip()

event.waitKeys()

for i in range(len(results)):
    distance = results[i,1]
    n_c = int(results[i,0])
    angle = results[i,2]
    abs_angle = results[i,3]
    if angle and not abs_angle:
        pos = [[-distance/2,-distance/2],[distance/2,distance/2]]
    elif angle and abs_angle:
        pos = [[-distance/2,distance/2],[distance/2,-distance/2]]
    elif not angle and not abs_angle:
        pos = [[-distance/2,0],[distance/2,0]]
    elif not angle and abs_angle: 
        pos = [[0,-distance/2],[0,distance/2]]
    pos = np.floor(np.array(pos))
    resList.append(Trial(window,clock,num_colors=n_c,positions=pos))
    results[i,2] = resList[i][3]
    if not (resList[i][2] is None):
      if len(resList[i][2])>0:
        if resList[i][2][0][0]=='z':
            results[i,5] = -1
        elif resList[i][2][0][0]=='m':
            results[i,5] = 1
        elif resList[i][2][0][0]=='q':
            break
        else:
            results[i,5] = 0
    else:
        results[i,5] = 0
    if resList[i][3]:
        results[i,4] = 1
    else:
        results[i,4] =-1
    print(results[i])


now = datetime.datetime.now()
np.save(now.strftime(res_folder+'result%Y_%m_%d_%H_%M_%S.npy'),results)
with open(now.strftime(res_folder+'resList%Y_%m_%d_%H_%M_%S.pickle'), 'wb+') as f:
    pickle.dump(resList,f)
core.wait(3,3)
window.close()

exit()