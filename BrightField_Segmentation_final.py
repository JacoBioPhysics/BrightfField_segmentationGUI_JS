# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:19:39 2014

@author: solari
"""

import scipy as sp
from scipy import optimize
import numpy as np

import mahotas as mht
from scipy import ndimage

from skimage.feature import peak_local_max
import glob
import os as os
from pylab import *
from skimage import morphology as mph
from skimage.filter import threshold_otsu, threshold_adaptive
from skimage import measure as msr
from skimage import segmentation as sgm

import scipy.ndimage as ndi

import matplotlib.pyplot as plt

def mySegmentation(img,s,method='adaptive',BoW='B',thr=0.75,l_th1=0,l_th2=550,seeds_thr1=50,seeds_thr2=500,block_size=7,offs=-0,visual=0):
    
    """the user can choose wether to use otsu for seeds (merkers) definition or get seeds from the standard deviation map"""
    img=Prepare_im(img);  
    sz=np.shape(img)
    seeds=np.zeros((sz[0],sz[1]))

    if method=='otsu':


        t=threshold_otsu(img.astype(uint16))*thr
        l=img<t

                #seeds=abs(seeds-1)      

        [l,N]=msr.label(l,return_num=True)

        [l,N]=remove_reg(l,l_th1,l_th2)   



    if visual:
        figure();imshow(img)
        figure();imshow(seeds)
        figure();imshow(l)
        
        
    if method=='adaptive':


        binary_adaptive = threshold_adaptive(-img, block_size, offset=offs)

        l=binary_adaptive

        [l,N]=msr.label(l,return_num=True)

        l,N=remove_reg(l,l_th1,l_th2)


        l=l!=0

        l=sgm.clear_border(l)

        l=mph.dilation(l)


        [l,n]=msr.label(l,return_num=True)


              

    if method=='std' :

#%compute otsu mask

        #s=std_image(img,0)
        t=mht.otsu(img.astype(uint16))*thr
        tempseeds=img<t
 

        s2=np.copy(s)
        s2=ndi.maximum_filter(s2,10)
        local_maxi = peak_local_max((s2 ).astype(np.double), indices=False,footprint=np.ones((10, 10)),min_distance=100000)
               

        #seeds=pymorph.regmin((-s2).astype(np.uint16)) #,array([[False,True,True,False],[False,True,True,False]]))
        seeds=local_maxi

        #seeds,n=mht.label(seeds)
        im=Prepare_im(img)
        t=threshold_otsu(im)
        mask=im<t*0.85

        seeds=msr.label(seeds)
        seeds,N=remove_reg(seeds,seeds_thr1,seeds_thr2)   
    
       # l = mht.cwatershed(img, seeds)
        l = mph.watershed(img, msr.label(local_maxi),mask=mph.binary_dilation(mask))
        #l=mph.watershed(img,seeds)
        l=l.astype(int32)        
        l,n=remove_reg(l,l_th1,l_th2)
        l=mht.labeled.remove_bordering(l)
        print 'label'
        print mht.labeled.labeled_size(l)
        [l,n]=msr.label(l,return_num=True)

    if visual:
        figure();imshow(img)
        figure();imshow(seeds)
        figure();imshow(l)
        
    return seeds,N,l


    if visual:
        figure();imshow(img)
        figure();imshow(seeds)
        figure();imshow(l)
        
    
        

    return seeds,N,l

        
#############################################################################    
"""functions used by many other functions"""
#############################################################################


    

"""Fitting the plane to subract"""
def myplane(p,x,y):

    return p[0]*x+p[1]*y+p[2] 
    
def res(p,data,x,y):
    
    a=(data-myplane(p,x,y));
    
    return array(np.sum(np.abs(a**2)))
    
def fitplane(data,p0):
    
    s=np.shape(data);
    
    [x,y]=meshgrid(arange(s[1]),arange(s[0]));
    #p=optimize.leastsq(res,p0,args=(data,x,y),factor=1,maxfev=1000000);
    p=optimize.fmin(res,p0,args=(data,x,y));
    return p

    
def createDisk( size ):
    x, y = np.meshgrid( np.arange( -size, size ), np.arange( -size, size ) )
    diskMask = ( ( x + .5 )**2 + ( y + .5 )**2 < size**2)
    return diskMask
 

# =prepare the image for watershed transform correcting for inhomogeneous illumination 
def Prepare_im(img):
        
    s=np.shape(img)    
    p0=np.array([0,0,0])
    
    p0[0]=(img[0,0]-img[0,-1])/s[0]    
    p0[1]=(img[1,0]-img[1,-1])/s[1]
    p0[2]=img.mean()
    
    [x,y]=np.meshgrid(np.arange(s[1]),np.arange(s[0]))
    
    p=fitplane(img,p0)    
    img=img-myplane(p,x,y)    
    
    img=ndimage.gaussian_filter(img,1)
    m=img.min()
    img=img-m
    #img=abs(img)
    
    img=img.astype(uint16)
    
    
    return img

"""compute and returns the standard deviation image of an image"""
def std_image(img,visual=0,subs=2):
    
    img=ndimage.gaussian_filter(img,3)
    img_std = ndimage.filters.generic_filter(img, np.std, size=subs);
    if visual:    
        imshow(img_std)
    
    return img_std

        
        
def remove_reg(l,th1,th2):        
    """l must be a labeled image"""



    for i in range(l.max()):
        
        label_size=(l==i).sum()

        if (label_size>th2):
            l[l==i]=0
        if (label_size<th1):
            l[l==i]=0

    [l,N]=msr.label(l,return_num=True)
                   
    return l,N
    
    


        


        
        
        
        
        
        
        
        
        
        
