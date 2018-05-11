#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:47:17 2018

@author: shubhamsinha
"""
import numpy as np
from numpy import mgrid, sum


def extractfeatures(image):
    x, y = mgrid[:image.shape[0],:image.shape[1]]
    moments = {}
    moments['mean_x'] = sum(x*image)/sum(image)
    moments['mean_y'] = sum(y*image)/sum(image)
          
    # raw or spatial moments
    moments['m00'] = sum(image)
    moments['m01'] = sum(x*image)
    moments['m10'] = sum(y*image)
    moments['m11'] = sum(y*x*image)
    moments['m02'] = sum(x**2*image)
    moments['m20'] = sum(y**2*image)
    moments['m12'] = sum(x*y**2*image)
    moments['m21'] = sum(x**2*y*image)
    moments['m03'] = sum(x**3*image)
    moments['m30'] = sum(y**3*image)
  
    # central moments
    # moments['mu01']= sum((y-moments['mean_y'])*image) # should be 0
    # moments['mu10']= sum((x-moments['mean_x'])*image) # should be 0
    moments['mu11'] = sum((x-moments['mean_x'])*(y-moments['mean_y'])*image)
    moments['mu02'] = sum((y-moments['mean_y'])**2*image) # variance
    moments['mu20'] = sum((x-moments['mean_x'])**2*image) # variance
    moments['mu12'] = sum((x-moments['mean_x'])*(y-moments['mean_y'])**2*image)
    moments['mu21'] = sum((x-moments['mean_x'])**2*(y-moments['mean_y'])*image) 
    moments['mu03'] = sum((y-moments['mean_y'])**3*image) 
    moments['mu30'] = sum((x-moments['mean_x'])**3*image) 

    feature=np.zeros((10),dtype=int)
    feature[0]=moments['mean_x']
    feature[1]=moments['mean_y']
    feature[2]=moments['m00']
    feature[3]=moments['mu11']
    feature[4]=moments['mu02']
    feature[5]=moments['mu20']
    feature[6]=moments['mu12']
    feature[7]=moments['mu21']
    feature[8]=moments['mu03']
    feature[9]=moments['mu30']
    
    return feature
