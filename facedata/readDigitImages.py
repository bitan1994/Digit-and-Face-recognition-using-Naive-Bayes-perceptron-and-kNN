#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:15:05 2018

@author: shubhamsinha
"""

import numpy as np
from extractfeature import *

flag=1
fname='/Users/shubhamsinha/Documents/data (1)/facedata/facedatatest'
with open(fname, "r") as ins:
    s=''
    a = []
    for line in ins:
        if(flag==70):
            s=s+line
            flag=1
            a.append(s)
            s=''
        else:
            s=s+line
            flag=flag+1
        
        
       
maxwidth=60
maxheight=70

digit=np.zeros((maxheight,maxwidth),dtype=int)
temp=[]
for element in a:
    digit=np.zeros((maxheight,maxwidth),dtype=int)
    row=0
    for line in element.splitlines():
        for i in range(0,len(line)):
            if(line[i]!=' '):
                digit[row][i]=1
        row=row+1
    
    temp.append(digit.flatten()) 
#    d=digit.flatten()
#    d1=extractfeatures(digit)
#    temp.append(np.concatenate((d,d1),axis=0))

#validation=[]
#for t in temp:
#    validation.append(t.flatten())
         
np.save('/Users/shubhamsinha/Documents/data (1)/facedata/test_images.npy',temp)            

