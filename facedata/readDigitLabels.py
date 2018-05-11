#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:25:28 2018

@author: shubhamsinha
"""

import numpy as np

flag=1
fname='/Users/shubhamsinha/Documents/data (1)/facedata/facedatatrainlabels'
with open(fname, "r") as ins:
    labels = []
    for line in ins:
        labels.append(int(line[0]))
        
np.save('/Users/shubhamsinha/Documents/data (1)/facedata/train_labels.npy',labels)