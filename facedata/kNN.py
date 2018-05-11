#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:08:12 2018

@author: shubhamsinha
"""

import math
import numpy as np
import warnings
from math import sqrt
import random
from collections import Counter

train_images=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/train_images.npy')
train_labels=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/train_labels.npy')
test_images=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/test_images.npy')
test_labels=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/test_labels.npy')
validation_images=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/validation_images.npy')
validation_labels=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/validation_labels.npy' )

percentage=100
length=int(5000*(percentage/100))
train_image=train_images[0:length][:]
train_labels=train_labels[0:length][:]



def seperateByClass(dataset,labels):
    seperated={}
    for i in range(len(labels)):
        vector=labels[i]
        data=dataset[i]
        if(vector not in seperated):
            seperated[vector]=[]
        seperated[vector].append(data)
    return seperated

def k_nearest_neighbors(data, predict, k=6):
#    if len(data) >= k:
#        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
#            print(euclidean_distance)
            distances.append([euclidean_distance,group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

#    return vote_result

def predict(seperated,testSet,k):
    predicted=[]
    for i in range(len(testSet)):
        result=k_nearest_neighbors(seperated,testSet[i],k)
        predicted.append(result)
    return predicted   

def getAccuracy(testLabels, predictions):
	correct = 0
	for x in range(len(testLabels)):
		if testLabels[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testLabels))) 


arr=[]
newtrain_images=[]
newtrain_labels=[]
#maxk=1
#maxa=0
percentage=50
olen=len(train_labels)
length=int(olen*(percentage/100))
print(length)
for _ in range(2):
    
    rand=random.sample(range(0, olen), length)
    for i in range(0,length):
        newtrain_images.append(train_images[rand[i]][:])
        newtrain_labels.append(train_labels[rand[i]])
#for k in range(1,51,2):
    s=seperateByClass(newtrain_images,newtrain_labels)
    p=predict(s,test_images,11)
    a=getAccuracy(test_labels,p)
#    print(str(k)+"\t"+str(a*100))
#    if a>maxa:
#        maxa=a
#        maxk=k

    arr.append(1-a)
    newtrain_images=[]
    newtrain_labels=[]

print("mean: "+str(np.mean(arr)))
print("std: "+str(np.std(arr)))

  