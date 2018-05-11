#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:19:02 2018

@author: shubhamsinha
"""

import math
import numpy as np
import time
import random

train_images=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/train_images.npy')
train_labels=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/train_labels.npy')
test_images=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/test_images.npy')
test_labels=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/test_labels.npy')
validation_images=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/validation_images.npy')
validation_labels=np.load('/Users/shubhamsinha/Documents/data (1)/facedata/validation_labels.npy' )


#percentage=100
#length=int(5000*(percentage/100))
#train_images=train_images[0:length][:]
#train_labels=train_labels[0:length][:]


iterations=5
l=train_images.shape[1]
weight_vectors=np.zeros((10,l+1),dtype=int)
bias=1
classes=[0,1,2,3,4,5,6,7,8,9]

def train(images,labels):
    for _ in range(iterations):
        for i in range(0,len(images)):
            featurelist=images[i]#taking one training vector
            feature_vector=np.append(featurelist,bias)#adding 1 to make it same as weight because bias
            
            
            #initializing
            arg_max, predicted_class = 0, classes[0]
            
            # Multi-Class Decision Rule:
            for c in classes:
                current_activation = np.dot(feature_vector, weight_vectors[c])
                if current_activation >= arg_max:
                    arg_max, predicted_class = current_activation, c

            # Update Rule:
            if not (labels[i] == predicted_class):
                weight_vectors[labels[i]] += feature_vector
                weight_vectors[predicted_class] -= feature_vector
            

def test(images):
    predicted=[]
    for i in range(0,len(images)):
        featurelist=images[i]#taking one training vector
        feature_vector=np.append(featurelist,bias)#adding 1 to make it same as weight because bias
        
        
        #initializing
        arg_max, predicted_class = 0, classes[0]
        
        # Multi-Class Decision Rule:
        for c in classes:
            current_activation = np.dot(feature_vector, weight_vectors[c])
            if current_activation >= arg_max:
                arg_max, predicted_class = current_activation, c
        
        predicted.append(predicted_class)
    
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
olen=len(train_labels)
percentage=100
length=int(olen*(percentage/100))
print(length)
for _ in range(10):
    
    rand=random.sample(range(0, olen), length)
    for i in range(0,length):
        newtrain_images.append(train_images[rand[i]][:])
        newtrain_labels.append(train_labels[rand[i]])

    iterations=5
    l=train_images.shape[1]
    weight_vectors=np.zeros((10,l+1),dtype=int)
    bias=1
    classes=[0,1,2,3,4,5,6,7,8,9]

    start_time=time.time()
    train(newtrain_images,newtrain_labels)
    print("--- %s seconds ---" % (time.time() - start_time))
    p=test(validation_images)
#    p2=test(train_images)
    a=getAccuracy(validation_labels,p)
    
#    a2=getAccuracy(train_labels,p2)*100
    arr.append(1-a)
    newtrain_images=[]
    newtrain_labels=[]

print("mean: "+str(np.mean(arr)))
print("std: "+str(np.std(arr)))