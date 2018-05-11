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

train_images=np.load('/Users/shubhamsinha/Documents/data (1)/digitdata/training_images.npy')
train_labels=np.load('/Users/shubhamsinha/Documents/data (1)/digitdata/training_labels.npy')
test_images=np.load('/Users/shubhamsinha/Documents/data (1)/digitdata/test_images.npy')
test_labels=np.load('/Users/shubhamsinha/Documents/data (1)/digitdata/test_labels.npy')
validation_images=np.load('/Users/shubhamsinha/Documents/data (1)/digitdata/validation_images.npy')
validation_labels=np.load('/Users/shubhamsinha/Documents/data (1)/digitdata/validation_labels.npy' )


#percentage=100
#length=int(5000*(percentage/100))
#train_images=train_images[0:length][:]
#train_labels=train_labels[0:length][:]


iterations=200
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
percentage=60
length=int(5000*(percentage/100))
print(length)
for _ in range(2):
    
    rand=random.sample(range(0, 5000), length)
    for i in range(0,length):
        newtrain_images.append(train_images[rand[i]][:])
        newtrain_labels.append(train_labels[rand[i]])

    iterations=200
    l=train_images.shape[1]
    weight_vectors=np.zeros((10,l+1),dtype=int)
    bias=1
    classes=[0,1,2,3,4,5,6,7,8,9]

    start_time=time.time()
    train(newtrain_images,newtrain_labels)
    print("--- %s seconds ---" % (time.time() - start_time))
    p=test(test_images)
    #p2=test(train_images)
    a=getAccuracy(test_labels,p)
    #a2=getAccuracy(train_labels,p2)
    arr.append(1-a)
    newtrain_images=[]
    newtrain_labels=[]

print("mean: "+str(np.mean(arr)))
print("std: "+str(np.std(arr)))