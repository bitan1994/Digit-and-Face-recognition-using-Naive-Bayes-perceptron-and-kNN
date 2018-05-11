#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:55:28 2018

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


#percentage=10
#length=int(5000*(percentage/100))
#newtrain_images=train_images[0:length][:]
#newtrain_labels=train_labels[0:length][:]

def seperateByClass(dataset,labels):
    seperated={}
    for i in range(len(labels)):
        vector=labels[i]
        data=dataset[i]
        if(vector not in seperated):
            seperated[vector]=[]
        seperated[vector].append(data)
    return seperated

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	return summaries

def summarizeByClass(dataset,labels):
	separated = seperateByClass(dataset,labels)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
    if(stdev==0):
        return 1
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities


def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

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
    print("ys")
    rand=random.sample(range(0, 5000), length)
    for i in range(0,length):
        newtrain_images.append(train_images[rand[i]][:])
        newtrain_labels.append(train_labels[rand[i]])
    
#    print("ys")
    start_time=time.time()
    s=summarizeByClass(newtrain_images,newtrain_labels)
    print("--- %s seconds ---" % (time.time() - start_time))
    p=getPredictions(s,validation_images)
    a=getAccuracy(validation_labels,p)
    arr.append(1-a)
    newtrain_images=[]
    newtrain_labels=[]

print("mean: "+str(np.mean(arr)))
print("std: "+str(np.std(arr)))