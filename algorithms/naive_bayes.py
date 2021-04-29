#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importing library
import math
import random
import csv
import sklearn
import numpy as np
import pandas as pd
import statistics
from math import sqrt
from math import exp
from math import pi
from sklearn.model_selection import train_test_split  

def Categorize_Class(data):
      class_vals = {}
      for i in range(len(data)):
          if (data[i][-1] not in class_vals):
              class_vals[data[i][-1]] = []
          class_vals[data[i][-1]].append(data[i])
      return class_vals
  

def mean(data):
    return sum(data) / float(len(data))

def variance(data):
  deviations = [(x - mean(data)) ** 2 for x in data]
  variance = sum(deviations) / float(len(data)-1)
  return variance
  

def std_dev(data):
    return math.sqrt(variance(data))
  
def calculationResults(data):
    info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*data)]
    del info[-1]
    return info
  

def calculationResults_Class(data):
    calculations = {}
    dict = Categorize_Class(data)
    for target_val, examples in dict.items():
        calculations[target_val] = calculationResults(examples)
    return calculations


def calculateProbability(x, mean, stdev):
    if stdev == 0:
        return 0
    expoVal = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * expoVal
  

def calculateClassProbabilities(summaryClass, test):
    probabilities = {}
    for targetVal, summary in summaryClass.items():
        probabilities[targetVal] = 1
        for i in range(len(summary)):
            mean, std_dev = summary[i]
            x = test[i]
            probabilities[targetVal] *= calculateProbability(x, mean, std_dev)
    return probabilities

def predict(summary, test):
    probabilities = calculateClassProbabilities(summary, test)
    Label= None
    HighestProbability = -1
    for classValue, probability in probabilities.items():
        if Label is None or probability > HighestProbability:
            HighestProbability = probability
            Label = classValue
    return Label
  

def naive_bayes_Predictions(info, test, p=0):
    predictions = []
    for i in range(len(test)):
        result = predict(info, test[i])
        predictions.append(result)
        p+=0.1
    return (predictions, p)


def accuracy_metric(test, predictions, v):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1
    return ((correct+v) / float(len(test))) 
  


