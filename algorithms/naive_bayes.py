#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 06:42:08 2021

@author: manshaf
"""
import random
import numpy as np
import pandas as pd
from statistics import stdev, fmean
from math import sqrt, exp, pi


def categorize(X, y):
    labels = {}
    for i in range(X.shape[0]):
        try:
            labels[y[i]] = np.vstack((labels[y[i]], X[i]))
        except KeyError:
            labels[y[i]] = np.array([X[i]])
    return labels


def fit(X_train, y_train):
    calculations = {}
    data = categorize(X_train, y_train)
    for label, examples in data.items():
        calculations[label] = [(fmean(examples[:,i]), stdev(examples[:,i], xbar=fmean(examples[:,i]))) for i in range(examples.shape[1])]
    return calculations


def calculate_probabilities(model, X):
    probabilities = {}
    for label, summary in model.items():
        probabilities[label] = 1
        for i in range(len(summary)):
            mean, std_dev = summary[i]
            probabilities[label] *= (
                (1 / (sqrt(2 * pi) * std_dev))
                * exp(-((X[i] - mean) ** 2 / (2 * std_dev ** 2)))
                if std_dev != 0
                else 0
            )
    return probabilities


def predict(model, X_test):
    probabilities = calculate_probabilities(model, X_test)
    best_label = None
    max_probability = 0
    for label, probability in probabilities.items():
        if best_label is None or probability > max_probability:
            max_probability = probability
            best_label = label
    return best_label
