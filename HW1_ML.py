#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 11:47:25 2018

@author: justinpeter
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = pd.read_csv('/Users/justinpeter/Documents/Data/iris.txt', header = None)
X = iris.iloc[:, 0:4]
y = iris.iloc[:,4]
NNR = KNeighborsRegressor()
NNR.fit(X, y)

X = iris.iloc[:, 0:4]
y= iris.iloc[:, 4]
NNC = KNeighborsClassifier()
NNC.fit(X, y)

def test():
    print("Hello World")
    
def test2(x):
    y=x+2
    return y

def NNR(X,y):
    d=np.sqrt(np.sum())