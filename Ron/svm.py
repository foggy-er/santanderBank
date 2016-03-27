# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:10:44 2016

@author: justinzhu
"""

from sklearn import svm
import numpy as np
import random
import matplotlib.pyplot as pltc

# Cross Validation random number
a = random.sample(range(0, len(X_Train)), int(len(X_Train)/5)) # Validation number
b = np.arange(len(X_Train)) # All number
c = list(set(b)-set(a)) # Training number

xTrain = X_Train[c]
yTrain  = Y_Train[c]
xValid = X_Train[a]
yValid = Y_Train[a]

clf = svm.SVR()
clf.fit(xTrain, yTrain)