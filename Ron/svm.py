# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:10:44 2016

@author: justinzhu
"""

from sklearn import svm
import numpy as np
import random
import matplotlib.pyplot as plt

# Cross Validation random number
a = random.sample(range(0, len(X_Train)), int(len(X_Train)/5)) # Validation number
b = np.arange(len(X_Train)) # All number
c = list(set(b)-set(a)) # Training number

xTrain = X_Train[c]
yTrain  = Y_Train[c]
xValid = X_Train[a]
yValid = Y_Train[a]

C_list = [1e-5,1e-4,1e-3,1e-2,1e-1,1]

errorate = []

for C in C_list:
    clf = svm.SVC(C=C, kernel='rbf')
    clf.fit(xTrain, yTrain)
    pred = clf.predict(xValid)
    errorate.append(np.sum(np.abs(pred-yValid))/np.sum(yValid))
    
plt.plot(C_list,errorate)
plt.xscale('log')