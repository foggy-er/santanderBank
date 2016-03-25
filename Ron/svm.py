# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:05:00 2016

@author: justinzhu
"""

from sklearn import svm
import random

a = random.sample(range(0, len(Train)), int(len(Train)/5)) # Validation number
b = np.arange(len(Train)) # All number
c = list(set(b)-set(a)) # Training number

Train_x = xTrain[c]
Train_y = yTrain[c]
Valid_x = xTrain[a]
Valid_y = yTrain[a]

C_list = [1e-4,1e-3,1e-2,1e-1,1]
# C_list = [1e-4,1e-3,1e-2,1e-1,1]

error_rate = []

for C in C_list:
    clf = svm.LinearSVC(C=C)
    clf.fit(Train_x,Train_y)
    pred = clf.predict(Valid_x)

    error_rate.append(np.sum(np.abs(Valid_y - pred)) / len(Valid_y))
    
plt.plot(C_list,error_rate)
plt.xscale('log')