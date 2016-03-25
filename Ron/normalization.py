# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:02:19 2016

@author: justinzhu
"""

import numpy as np
# This code normalize the data

X_Train_nor = np.copy(X_Train)

for i in range(np.shape(X_Train)[1]):
    temp = X_Train[:,i]
    if len(np.unique(temp))!=2: # do not normalize binary features
        X_Train_nor[:,i] = (temp - np.mean(temp))/np.std(temp)