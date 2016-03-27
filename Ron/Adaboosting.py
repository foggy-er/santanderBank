# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:10:44 2016

@author: justinzhu
"""

from sklearn import ensemble
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Cross Validation random number
a = random.sample(range(0, len(X_Train)), int(len(X_Train)/5)) # Validation number
b = np.arange(len(X_Train)) # All number
c = list(set(b)-set(a)) # Training number

xTrain = X_Train[c]
yTrain  = Y_Train[c]
xValid = X_Train[a]
yValid = Y_Train[a]

clf_gb = ensemble.GradientBoostingClassifier(n_estimators=100)
clf_gb.fit(xTrain, yTrain)
prob_gb = clf_gb.predict_proba(xValid)

clf_ab = ensemble.AdaBoostClassifier(n_estimators=100)
clf_ab.fit(xTrain, yTrain)
prob_ab = clf_ab.predict_proba(xValid)

C_list = [0,0.2,0.4,0.6,0.8,1]
C_list = np.arange(0,0.5,0.05)
auc = []

for C in C_list:

    prob = C * prob_gb[:,1] + (1-C) * prob_ab[:,1]
    
    auc.append(roc_auc_score(yValid, prob))

plt.plot(auc)

#==============================================================================
# for C in C_list:
#     prob_gb = clf_gb.predict_proba(X_Train)
#     prob_ab = clf_ab.predict_proba(X_Train)
# 
#     prob = C * prob_gb[:,1] + (1-C) * prob_ab[:,1]
#     
#     auc.append(roc_auc_score(Y_Train, prob))
# 
# plt.plot(auc)
#==============================================================================



### Maximum at 0.1