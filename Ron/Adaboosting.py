# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:10:44 2016

@author: justinzhu
"""

from sklearn import ensemble
import numpy as np
import random

# Cross Validation random number
a = random.sample(range(0, len(X_Train)), int(len(X_Train)/5)) # Validation number
b = np.arange(len(X_Train)) # All number
c = list(set(b)-set(a)) # Training number

xTrain = X_Train[c]
yTrain  = Y_Train[c]
xValid = X_Train[a]
yValid = Y_Train[a]







clf_gb = ensemble.GradientBoostingClassifier(n_estimators=300)
clf_gb.fit(xTrain, yTrain)
prob_gb = clf.predict_proba(xValid)

#clf_rf = ensemble.RandomForestClassifier(n_estimators=20)
#clf_rf.fit(xTrain, yTrain)
#prob_rf = clf_rf.predict_proba(xValid)

clf_ab = ensemble.AdaBoostClassifier(n_estimators=300)
clf_ab.fit(xTrain, yTrain)
prob_ab = clf_ab.predict_proba(xValid)

prob = prob_gb[:,1] + prob_rf[:,1] + prob_ab[:,1]

# errorate = np.sum(np.abs(pred-yValid))/np.sum(yValid)