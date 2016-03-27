# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 22:53:55 2016

@author: justinzhu
"""
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
from sklearn.cross_validation import train_test_split
import xgboost as xgb

number = 1

for num in range(number):
    # Cross Validation random number
    a = random.sample(range(0, len(X_Train)), int(len(X_Train)/5)) # Validation number
    b = np.arange(len(X_Train)) # All number
    c = list(set(b)-set(a)) # Training number

    xTrain = X_Train[c]
    yTrain = Y_Train[c]
    xValid = X_Train[a]
    yValid = Y_Train[a]
    
    # Gradient Boosting Classifier
    clf_gb = ensemble.GradientBoostingClassifier(n_estimators=200)
    clf_gb.fit(xTrain, yTrain)
    prob_gb = clf_gb.predict_proba(xValid)
    
    # AdaBoosting
    clf_ab = ensemble.AdaBoostClassifier(n_estimators=50)
    clf_ab.fit(xTrain, yTrain)
    prob_ab = clf_ab.predict_proba(xValid)
    
    # XGB Classifier
    clf_xgb = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.05, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
    X_fit, X_eval, y_fit, y_eval= train_test_split(xTrain, yTrain, test_size=0.3)
    clf_xgb.fit(xTrain, yTrain, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_eval, y_eval)])
    prob_xgb = clf_xgb.predict_proba(xValid)
    
    C1_list = np.arange(0,1.05,0.05)
    C2_list = np.arange(0,1.05,0.05)
    auc_xgb_gb_ab = np.zeros([21,21])
    
    for i, C1 in enumerate(C1_list):
        for j, C2 in enumerate(C2_list):
            prob = C1 * prob_xgb + (1-C1) * (C2 * prob_gb + (1-C2) * prob_ab)
            auc_xgb_gb_ab[i,j] = roc_auc_score(yValid, prob[:,1])
            
    auc = auc + auc_xgb_gb_ab

# plt.imshow(auc, cmap='hot', interpolation='nearest')


