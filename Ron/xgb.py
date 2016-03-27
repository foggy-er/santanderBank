# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 18:58:51 2016

@author: justinzhu
"""


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
import random as random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


#df_train.drop(remove, axis=1, inplace=True)
#df_test.drop(remove, axis=1, inplace=True)

#y_train = df_train['TARGET'].values
#X_train = df_train.drop(['ID','TARGET'], axis=1).values

#id_test = df_test['ID']
#X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
#len_train = len(X_train)
#len_test  = len(X_test)


# Cross Validation random number
a = random.sample(range(0, len(X_Train)), int(len(X_Train)/5)) # Validation number
b = np.arange(len(X_Train)) # All number
c = list(set(b)-set(a)) # Training number

xTrain = X_Train[c]
yTrain  = Y_Train[c]
xValid = X_Train[a]
yValid = Y_Train[a]

# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.05, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

# X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)
# X_fit, X_eval, y_fit, y_eval= train_test_split(X_Train, Y_Train, test_size=0.3)
X_fit, X_eval, y_fit, y_eval= train_test_split(xTrain, yTrain, test_size=0.3)

# fitting
clf.fit(xTrain, yTrain, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])

print('Overall AUC:', roc_auc_score(yTrain, clf.predict_proba(xTrain)[:,1]))
print('Overall Validation AUC:', roc_auc_score(yValid, clf.predict_proba(xValid)[:,1]))

#==============================================================================
# # predicting
# y_pred= clf.predict_proba(X_test)[:,1]
# 
# submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
# submission.to_csv("submission.csv", index=False)
#==============================================================================

print('Completed!')