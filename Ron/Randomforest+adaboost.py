# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:45:42 2016

@author: justinzhu
"""

from sklearn import ensemble, grid_search
import random

a = random.sample(range(0, len(Train)), int(len(Train)/5)) # Validation number
b = np.arange(len(Train)) # All number
c = list(set(b)-set(a)) # Training number

Train_x = xTrain[c]
Train_y = yTrain[c]
Valid_x = xTrain[a]
Valid_y = yTrain[a]

#params_rf = {'n_estimators':[5,10,20,30,40]}
#RF = ensemble.RandomForestClassifier()
#clf = grid_search.GridSearchCV(RF, params_rf, cv=4)
#clf.fit(xTrain,yTrain)

C_list = [0.6,0.65,0.7,0.75,0.8,0.9,1]

error_rate = []

for C in C_list:
    clf_rf = ensemble.RandomForestClassifier(n_estimators=20)
    clf_ab = ensemble.AdaBoostClassifier(n_estimators=50)
    clf_rf.fit(Train_x,Train_y)
    clf_ab.fit(Train_x,Train_y)
    pred_rf = clf_rf.predict(Valid_x)
    pred_ab = clf_ab.predict(Valid_x)
    pred_mix = pred_rf * C + pred_ab * (1-C)
    pred_mix = pred_mix + 0.5
    pred = np.zeros(len(pred_mix))
    for i, p in enumerate(pred_mix):
        pred[i] = int(p)
    error_rate.append(np.sum(np.abs(Valid_y - pred)) / len(Valid_y))
    
plt.plot(C_list,error_rate)
# plt.xscale('log')