# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:42:43 2016

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

#params_rf = {'n_estimators':[20,50,100,150,200]}
#RF = ensemble.RandomForestClassifier()
#clf = grid_search.GridSearchCV(RF, params_rf, cv=4)
#clf.fit(xTrain,yTrain)

C_list = [10,20,50,100]
# C_list = 20

error_rate = []

for C in C_list:
    clf = ensemble.GradientBoostingClassifier(n_estimators=C)
    clf.fit(Train_x,Train_y)
    pred = clf.predict(Valid_x)
    error_rate.append(np.sum(np.abs(Valid_y - pred)) / np.sum(Valid_y))

plt.plot(error_rate)    
#plt.plot(C_list,error_rate)
#plt.xscale('log')