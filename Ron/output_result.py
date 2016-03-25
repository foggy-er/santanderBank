# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:38:29 2016

@author: justinzhu
"""

import numpy as np
import csv
from sklearn import ensemble

with open('sample_submission.csv') as f:
    a = csv.reader(f, delimiter=',')
    sample = []
    for row in a:
        sample.append(row)
        
clf_rf = ensemble.RandomForestClassifier(n_estimators=20)
clf_ab = ensemble.AdaBoostClassifier(n_estimators=50)

pred = clf.predict(xTest)

for i, p in enumerate(pred):
    sample[i+1][1] = p