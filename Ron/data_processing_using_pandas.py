# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:57:56 2016

@author: justinzhu
"""

from sklearn import ensemble
import numpy as np

clf = ensemble.AdaBoostClassifier(n_estimators=50)

clf.fit(X_Train, Y_Train)