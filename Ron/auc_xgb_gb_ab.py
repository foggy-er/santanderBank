# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:45:17 2016

@author: justinzhu
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

prob_xgb = clf.predict_proba(xValid)

C1_list = np.arange(0,1.05,0.05)
C2_list = np.arange(0,1.05,0.05)

auc_xgb_gb_ab = np.zeros([21,21])

for i, C1 in enumerate(C1_list):
    for j, C2 in enumerate(C2_list):
        prob = C2 * prob_xgb + (1-C2) * (C1 * prob_gb + (1-C1) * prob_ab)
        auc_xgb_gb_ab[i,j] = roc_auc_score(yValid, prob[:,1])
        
plt.imshow(auc_xgb_gb_ab, cmap='hot', interpolation='nearest')