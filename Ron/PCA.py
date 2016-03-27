# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:22:11 2016

@author: justinzhu
"""

# This code does the PCA
from sklearn.decomposition import PCA

XY_Train = np.zeros([np.shape(X_Train)[0], np.shape(X_Train)[1]+1])
XY_Train[:,:-1] = X_Train
XY_Train[:,-1] = Y_Train

pca = PCA(n_components = np.shape(X_Train)[1])
pca.fit(X_Train)

X_Train_pca = pca.transform(X_Train)
X_Train_pca = X_Train_pca[:,:237]