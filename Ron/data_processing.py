# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np

with open('train.csv') as f:
    a = csv.reader(f, delimiter=',')
    data = []
    for row in a:
        data.append(row)

Data = data[1:]
Data = np.array(Data)

xTrain_data = Data[:,1:-1]
yTrain_data = Data[:,-1]
xTrain = np.zeros([np.shape(xTrain_data)[0],np.shape(xTrain_data)[1]])
yTrain = np.zeros(len(yTrain_data))

for i in range(np.shape(xTrain_data)[0]):
    for j in range(np.shape(xTrain_data)[1]):
        xTrain[i,j] = float(xTrain_data[i,j])
        
for i in range(len(yTrain_data)):
    yTrain[i] = int(yTrain_data[i])