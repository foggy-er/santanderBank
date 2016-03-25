# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:35:03 2016

@author: justinzhu
"""

import csv
import numpy as np

with open('test.csv') as f:
    a = csv.reader(f, delimiter=',')
    testdata = []
    for row in a:
        testdata.append(row)

testdata = testdata[1:]
testdata = np.array(testdata)

xTest_data = testdata[:,1:]
xTest = np.zeros([np.shape(xTest_data)[0],np.shape(xTest_data)[1]])

for i in range(np.shape(xTest_data)[0]):
    for j in range(np.shape(xTest_data)[1]):
        xTest[i,j] = float(xTest_data[i,j])