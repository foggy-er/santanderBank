# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:34:55 2016

@author: justinzhu
"""

import numpy as np

# panda reads csv
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Remove constant columns
col_rm = []
for col in df_train.columns:
    print(len(np.unique(df_train[col].values)))
    if len(np.unique(df_train[col].values))==1:
        col_rm.append(col)

df_train.drop(col_rm, axis=1, inplace=True)
df_test.drop(col_rm, axis=1, inplace=True)

# Remove duplicated columns
col_rm = []
columns = df_train.columns
for i in range(len(columns)-1):
    temp = df_train[columns[i]].values
    for j in range(i+1, len(columns)):
        if np.array_equal(temp, df_train[columns[j]].values):
            col_rm.append(columns[j])
            
df_train.drop(col_rm, axis=1, inplace=True)
df_test.drop(col_rm, axis=1, inplace=True)

ID_Train = df_train['ID'].values
Y_Train = df_train['TARGET'].values
X_Train = df_train.drop(['ID', 'TARGET'], axis=1).values

ID_Test = df_test['ID'].values
X_Test = df_train.drop(['ID'], axis=1).values