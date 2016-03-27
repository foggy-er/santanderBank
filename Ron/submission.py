# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:53:33 2016

@author: justinzhu
"""

import pandas as pd
from sklearn import ensemble
import numpy as np
import xgboost as xgb

clf_gb = ensemble.GradientBoostingClassifier(n_estimators=200)
clf_gb.fit(X_Train, Y_Train)

clf_xgb = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.05, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X_Train, Y_Train, test_size=0.3)
clf_xgb.fit(X_Train, Y_Train, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_eval, y_eval)])

y_pred= (0.75 * clf_xgb.predict_proba(X_Test) + 0.25 * clf_gb.predict_proba(X_Test))[:,1]

submission = pd.DataFrame({"ID":ID_Test, "TARGET":y_pred})
submission.to_csv("submission75025.csv", index=False)