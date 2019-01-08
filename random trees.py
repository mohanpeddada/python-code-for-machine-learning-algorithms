# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:24:25 2018

@author: sekhar
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv('ilpd.csv',header=None)
dataset = pd.get_dummies(dataset, drop_first = True)

X = dataset.drop(10,axis=1)
y = dataset.iloc[:,-2]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

reg_model=RandomForestClassifier(n_estimators=50,oob_score=True).fit(x_train.fillna(0), y_train)
reg_model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(reg_model.predict(x_test),x_test)