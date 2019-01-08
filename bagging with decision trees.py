# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:46:18 2018

@author: sekhar
"""

## bagging with decision trees
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

dataset=pd.read_csv('ilpd.csv',header=None)
dataset = pd.get_dummies(dataset, drop_first = True)

X = dataset.drop(10,axis=1)
y = dataset.iloc[:,-2]



x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

reg_model=BaggingClassifier(DecisionTreeClassifier(), n_estimators = 50, oob_score = True).fit(x_train.fillna(0), y_train)
reg_model.predict(x_test)
from sklearn.metrics import accuracy_score