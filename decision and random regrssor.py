# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 12:30:40 2018

@author: sekhar
"""
##Decison tree regressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('Prestige.csv')
X = data.drop('income',axis=1)
y = data.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.85, random_state= 123)

## random forest regressor

from sklearn.ensemble import RandomForestRegressor
reg_model = RandomForestRegressor(n_estimators=100,n_jobs=-1)


reg_model.fit(x_train, y_train)
predict_values_test = reg_model.predict(x_test)
predict_values_train = reg_model.predict(x_train)
reg_model.score(x_train, y_train)
reg_model.score(x_test, y_test)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_train,reg_model.score(x_train, y_train)))

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predict_values)
reg_model.score

##decision tree regressor
 from sklearn.tree import DecisionTreeRegressor
 reg_model1= DecisionTreeRegressor(random_state=123)
 reg_model1.fit(x_train, y_train)
 pred= reg_model1.predict(x_test)
 from sklearn.metrics import mean_squared_error
 mean_squared_error(y_test, pred)
 np.sqrt(mean_squared_error(y_test, pred))
 from sklearn.metrics import 
 
 