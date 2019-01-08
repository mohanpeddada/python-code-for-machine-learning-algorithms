# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:09:43 2018

@author: sekhar
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Prestige.csv')
x= data.drop('income',axis=1)
y= data.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=123)
ss= MinMaxScaler()
ss.fit_transform(x_train)
ss.transform(x_test)
## lasso
from sklearn.linear_model import Lasso
reg_lasso =  Lasso(alpha=10)
reg_lasso.fit(x_train,y_train)
pred= reg_lasso.predict(x_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred)
np.sqrt(mean_squared_error(y_test, pred))
#3 Ridge
from sklearn.linear_model import Ridge
reg_ridge = Ridge(alpha=20)
reg_ridge.fit(x_train,y_train)

pred_ridge= reg_ridge.predict(x_test)

from 