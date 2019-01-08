# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:02:39 2018

@author: sekhar
"""

##Multiple Linear Regression 
##install package scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS
dataset2= pd.read_csv('datset2.csv')
x_train,x_test,y_train,y_test = train_test_split(dataset2.iloc[:,[0,2,3]],dataset2.iloc[:,1],test_size=0.2,random_state=234)
import statsmodels.api as sm

reg__model=sm.OLS(y_train,x_train).fit()


reg__model.summary()
reg__model

predicted_values=(reg__model.predict(x_test))


from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test, predicted_values))

np.exp(predicted_values)


import pandas as pd
dataset2.sort_values('income', ascending = False)

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

model = sfs(LinearRegression(),2,forward=False,n_jobs=-1,floating=True,verbose=3,scoring='r2').fit(np.array(x_train),y_train)


model.k_feature_idx_
from mlxtend.feature_selection import ExhaustiveFeatureSelector as efs
model1 = efs(LinearRegression(),2,forward=False,n_jobs=-1,floating=True,verbose=3,scoring='r2').fit(np.array(x_train),y_train)


from mlxtend.feature_selection import ExhaustiveFeatureSelector as efs
efs(LinearRegression(),1, 3,n_jobs=-1,scoring='r2', print_progress= True, clone_estimator=True).fit((x_train),y_train)