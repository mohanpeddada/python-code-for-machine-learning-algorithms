# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:59:42 2018

@author: sekhar
"""
##simple linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,:1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
reg_model=LinearRegression(n_jobs=-1)
reg_model.fit(x_train,y_train)
y_predict=reg_model.predict(x_test)
plt.scatter(x_train,y_train)
plt.plot(x_train, reg_model.predict(x_train), color = "red")

