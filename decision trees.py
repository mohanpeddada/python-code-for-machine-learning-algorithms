# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:50:41 2018

@author: sekhar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

RandomFor

from sklearn.model_selection import train_test_split
dataset = pd.read_csv('ilpd.csv')



dataset=pd.get_dummies(dataset,drop_first=True)




from sklearn.tree import DecisionTreeClassifier

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X = pd.get_dummies()


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15)
reg_model=DecisionTreeClassifier(criterion='gini')
reg_model.fit(x_train,y_train)
predicted_values = reg_model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predicted_values)




ex-2

data=pd.read_csv('ilpd.csv',header=None )

data=pd.get_dummies(data,drop_first=True)
X=data.iloc[:,0,1_Male]
y=data.iloc[:,9]









y_predict=reg_model.predict(x_test)
plt.scatter(x_train,y_train)
plt.plot(x_train, reg_model.predict(x_train), color = "red")
