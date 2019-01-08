# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:15:44 2018

@author: sekhar
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_excel('Iris.xls')
dataset= dataset.iloc[:, 0:5]
StandardScaler(dataset)
pca = PCA(n_components=2)
pca.fit(dataset)




import numpy as np
import pandas as pd

data= pd.read_csv('ilpd.csv', header= None)
data= pd.get_dummies(data, drop_first= True)

data.dropna(inplace=True)

from sklearn.model_selection import  cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

x= data.drop(10, axis=1)
y=data.iloc[:, 10]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=123)
model = DecisionTreeClassifier(criterion='gini')



score1= cross_val_score(model, x,y,scoring= 'accuracy',  cv=10, n_jobs=-1)

mean_score= np.mean(score1)


from sklearn.model_selection import GridSearchCV

DecisionTreeClassifier()
min_samples_split()
max_depth

params = {'min_samples_split': [6, 2, 3, 4, 5], 'max_depth': [2, 5, 10, 15]}

model = GridSearchCV(DecisionTreeClassifier(), params, cv=10, n_jobs= -1, scoring= 'accuracy')
model.fit(x, y)

model.best_params_











