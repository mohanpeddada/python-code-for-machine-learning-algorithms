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
