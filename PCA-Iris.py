import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn.model_selection as sk
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('iris.csv')
data = data.set_index("Unnamed: 0")

features = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
# Separating out the features
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,['Species']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data[['Species']]], axis = 1)

pca.explained_variance_ratio_





x_train, x_test, y_train, y_test = sk.train_test_split(principalDf, data[['Species']], train_size  = 0.85,  random_state = 123)

model = RandomForestClassifier(n_estimators=100, random_state=123).fit(x_train.fillna(0), y_train)

y_predict = model.predict(x_test.fillna(0))

model.score(x_train, y_train)

np.sqrt(mean_squared_error(y_test.iloc[:,0], y_predict))