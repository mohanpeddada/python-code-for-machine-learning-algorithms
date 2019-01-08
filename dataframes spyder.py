# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
dict=datasets.load_iris()
df=pd.DataFrame(data=dict['data'],columns=dict['feature_names'])
df['Target']=dict['target']
df[df['Target']==0]
df[(df['sepal length (cm)'] > 6) & (df['petal length (cm)']<7)]
df.shape
##descrbe is same as summary in R
df.describe()
##columns will give columns names
df.columns
##info is same as str in R
df.info()
##changing normal vaules as categorical values
df['Target']=df['Target'].astype('category')
df.info()
##knowing what type of unique values are present i the column
df['Target'].unique()
##knowing what type of no .values are present i the column
df['Target'].nunique()
##knowing no. ofcount  unique values are present i the column
df['Target'].value_counts()
df.apply()
sq=lambda x:x**2
sq(2)
df['new target']=df['Target'].apply(lambda x:'cat1' if x<=1 else 'cat2')
df
#3deleting a column
df.drop('new target',axis=1,inplace=True)
##adding a row
df.loc[150]=[10,10,10,10,1]
df.loc[151]=[10,10,np.NaN,np.NaN,10]

np.isnan(df)

import pandas as pd
##to know where the null values present

is_df = pd.isnull(df)

## to find the no.of null values
is_df.apply(lambda x: x.sum(), axis = 0)


df.fillna(method='ffill')
df.groupby('Target')['petal width (cm)'].mean() 

df.groupby('Target').mean()


df.groupby('Target')['sepal width (cm)'].agg({'avg:np.mean,'sum':np.sum})
##creating a new  data frame
df2=pd.DataFrame(data=df['Target'])
## adding a new column pf cat.
df2['new_target']=df2['Target'].apply(lambda x:'cat1' if (x==0 or x==1) else 'cat2')
##merging two data frames
df3=pd.merge(df,df2,how='outer',on='Target')
df2.drop('Target',axis=1,inplace=True)

df4=df.join(df2,on='Target')

df5 =pd.concat([df,df2],'outer',1)


x1=np.random.randint(1,100,10)
x2=np.random.randint(1,100,10)

import matplotlib.pyplot as plt

plt.plot(np.sort(x1), np.sort(x2))

plt.scatter(np.sort(x1), np.sort(x2))
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.title("Title")

plt.bar()

import seaborn as sns
##importing file in python
 pd.read_csv(file name)
 exporting
 df.to_csv()
 
 
 importing file in R
 read.csv('filename.csv)
 exporting
 write.csv()
 
 
 ##linear regression
 import pandas as pd
 from sklearn import linear_model
 from sklearn.linear_model import LinearRegression
 from sklearn.model_selection import train_test_
 pd.read_csv("women.csv")
 
 x=data.iloc[:, :-1]
 y=data.iloc[:, -1]
 
 x_train
 








