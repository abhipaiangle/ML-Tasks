#different method of encoding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('C:\\Users\\Abhishek Pai Angle\\Desktop\\Coding\\Machine learning\\Machine Learning A-Z Template Folder\\Part 1 - Data Preprocessing\\Data.csv')
X= dataset.iloc[:, :-1].values #matrix of features( ind variables)
y= dataset.iloc[:, -1].values #dependent variables

from sklearn.impute import SimpleImputer #to fill the missing places
imputer= SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
#same upto here

from sklearn.compose import ColumnTranformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))