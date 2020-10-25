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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder #to change categorical variables to numbers 
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y) 

labelencoder_x=LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])
onehotencoder_x=OneHotEncoder(categorical_features=[0]) #0 is the column here
X=onehotencoder_x.fit_transform(X).toarray()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X=sc.fit_transform(X)
#to split dta into training set and test set so that it is easy for machine to corelate between the data.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)