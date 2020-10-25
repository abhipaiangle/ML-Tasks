import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
X= dataset.iloc[:, :-1].values #matrix of features( ind variables)
y= dataset.iloc[:, -1].values #dependent variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
X[:,3]=le.fit_transform(X[:,3])
ohe=OneHotEncoder(categorical_features=[3])
X=ohe.fit_transform(X).toarray()
#Avoiding the dummy variable trap
X=X[:,1:]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

"""from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
X_train=scx.fit_transform(X_train)
X_test=scx.fit_transform(X_test)
scy= StandardScaler()
y_train=scy.fit_transform(y_train)
y_test=scy.fit_transform(y_test)"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,2,3,4,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,3,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,3]]
regressor_OLS = sm.OLS(y,X_opt).fit()
print(regressor_OLS.summary())

