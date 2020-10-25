import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:, :-1].values #matrix of features( ind variables)
y= dataset.iloc[:, -1].values #dependent variables

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

"""from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
X_train=scx.fit_transform(X_train)
X_test=scx.fit_transform(X_test)
scy= StandardScaler()
y_train=scy.fit_transform(y_train)
y_test=scy.fit_transform(y_test)"""

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
#Visualising the results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience: training set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience: test set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()