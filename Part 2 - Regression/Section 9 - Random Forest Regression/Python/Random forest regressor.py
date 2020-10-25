import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values  #1:2 is done so that X is seen as a matrix and not a vector
y=dataset.iloc[:,2:3].values

"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)"""

"""from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
X_train=scx.fit_transform(X_train)
X_test=scx.fit_transform(X_test)
scy= StandardScaler()
y_train=scy.fit_transform(y_train)
y_test=scy.fit_transform(y_test)"""

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=1000,random_state=0)
regressor.fit(X,y)

d=np.array([6.5])
d=d.reshape(-1,1)
y_pred=regressor.predict(d)

#Visualising the model
x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid[:,np.newaxis]
plt.scatter(X,y,c='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('truth or bluff')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
