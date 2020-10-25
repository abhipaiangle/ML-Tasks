import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values  #1:2 is done so that X is seen as a matrix and not a vector
y=dataset.iloc[:,2:3].values

"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)"""

from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
X=scx.fit_transform(X)
scy= StandardScaler()
y=scy.fit_transform(y)

from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(X,y)

d=np.array([6.5])
d=d.reshape(-1,1)
y_pred=scy.inverse_transform(regressor.predict(scx.transform(d)))

#Visualising the model
x_grid=np.arange(min(X),max(X),0.1)
x_grid=np.reshape(x_grid,len(x_grid),1)
plt.scatter(X,y,c='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('truth or bluff')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
