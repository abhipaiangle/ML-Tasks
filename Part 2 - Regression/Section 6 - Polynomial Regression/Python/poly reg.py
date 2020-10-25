import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values  #1:2 is done so that X is seen as a matrix and not a vector
y=dataset.iloc[:,2:3].values

from sklearn.linear_model import LinearRegression
lin_reg1=LinearRegression()
lin_reg1.fit(X,y)


from sklearn.preprocessing import  PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)


lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

plt.scatter(X,y,c='red')
plt.plot(X,lin_reg1.predict(X),color='blue')
plt.title('truth or bluff')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

x_grid=np.arange(min(X),max(X),0.1)
x_grid=np.reshape(x_grid,len(x_grid),1)
plt.scatter(X,y,c='red')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.title('truth or bluff')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
d=np.array([6.5])
d=d.reshape(-1,1)
#prdeicting with linear model
lin_reg1.predict(d)
#predicting with polynomial model
lin_reg2.predict(poly_reg.fit_transform(d))