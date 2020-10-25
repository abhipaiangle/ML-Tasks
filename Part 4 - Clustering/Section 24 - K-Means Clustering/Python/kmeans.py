# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:51:17 2020

@author: Abhishek Pai Angle
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Mall_Customers.csv")
X= dataset.iloc[:,3:5]

from sklearn.cluster import KMeans
wcss=[] 
for i in range(1,11):
     kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel('clusters')
plt.ylabel('WCSS')
plt.show()     

kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_pred=kmeans.fit_predict(X)
plt.scatter(X[y_pred == 0 ,0],X[y_pred == 0,1],color='red',label='Label1',s=100)
plt.scatter(X[y_pred == 1,0],X[y_pred == 1,1],color='blue',label='Label2',s=100)
plt.scatter(X[y_pred == 2,0],X[y_pred == 2,1],color='magenta',label='Label3',s=100)
plt.scatter(X[y_pred == 3,0],X[y_pred == 3,1],color='cyan',label='Label4',s=100)
plt.scatter(X[y_pred == 4,0],X[y_pred == 4,1],color='green',label='Label5',s=100)     
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='yellow',label='Centroids',s=300)
plt.xlabel('Salary')
plt.ylabel('Expenditure')
plt.title('KMeans Clustering')
plt.legend()
plt.show()