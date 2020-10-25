import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,3:5]


import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendograph')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

print(X[1,1])

plt.scatter(X[y_hc == 0 ,0],X[y_hc == 0,1],color='red',label='Label1',s=100)
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],color='blue',label='Label2',s=100)
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],color='magenta',label='Label3',s=100)
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],color='cyan',label='Label4',s=100)
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],color='green',label='Label5',s=100)
plt.xlabel('Salary')
plt.ylabel('Expenditure')
plt.title('KMeans Clustering')
plt.legend()
plt.show()