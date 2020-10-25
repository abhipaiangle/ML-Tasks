import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)#quoting is to ignore double quotes

#Cleaning the dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#nltk.download('stopwords') #run it once to download the module
corpus=[]
for i in range(len(dataset)):
    
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#only keeping the words
    review=review.lower()#lowering case
    review=review.split()#splitting sentence into string array
    ps=PorterStemmer()#for stemming word(most basic version)
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)#join the string array to sentence
    corpus.append(review)

#making the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1:2].values

#Classifying data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

'''Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)'''
                                     
Accuracy=(cm[1,1]+cm[0,0])/(cm[1,1]+cm[1,0]+cm[0,0]+cm[0,1])   

Precision=cm[1,1]/(cm[0,1]+cm[1,1])

Recall=cm[1,1]/(cm[1,0]+cm[1,1])   

F1_Score= (2 * Precision * Recall) / (Precision + Recall)                           