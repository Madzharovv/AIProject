#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#load dataset
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

df = df.select_dtypes(include=['int', 'float'])

#print(dataset.shape)
#print(dataset[:5])


result = []
for x in df.columns:
    if x != 'SalePrice':
        result.append(x)
   
X = df[result].values
y = df['SalePrice'].values

#print(X[:5])
print(X[0:5])
print(y[0:5])
print(X.shape)

#split into testing and training
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

print(X_train.shape)

#set up kNN
knn_model = KNeighborsClassifier(n_neighbors=12)

#fit and test kNN
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn))


import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure(figsize=(10,10))

accuracy_data = []
nums = []
for i in range(1,100):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    y_model = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_model)
    accuracy_data.append(accuracy)
    nums.append(i)
    
print(accuracy_data)
print(nums)
    
plt.plot(nums,accuracy_data)
plt.xlabel("Number of neighbours")
plt.ylabel("Accuracy")
plt.show()
    

fig2 = plt.figure(figsize=(10,10))

accuracy_data = []
nums = []
for i in range(1,100):
    rf_model = RandomForestClassifier(n_estimators=i, criterion='gini')
    rf_model.fit(X_train, y_train)
    y_model = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_model)
    accuracy_data.append(accuracy)
    nums.append(i)
    
print(accuracy_data)
plt.plot(nums,accuracy_data)
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.show()




#fit and test kNN
knn_model = KNeighborsClassifier(n_neighbors=12)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn))

#set up random forest
rf_model = RandomForestClassifier(n_estimators = 24, criterion='entropy')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('RF Accuracy: %.3f' % accuracy_score(y_test, y_pred_rf))

#set up naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
y_pred_gnb = gnb_model.predict(X_test)
print('GNB Accuracy: %.3f' % accuracy_score(y_test, y_pred_gnb))