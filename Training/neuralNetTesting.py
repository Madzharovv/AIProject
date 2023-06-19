#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.datasets import load_digits

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras.utils
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
import io
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split

#-------------------- Load file ----------------------
# File path
path = "../Data/Laundried/"      
filename_read = os.path.join(path, "nullCleaned.csv")

# Load csv into dataframe
df = pd.read_csv(filename_read, na_values=['NA', '?'])
#-----------------------------------------------------

# Using .selectdtype(string) found all columns which will be able to find our encodable entries
string_columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
      'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
      'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
      'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
      'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
      'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
      'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
      'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
      'SaleType','SaleCondition']


# List of columns to omit from data training based on previous analysis
omitted_columns = ['Id', 'KitchenQual', 'LandSlope', 'SaleType', 'LotShape', 
                   'GarageFinish', 'GarageCond','RoofStyle', 'RoofMatl', 
                   'BsmtFinType1', 'MSZoning', 'BsmtFinType2', 'ExterQual', 
                   'FireplaceQu', 'Street', 'PavedDrive', 'TotRmsAbvGrd', 
                   'BsmtFullBath', 'MoSold', 'YearRemodAdd', 'LandContour', 'Utilities',
                   'BsmtHalfBath', 'Foundation']


#------------------- for label encoding from Joe -------------
from sklearn.preprocessing import LabelEncoder

# Data enumeration
for label in string_columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])
    
# Data normalization
for col in df.select_dtypes(include=['int', 'float']).columns:
    if col != 'SalePrice':
        mean=df[col].mean()
        sd=df[col].std()
        df[col]=(df[col]-mean)/sd 
#------------------- end label encoding from Joe -----------------


#Load columns
result = []
for x in df.columns:
    if x != 'SalePrice' and x not in omitted_columns:
        result.append(x)
   
X = df[result].values
y = df['SalePrice'].values


print(y[:20])
#use the keras built in to ensure the targets are categories
y = keras.utils.to_categorical(y)
#and check this...
print(y[:5])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=40)

print(X_train.shape)

# Keras model 
model = Sequential()

# Add layers
# Relu activations
model.add(Dense(4, input_dim=X.shape[1], activation='relu'))
model.add(Dense(5, activation='relu'))

# Sigmoid activations
#model.add(Dense(128, input_dim=X.shape[1], activation='sigmoid'))
#model.add(Dense(20, activation='sigmoid'))

model.add(Dense(y.shape[1],activation='softmax'))

# Setting error measure
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Fit model
model.fit(X_train,y_train,verbose=1,epochs=5)

# Make predictions (will give a probability distribution)
pred = model.predict(X_test)

print(pred[4])
# Pick the most likely outcome
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1) 

# Evaluate outcome
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))