#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#-------------------- Load file ----------------------
# File path
path = "../Data/Original/"      
filename_read = os.path.join(path, "train.csv")

# Load csv into dataframe
df = pd.read_csv(filename_read, na_values=['NA', '?'])
#-----------------------------------------------------

#print df col/rows
print(df.shape)

# Using .selectdtype(string) found all columns which will be able to find our encodable entries
string_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
      'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
      'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
      'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
      'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
      'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
      'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
      'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
      'SaleType','SaleCondition']

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

# Strip non-numerics
df = df.select_dtypes(include=['int', 'float'])

#print df col/rows
print(df.shape)


print(df.isnull().any())
# Drop NA values
df = df.dropna()

#df col/rows after clearing
print(df.shape)

        
result = []
for x in df.columns:
    if x != 'SalePrice':
        result.append(x)
   
X = df[result].values
y = df['SalePrice'].values

#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)

# build the model
model = LinearRegression()  
model.fit(X_train, y_train)

print(model.coef_)

#calculate the predictions of the linear regression model
y_pred = model.predict(X_test)

#build a new data frame with two columns, the actual values of the test data, 
#and the predictions of the model
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

df_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# RMSE as a percantage of the mean
print('RMSE in %: ' + str(((np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.mean(y_test))*100)))


# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)   



