#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.svm import SVR

#load dataset zeros nulled dataset 
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "non_null.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])



from sklearn.preprocessing import LabelEncoder
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

#-------------------- for label encoding -------------
for label in string_columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])
# #-----------------------------------------------

X = df[['Id','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
        'SaleType','SaleCondition']].values
y = df['SalePrice'].values
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

model = SVR(kernel='linear')
# model2 = SVR(kernel='rbf',C=10000000)
# model3 = SVR(kernel='poly')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)



print('RMSE: %.2f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('Mean: %.2f'%np.mean(y_test))
plt.rc('figure', figsize=(8, 8))