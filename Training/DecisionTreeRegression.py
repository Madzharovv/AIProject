#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (2*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)

string_columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                'SaleType','SaleCondition']


for label in string_columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])

X = df.iloc[:,:-1].values
y = df['SalePrice'].values

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# fit decision tree regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# make a single prediction
result = model.predict(X_test)

# rmse
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, result))
print(rms)