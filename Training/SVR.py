#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#library imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.svm import SVR

#load dataset
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])


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


#------------------ Removing outliers ------------------------
# Remove all rows where column is 1 standard deviation out
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)
    
# Function call from above 
remove_outliers(df,'SalePrice',2)

print('After removing outliers: ' + str(df.shape))
#------------------ end removing outliers ------------------------



result = []
for x in df.columns:
    if x != 'SalePrice' and x not in omitted_columns:
        result.append(x)
   
X = df[result].values
y = df['SalePrice'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

# Create LinReg
linReg = LinearRegression().fit(X_train, y_train)

# Predict values
y_pred = linReg.predict(X_test)

# Compare actual to test data
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

# Evaluate accuracy
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 
print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))
print('Correlation: ', stats.pearsonr(y_test,y_pred))


# Plot the outputs
plt.rc('figure', figsize=(5, 5))

# Plot line of best fit
x = np.linspace(500000,0,10)
plt.plot(x, x, '-r')

# Plot prediction / actual
plt.scatter(y_test, y_pred, color='black')

plt.xticks(())
plt.yticks(())

plt.show()

# Plot the values to emphasise the noise
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression(y_pred,y_test,sort=True)  
