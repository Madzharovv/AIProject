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
path = "../Data/Laundried/"      
filename_read = os.path.join(path, "nullCleaned.csv")

# Load csv into dataframe
df = pd.read_csv(filename_read, na_values=['NA', '?'])
#-----------------------------------------------------

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


all_columns = ['Id', 'MSSubClass', 'MSZoning', 
               'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 
               'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
               'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
               'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 
               'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
               'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 
               'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
               'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 
               'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
               'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 
               'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 
               'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 
               'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 
               'SalePrice']


# List of columns to omit from data training based on previous analysis
omitted_columns = ['Id', 'KitchenQual', 'LandSlope', 'SaleType', 'LotShape', 
                   'GarageFinish', 'GarageCond','RoofStyle', 'RoofMatl', 
                   'BsmtFinType1', 'MSZoning', 'BsmtFinType2', 'ExterQual', 
                   'FireplaceQu', 'Street', 'PavedDrive', 'TotRmsAbvGrd', 
                   'BsmtFullBath', 'MoSold', 'YearRemodAdd', 'LandContour', 'Utilities',
                   'BsmtHalfBath', 'Foundation']

print(len(omitted_columns))
print(df.shape)

#------------------ Removing outliers ------------------------
# Remove all rows where column is 1 standard deviation out
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)
    
# Function call from above 
remove_outliers(df,'SalePrice',1)

print('After removing outliers: ' + str(df.shape))
#------------------ end removing outliers ------------------------



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

# Print df col/rows
# print(df.shape)


# print(df.isnull().any())
# # Drop NA values
# df = df.dropna()

# #df col/rows after clearing
# print(df.shape)


#----------------- Testing columns to omit -------------------

# Variables for comparison with default values set
# RMSE value
lowest_rmse = 200000
# Column with above RMSE value
lowest_rmse_column = "NA"

# Array of features
result = []

# Loop through every item in columns list
for item in all_columns:
    if item not in omitted_columns:
        #Reset result array
        result = []
        
        # Loop through columns in dataframe
        for x in df.columns:
            
            # Omits SalePrice (target), current item and any other omitted_columns
            if x != 'SalePrice' and x != item and x not in omitted_columns:
                result.append(x)
           
        X = df[result].values
        y = df['SalePrice'].values
        
        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
        
        # Build regression model
        model = LinearRegression()  
        model.fit(X_train, y_train)
            
        # Calculate predictions
        y_pred = model.predict(X_test)
        
        # Compare test and prediction values
        df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df_head = df_compare.head(50)
        print("Dropped column: " + item)
        print('Mean:', np.mean(y_test))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        if np.sqrt(metrics.mean_squared_error(y_test, y_pred)) < lowest_rmse:
            lowest_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            lowest_rmse_column = item
        

# Suggested omission: Column, resulting RMSE
print("Suggested omission: " + lowest_rmse_column + ", " + str(lowest_rmse))

#----------------- end testing columns to omit -------------------



#-------------------- Current data training ----------------------

#Reset result array
result = []

 # Loop through columns in dataframe
for x in df.columns:
    
    # Ignore SalePrice and previously omitted columns
    if x != 'SalePrice' and x not in omitted_columns:
        result.append(x)
   
X = df[result].values
y = df['SalePrice'].values

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

# Build LinReg model
model = LinearRegression()  
model.fit(X_train, y_train)
    
# Calculate predictions
y_pred = model.predict(X_test)

# Compare test and prediction values
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(50)
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(df_head)


# Plot values onto bar chart
df_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.ylabel('SalePrice')
plt.xlabel('First 100 houses sorted')
plt.show()


# Plot regression chart
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('SalePrice')
    plt.xlabel('First 100 houses sorted')
    plt.legend()
    plt.show()
    
chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)  



# Mean of data
print('Mean:', np.mean(y_test))

# RMSE of data
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# RMSE as a percantage of the mean
print('RMSE in %: ' + str(((np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.mean(y_test))*100)))


#------------------ end current data training --------------------

