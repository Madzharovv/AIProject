# -*- coding: utf-8 -*-

import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

##Utilising the outlier code

droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (1*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)
# medians = ["LotFrontage","2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch"]
# for col in medians:
#     med = df[col].median()
#     df[col]=df[col].fillna(med)
  

# preset 1--------------
# X = df[['YrSold','MoSold','LotArea','BedroomAbvGr']].values.astype(np.float32)
# y = df['SalePrice'].values.astype(np.float32)
#-----------------

#preset 2-----------
# Temp = df.select_dtypes(include=['int', 'float'])

# X = Temp.iloc[:,:-1].values
# y = Temp.iloc[:,-1].values

#--------------------------

#preset 3--------------------
# from sklearn.preprocessing import LabelEncoder
# # Using .selectdtype(string) found all columns which will be able to find our encodable entries
# string_columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
#       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
#       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
#       'SaleType','SaleCondition']

# # #-------------------- for label encoding -------------
# for label in string_columns:
#     df[label] = LabelEncoder().fit(df[label]).transform(df[label])
# # #-----------------------------------------------




# X = df[['Id','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#         'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#         'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#         'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#         'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#         'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
#         'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
#         'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
#         'SaleType','SaleCondition']].values
# y = df['SalePrice'].values
    



#--------------------------

# #preset 4--------------------
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

# #-------------------- for label encoding -------------
for label in string_columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])
# for col in df.columns:
#     mean=df[col].mean()
#     sd=df[col].std()
#     df[col]=(df[col]-mean)/sd    
# #-----------------------------------------------
X = df.iloc[:,:-1].values
y = df['SalePrice'].values


# #--------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)



#### I found that Using the Standard Scaler had no effect here

from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit(X_train)
# X_train=sc.transform(X_train)
# X_test=sc.transform(X_test)

###using the method described in the tutorial, i built an iterator to produce a batch of results for Random
###Forest results using n_estimators up to a value of 50, it also produced a graph to show that it began to produce a
###plateau'd result around 20-30 trees, meaning it would be useless to produce more
###It is clear however that the most increase in accuracy is found in the first 5 trees
accuracy_data=[]
nums=[]
for i in range(1,50):
    rf = RandomForestRegressor(n_estimators=i,random_state=44)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    print("Mean: ",np.mean(y_test))
    print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    accuracy_data.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    nums.append(i)
    
plt.plot(nums,accuracy_data)
plt.xlabel("Number of Trees")
plt.ylabel("RMSE Produced")

###Our RGR model managed to produce an RMSE of approximately 21,000. which also fit underneath the 15% of the mean
###threshold that we were judging our results on.