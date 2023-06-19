# -*- coding: utf-8 -*-

###This document uses largely the same code as the LinearRegression.py file Therefore will be commented less extensively
import pandas as pd
import io
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (1*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

#preset 1--------------
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

# #-------------------- for label encoding -------------
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
model = LinearRegression()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=59)
#model.fit(X_train,y_train)
# print(model.coef_)
# y_pred=model.predict(X_test)
    

from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit(X_train)
# X_train=sc.transform(X_train)
# X_test=sc.transform(X_test)


###Here i tested a range of Kfold values between 1 and 10, and then each 10 up to 50. 
###I found that higher Kfold values would produce a greatly random result as theres so much variation in the data, 
###by having more folds it felt as though it would be trained less effectively than using a test/train split.
###Around 3 folds produced the most optimal results for me. 

kf = KFold(3,shuffle=True)
fold = 1
for train_index,validate_index in kf.split(X,y):
    model.fit(X[train_index],y[train_index])
    y_test=y[validate_index]
    y_pred=model.predict(X[validate_index])
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('RMSE: %.2f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    print('Mean: %.2f'%np.mean(y_test))
    fold+=1

###It produced in a standard test, RMSE values between 22,000 and 31,000. less reliable than if using the test/train split
###Therefore I decided it would be best to avoid it in the implementation of other models.

chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)   

df_compare = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df_head=df_compare.head(50)
df_head.plot(kind='bar',figsize=(10,8))

print("Mean: ",np.mean(y_test))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))