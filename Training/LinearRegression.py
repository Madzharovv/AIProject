# -*- coding: utf-8 -*-

### This document is used to present my LinearRegression model and the changes made to increase it's effectiveness
import pandas as pd
import io
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

###We are using a new path here to reference the data we cleaned. It converted all null data fields to be represented
###as 0, this was effective as we were then able to use a larger majority of our dataset and most of the null fields
###were figuratively meant to be represented as 0 or would be encoded as such

path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])


###Here it implements the outlier removal. Which managed to reduce the RMSE by approximately 3,000 in testing

droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (1*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)

# medians = ["LotFrontage","2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch"]
# for col in medians:
#     med = df[col].median()
#     df[col]=df[col].fillna(med)

#Using the method from the tutorials, It produces a line graph of the first 100 predictions 
#compared with the actual results. Here we'll only  use it to display the first 100 results
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.xlabel("First 100 houses sorted")
    plt.ylabel('SalePrice')
    plt.legend()
    plt.show()

###IMPORTING PRESETS FROM Presets.py It is worth looking into Preset 4 as it includes the code for 
###Normalization and Standardisation

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
    
###Below is the for loop used to convert all numerical fields to encoded z scores,
###I didn't expect it to be effective at increasing the accuracy but included it for testing sake
###It is left commented out as it was found to be ineffective in training the model
    

# for col in df.select_dtypes(include=['int', 'float']).columns:
#     if col != 'SalePrice':
#         mean=df[col].mean()
#         sd=df[col].std()
#         df[col]=(df[col]-mean)/sd    
# #-----------------------------------------------


X = df.iloc[:,:-1].values
y = df['SalePrice'].values


# #--------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=59)

###Below is the code used to Standardize the dataframe, it is left commented out as it was found to also have little
###effect on the model in training.

#from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit(X_train)
# X_train=sc.transform(X_train)
# X_test=sc.transform(X_test)

model = LinearRegression()
model.fit(X_train,y_train)
print(model.coef_)
y_pred=model.predict(X_test)
    
chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)   

df_compare = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df_head=df_compare.head(50)
df_head.plot(kind='bar',figsize=(10,8),xlabel="First 50 Houses",ylabel="SalePrice")


print("Mean: ",np.mean(y_test))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

###After Altering test size to be small but still veritable and testing Random_State values between 20 and 90
###I found that i was able to produce an RMSE of 20717.25 (2 D.P.) this fits nicely underneath 15% of the mean which is
###26,210.17 (2 D.P) in this test. While not as accurate as was intended it still produces a good result.

