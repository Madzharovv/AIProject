# -*- coding: utf-8 -*-

###This document is the second iteration of the NN models. I used this file to implement
###All the presets to view how they would affect the accuracy of the model
###I found that preset 2 and 4 produce the most accurate results of the group
###I would also come back to this document before the Final version to test data manipulation/preprocessing.
###This includes normalisation, standardisation, removal of outliers.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (2*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)


    
#preset 1--------------
#X = df[['YrSold','MoSold','LotArea','BedroomAbvGr']].values.astype(np.float32)
#y = df['SalePrice'].values.astype(np.float32)
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

#preset 4--------------------
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
    
    
### I had read that Normalizing data was effective in building a stable and effective neural network model.
### Therefore i decided to try normalize all my numeric fields. however this produced a negligible result.
### Similarly to other tests, even if normalising the data, it would produce less accurate results when increasing the 
### learning rate
    
# for col in df.select_dtypes(include=['int', 'float']).columns:
#     if col != 'SalePrice':
#         mean=df[col].mean()
#         sd=df[col].std()
#         df[col]=(df[col]-mean)/sd      
# #-----------------------------------------------
X = df.iloc[:,:-1].values
y = df['SalePrice'].values


# #--------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=32)

###I found that Using a StandardScaler would produce the model faster, which was effective for testing.
###i decided to attempt using it as i read online that it shouldn't have an overall negative effect.
###However in testing this model. I found there was a difference of approximately 8,000 in the RMSE of the
###model when run with and without the StandardScaler. I am unsure as to why it negatively affected the result here.
###I assume that there is already a strong enough correlation in the characteristics of the dataset's inputs.

# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit(X_train)
# X_train=sc.transform(X_train)
# X_test=sc.transform(X_test)


### I decided to add a third hidden layer to the model, as it was discussed that more than three layers would be able
### to produce a better result but only by negligible amounts.
### I also added an activation function to my output layer as in testing, the average result would be more accurate.
### I cant say for sure if it did actually affect the result. but i kept it moving forward. 
### No other activation functions would produce a result under 100,000 for the RMSE on the output layer either.
model=Sequential()
model.add(Dense(64,input_shape=X[1].shape,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
### I changed to this format as it would allow me to alter the learning rate. I found that 0.001 would provide 
### The most decrease in loss and last the longest before divergence of weights, but still reach a minimum
### however anything lower or higher would drastically increase the RMSE.
### I believe that 0.001 is the default value of Adam's learning rate. but it was worth testing
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error',optimizer=optimizer) #rmsprop,adam,adamax,
###In the case that the model did become accurate enough, i implemented a simple monitor to save time and resources
###it was not utilised here. However, it would have been able to prevent overfitting if the condition was met before
###the end of epochs.
monitor=EarlyStopping(monitor='loss',min_delta=1e-3,patience=5,verbose=1,mode='auto')
model.fit(X_train,y_train,verbose=2,epochs=200,callbacks=[monitor])
model.summary()
pred=model.predict(X_test)
print("Shape: {}".format(pred.shape))
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"final score (RMSE): {score}")
###Average RMSE produced here is between 30,000 and 33,000