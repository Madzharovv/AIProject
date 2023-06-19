# -*- coding: utf-8 -*-

###This document is the final iteration of the NN models.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
from sklearn.linear_model import Lasso
import tensorflow as tf
import pandas as pd
import io
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (2*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)

# medians = ["LotFrontage","2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch"]
# for col in medians:
#     med = df[col].median()
#     df[col]=df[col].fillna(med)

# preset 2-----------
# Temp = df.select_dtypes(include=['int', 'float'])

# X = Temp.iloc[:,:-1].values
# y = Temp.iloc[:,-1].values

# #--------------------------

    

# preset 4--------------------
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

# # #-------------------- for label encoding -------------
for label in string_columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])
    
    
###One hot encoding produced less accurate results    
    
##----------for one hot encoding--------------
# def encode_text_dummy(df, name):
#     dummies = pd.get_dummies(df[name])
#     for x in dummies.columns:
#         dummy_name = f"{name}-{x}"
#         df[dummy_name] = dummies[x]
#     df.drop(name, axis=1, inplace=True)  
# for label in string_columns:
#     encode_text_dummy(df,label)
    
    
    
## --------------for normalization -------------------------   
# for col in df.select_dtypes(include=['int', 'float']).columns:
#     if col != 'SalePrice':
#         mean=df[col].mean()
#         sd=df[col].std()
#         df[col]=(df[col]-mean)/sd    
# #-----------------------------------------------

###omitting the least relevant features had no effect here
# omit= ['KitchenQual', 'LandSlope',
# 'SaleType', 'LotShape', 'GarageFinish', 'GarageCond','RoofStyle', 'RoofMatl', 'BsmtFinType1','MSZoning', 'BsmtFinType2']
# for col in omit:
#     df.drop(col,inplace=True,axis=1)

X = df.iloc[:,:-1].values
# X = df[ ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',  'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType' and 'SaleCondition']].values
y = df['SalePrice'].values


# #--------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=59)


### Here the use of StandardScaling actually had a great effect, usually decreasing the RMSE value by approximately
### 5,000. I am unsure why it is effective here in comparison to the previous iterations. It might be thanks to the
### inclusion of L1 and L2 regularizers forcing the weights into a better distribution
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

###I settled using 80 units for my 3 hidden layers, and implemented an activity and kernal regularizer for the first layer
###as it had no effect when added to the others. I found that using an L2 activity regularizer and an L1 
###kernel regularizer most effective, following a general trend of lower RMSE results. However only in a range of a 
###few thousand of the RMSE. This concludes that a majority of the features are seen as important in runtime, otherwise
###L1 would have been able to ignore them. and I added L2 as a preventative to hopefully reduce overfitting.
###This final structure was able to get rid of underfitting by scaling up the number of layers and units used.
model=Sequential()
#,kernel_regularizer=regularizers.l1(0.01)
model.add(Dense(80,input_shape=X[1].shape,activation='relu',
                activity_regularizer=regularizers.l2(0.01),
                kernel_regularizer=regularizers.l1(0.01)))
#,activity_regularizer=regularizers.l2(0.01)
model.add(Dense(80,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='relu'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0015)
model.compile(loss='mean_squared_error',optimizer=optimizer) #rmsprop,adam,adamax,

###The monitor here is actually used in the Secondary mode. a patience of 7-10 produced the best results. and in first
###when using validationsplits

monitor=EarlyStopping(monitor='loss',min_delta=1e-1,patience=7,verbose=2,mode='auto')

###This mode of running is the typical runtime, producing an RMSE between 20,000 and 23,000
###I didn't include the validation split here as it had no effect


model.fit(X_train,y_train,verbose=2,epochs=250,callbacks=[monitor],validation_split=0.25)
model.summary()
pred=model.predict(X_test)
print("Shape: {}".format(pred.shape))
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"final score (RMSE): {score}")

### I adapted this version from the tutorial. Which allowed us to track the validation loss and the regular loss.
### as we can see in the graph, the validation loss plateau's and the loss continues to decrease. This is a good
### indicator that it is helping reduce overfitting

# training_trace=model.fit(X_train,y_train,callbacks=[monitor],validation_split=0.25,verbose=0,epochs=250)
# pred=model.predict(X_test)
# score=np.sqrt(metrics.mean_squared_error(pred,y_test))
# print("Final Score: (RMSE): {}".format(score))
# plt.figure(figsize=(10,10))
# plt.plot(training_trace.history['loss'],label ="loss")
# plt.plot(training_trace.history['val_loss'],label="val_loss")
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.legend()
# plt.show()
# print("Shape: {}".format(pred.shape))
# score = np.sqrt(metrics.mean_squared_error(pred,y_test))
# print(f"final score (RMSE): {score}")

###The RMSE here is equal to the other method