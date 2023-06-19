# -*- coding: utf-8 -*-

##This document was built to speed up the building process of other modules.
#I intended for these to be presets that would be copy/pasteable into other files for various testing
#All presets were built and tested on the original LinearRegression model built by Patrick that was later removed
#during an unknown github push/pull

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#built before most of our data cleaning therefore it used the original csv.
path = "./Data/Original/" 
filename_read = os.path.join(path, "train.csv")
pd.options.display.max_columns = None

#Base for presets
PresetBase = pd.read_csv(filename_read)

###Preset 1 is built to mimic the original model variables used by the competition winner of the study.
###We do not know the extent of their data cleaning/manipulation however so it was ideal to compare the difference 
###in model accuracy between ours and theirs.

# ##------------------------------------------------------------------
# # Preset 1, only selecting fields used in model used by competition winner (Tested with Patricks 
# # original Linear Regression model)
# X = PresetBase[['Id','YrSold','MoSold','LotArea','BedroomAbvGr']].values
# y = PresetBase['SalePrice'].values
# print(X)
# #split data into testing and training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# ##-------------------------------------------------------------------

###Preset 2 was built to select all Numerical fields for model input.
###I did this by using a Temporary dataframe variable as I didn't want to make direct manipulations to the original
###dataframe, incase we would need to tweak anything before/after.
###In testing, the numerical fields appeared to hold the most weight in producing accurate results
###Luckily our target feature was the Last feature, so we could use iloc to take all features except the last for x

# ##------------------------------------------------------------------
# # Preset 2, only selecting Numerical fields (Tested with Patricks 
# # original Linear Regression model)
# Temp = PresetBase.select_dtypes(include=['int', 'float'])
# #for testing purposes remove Na fields
# Temp = Temp.dropna()
# X = Temp.iloc[:,:-1]
# y = Temp.iloc[:,-1]

# print(X)
# #split data into testing and training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# ##-------------------------------------------------------------------


###Preset 3 was built to specifically test the accuracy of models using encoded categorical data.
###This was strictly for testing purposes and enabled me to easily identify all categorical features in the future
###Using .selectdtype(object) I was able to find all of these features and store them in an array.
### This array could then be passed into a loop for label encoding


# ##------------------------------------------------------------------
# # Preset 3, only selecting String fields, Under various encoding (Tested with Patricks 
# # original Linear Regression model)
# from sklearn.preprocessing import LabelEncoder

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
#     PresetBase[label] = LabelEncoder().fit(PresetBase[label]).transform(PresetBase[label])
# # #-----------------------------------------------




# X = PresetBase[['Id','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#         'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#         'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#         'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#         'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#         'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
#         'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
#         'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
#         'SaleType','SaleCondition']].values
# y = PresetBase['SalePrice'].values
    

# print(X)
# #split data into testing and training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
# ##-------------------------------------------------------------------


###Preset 4 includes all fields in the dataset, with all categorical fields being label encoded
###Uses the same method as preset 3.
###Since it is using all features, i can set X to hold the value of all features except the last as it is our 
###target variable. I got this idea from a stackoverflow article: https://stackoverflow.com/questions/53608653/how-to-select-all-but-the-3-last-columns-of-a-dataframe-in-python


##------------------------------------------------------------------
# Preset 4, Selecting all fields (Tested with Patricks 
# original Linear Regression model)
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
    PresetBase[label] = LabelEncoder().fit(PresetBase[label]).transform(PresetBase[label])
# #-----------------------------------------------
#for testing purposes remove Na fields
PresetBase = PresetBase.dropna()
print


X = PresetBase.iloc[:,:-1]
y = PresetBase['SalePrice'].values
    

print(X)
#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
##-------------------------------------------------------------------





####This is the original linearRegression model created by Patrick


# build the model and fit the training data
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
#Creates the bar chart comparison
df_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

##We are using mean and Root mean squared error for measurements of accuracy for our Models as 
##we are solving a regression problem.
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Using the method from the tutorials, It produces a line graph of the first 100 predictions 
#compared with the first 100 actual results.
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

#### RESULT of model
#as an unoptimized model using an unclean dataset, it produced extremely inaccurate results.
#The RMSE usually in the ranges of 50,000 to 70,000 Where we are targeting an RMSE of 15% (or under) of the mean.
#which in our case is usually around 25,000 to 28,000
