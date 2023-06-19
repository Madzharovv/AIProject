# -*- coding: utf-8 -*-

###This document was meant to properly produce an SVR model 

#library imports
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

#load dataset
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
#filename_read = os.path.join("../ipynb/nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])


###patrick found that models may be more accurate when removing the outliers 1 standard deviation away from the mean
droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (1*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)
# omit= ['KitchenQual', 'LandSlope',
# 'SaleType', 'LotShape', 'GarageFinish', 'GarageCond','RoofStyle', 'RoofMatl', 'BsmtFinType1','MSZoning', 'BsmtFinType2']
# for col in omit:
#     df.drop(col,inplace=True,axis=1)
# medians = ["LotFrontage","2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch"]
# for col in medians:
#     med = df[col].median()
#     df[col]=df[col].fillna(med)
  

#preset 1--------------
# X = df[['YrSold','MoSold','LotArea','BedroomAbvGr']].values.astype(np.float32)
# y = df['SalePrice'].values.astype(np.float32)
#-----------------

#preset 2-----------
Temp = df.select_dtypes(include=['int', 'float'])

X = Temp.iloc[:,:-1].values
y = Temp.iloc[:,-1].values

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
# # for col in df.columns:
# #     mean=df[col].mean()
# #     sd=df[col].std()
# #     df[col]=(df[col]-mean)/sd    
# # #-----------------------------------------------
# X = df.iloc[:,:-1].values
# y = df['SalePrice'].values


# #--------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=59)
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit(X_train)
# X_train=sc.transform(X_train)
# X_test=sc.transform(X_test)

###I found that preset 2 had roughly the same accuracy as preset 4 here, the encoded variables appeared
###to very innaccurate results by themselves. Therefore i decided to run the SVR using only the 2nd preset
###as it would reduce the time taken to run.
### Linear was the most accurate kernel used, rbf would be able to nearly be as accurate when using 
### extremely high values of C, however the time taken to perform the model fitting became problematic
### I kept the default C value for the linear kernel as it had the same issue otherwise, with no 
### positive effect

model = SVR(kernel='linear')
# model2 = SVR(kernel='rbf',C=10000000)
# model3 = SVR(kernel='poly')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('RMSE: %.2f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('Mean: %.2f'%np.mean(y_test))

###Overall it produced RMSE values that varied drastically dependent on the Random state
###this ranged in values between 20,000 and 40,000






### These were ghost attempts to try them all simultaneously 

# model2.fit(X_train,y_train)
# y_pred=model2.predict(X_test)
# print('RMSE: %.2f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# print('Mean: %.2f'%np.mean(y_test))
# model3.fit(X_train,y_train)
# y_pred=model3.predict(X_test)
# print('RMSE: %.2f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# print('Mean: %.2f'%np.mean(y_test))

######These are all remnants of ports from Patricks original SVR file. Unsure what their purpose is in terms of SVR

#looks at some outputs side by side
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

#some stats
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#and consider the output
print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))
print('Correlation: ', stats.pearsonr(y_test,y_pred))


#plot the outputs
#set figure size
plt.rc('figure', figsize=(8, 8))

#plot line down the middle
x = np.linspace(550000,0,50)
plt.plot(x, x, '-r')

#plot the points, prediction versus actual
plt.scatter(y_test, y_pred, color='black')

plt.xticks(())
plt.yticks(())

plt.show()

#and plot the values to emphasise the noise
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('SalePrice')
    plt.xlabel("First 150 houses sorted")
    plt.legend()
    plt.show()
    
chart_regression(y_pred[:150],y_test[:150],sort=True)  