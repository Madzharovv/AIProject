# -*- coding: utf-8 -*-


#Document used to test removing outliers
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#pd.options.display.max_columns = None

path = "./Data/Laundried/" 
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
print(df.select_dtypes(include=['int', 'float']))


#originally attempted to remove outliers from every feature using a loop to iterate through each column
#however this resulted in the removal of approximately 1000 entries which would leave us with only *380 records left
#for col in df.select_dtypes(include=['int', 'float']):
#therefore i settled on using it to remove outliers from our target variable the SalePrice
#this removed approximately 100 records, but had a positive effect to the accuracy of most models tested
print(df.shape)
droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (2*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)
print(df.shape)
print(df.select_dtypes(include=['int', 'float']))