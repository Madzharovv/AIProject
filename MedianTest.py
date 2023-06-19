# -*- coding: utf-8 -*-

##Test Code to see if adding some medians to some numeric fields may 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#pd.options.display.max_columns = None
pd.options.display.max_rows = None
path = "./Data/Laundried/" 
filename_read = os.path.join(path, "ZerosNulled.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

medians = ["LotFrontage","2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch"]
medianValues=[]



### I tried to select features that were not logically producing NA/0 values, however with this dataset it was
### the way the dataset was laid out, so i tried to select less relevant features to add medians to.
print(medianValues)
print(df["LotFrontage"])
for col in medians:
    med = df[col].median()
    df.loc[df[col]==0,col]=med
    medianValues.append(med)
print(df["LotFrontage"])
print(medianValues)
##MedianValues produced were [69,776,377.5,171,63,144.5]