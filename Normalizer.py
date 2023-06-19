# -*- coding: utf-8 -*-

#This is a document used to convert numerical fields to a normalised format.
#originally implemented for use with the Neural Network, However it seemed to have little effect on all models tested

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

#By iterating through each numerical column, it calculates their numeric z-score
#this is done using code from the tutorials. Ex6 part1 and Ex7 part2
for col in df.select_dtypes(include=['int', 'float']).columns:
    mean=df[col].mean()
    sd=df[col].std()
    df[col]=(df[col]-mean)/sd
    
print(df.select_dtypes(include=['int', 'float']))