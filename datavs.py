# -*- coding: utf-8 -*-

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


path = "./Data/Original/" 
filename_read = os.path.join(path, "train.csv")
pd.options.display.max_columns = None

#Base for presets
df = pd.read_csv(filename_read)

plt.scatter(df["Id"],df["SalePrice"],c="red")
plt.ylabel("Sale Price")
plt.xlabel("HouseID")

plt.show()