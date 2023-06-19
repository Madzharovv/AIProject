#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing

# Calculating the mean and margin of sale prices as a benchmark for all other models

path = "../Data/Original/"  
filename_read = os.path.join(path, "train.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
mean = df["SalePrice"].mean()
print("Mean " + str(mean))
print ("Margin " + str(mean*0.15))
