# -*- coding: utf-8 -*-

### Scrapped file, intended to be used as the original neural network to be built using tutorials.
import tensorflow as tf
from tensorflow import keras

import base64
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing

def chart_regression(pred,y,sort=True):
    t=pd.DataFrame({'pred':pred,'y':y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    plt.plot(t['y'].tolist(),label="expected")
    plt.plot(t['pred'].tolist(),label="prediction")
    plt.ylabel('output')
    plt.legend()
    plt.show()
             