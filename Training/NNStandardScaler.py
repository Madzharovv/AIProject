# -*- coding: utf-8 -*-

###This document was to test whether implementing Standard Scaling would work on the NN
###This was then ported to the NNPresets file.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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

X = df[['YrSold','MoSold','LotArea','BedroomAbvGr']].values.astype(np.float32)
y = df['SalePrice'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=32)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train=sc.transform(X_train)
# X_test=sc.transform(X_test)

model=Sequential()
model.add(Dense(104,input_shape=X[1].shape,activation='relu'))
model.add(Dense(104,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam') #rmsprop,adam,adamax,
model.fit(X_train,y_train,verbose=2,epochs=400)
model.summary()
pred=model.predict(X_test)
print("Shape: {}".format(pred.shape))
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"final score (RMSE): {score}")