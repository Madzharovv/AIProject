# -*- coding: utf-8 -*-

###This is what the NN.py file became. It was easier to produce a new file as i could use the older one to dump unnecessary
###or prototypes.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

###As this is the first file in a list of iterations i didn't use any data manipulation to have a raw base result

path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

###Using Preset1 as a default to ensure the model could operate

X = df[['YrSold','MoSold','LotArea','BedroomAbvGr']].values.astype(np.float32)
y = df['SalePrice'].values.astype(np.float32)

###As an initial Test i implemented a two hidden-layer machine that used 'RELU' activation.
###All other activations tested would not produce an RMSE under 100,000. So it was clear the data being used
###would not work well for those activations.
###i decided to use a value such as 64 to measure how increasing the amount of units would affect the machine
###i found that a larger value, approximately 100 would produce most optimal results, with exponentially less benefit
###as the units is increased.
###The adam optimizer was used as it produced far more reliable and accurate results than the rest of the optimizers listed
###The only other optimizers found were rmsprop, and adamax. even though they were still less effective than adam
###Verbose 2 was used for testing purposes.
###Using roughly 200 epochs appeared to produce the best results as the loss would begin to jump about above 150 epochs.
###This could be due to a high number of units, however using a lower amount would reduce the accuracy of the model in testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=32)
model=Sequential()
model.add(Dense(64,input_shape=X[1].shape,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam') #rmsprop,adam,adamax,
model.fit(X_train,y_train,verbose=2,epochs=200)
model.summary()
pred=model.predict(X_test)
print("Shape: {}".format(pred.shape))
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"final score (RMSE): {score}")


###as a result, for this preset it would produce an RMSE of 70549.88 This is unacceptable and would not be approved
###for use
