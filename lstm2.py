import pandas as pd 
import pandas as pd  
import pandas as pd  
import numpy as np  
import numpy as np
import torch
from torch import nn
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data   
import sys
import os
import csv
import scipy.io as sio       
import matplotlib.pyplot as plt        
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras 
from keras.layers import BatchNormalization
from keras.models import Model
import keras.api._v2.keras as keras
from keras.layers import Input, LSTM, Dense, Bidirectional, Conv1D, ReLU, TimeDistributed

ppg = pd.read_csv('preprocessed/ppg_pp.csv')
dbp = pd.read_csv('preprocessed/dbp_pp.csv')
sbp = pd.read_csv('preprocessed/sbp_pp.csv')
abp = pd.read_csv('preprocessed/abp_pp.csv')
ecg = pd.read_csv('preprocessed/ecg_pp.csv')
abp = np.divide(np.subtract(abp, 50), 150)


X_train = ppg[:3500]
X_val = ppg[3500:4500]
X_test = ppg[4500:]

y_train = dbp[:3500]
y_val = dbp[3500:4500]
y_test = dbp[4500:]

for i in y_train:
    for j in i:
        print(j)

plt.show()
inputs = Input(shape=(1024, 1))

encoder = LSTM(512, return_sequences=True)(inputs)
decoder = LSTM(512, return_sequences=True)(encoder)
sequence_prediction = TimeDistributed(Dense(1, activation='linear'))(decoder)

model = Model(inputs, sequence_prediction)
model.compile('adam', 'mae')
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))
model.evaluate((X_test),y_test)

prediction = model.predict(X_test)

plt.plot(y_test.iloc[10])
plt.plot(prediction[10])
plt.savefig('graph.png')
plt.show()
print("Mean absolute error : " + str(error))