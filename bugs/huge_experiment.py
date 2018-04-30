import pandas as pd
import numpy as np

from comet_ml import Experiment

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, GRU, LSTM, Flatten, TimeDistributed
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Comet-ml
experiment = Experiment(api_key="DpTmAn1n7JHUJQDSRS7WRPbm1", project_name="CometTest-RNN", log_code=True)

# trainXUnscaled = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
trainXUnscaled = pd.DataFrame(np.random.random(size=(1200000, 28)))
trainYUnscaled = pd.DataFrame(np.random.random(size=(1200000, 2)))
testXUnscaled = pd.DataFrame(np.random.random(size=(250000, 28)))
testYUnscaled = pd.DataFrame(np.random.random(size=(250000, 2)))

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

trainX = scaler.fit_transform(trainXUnscaled)
trainY = scaler.fit_transform(trainYUnscaled)
testX = scaler.fit_transform(testXUnscaled)
testY = scaler.fit_transform(testYUnscaled)

# create and fit the LSTM network
actions = ['Long', 'Short']
numActions = len(actions)
numFeatures = 28
inputShape = (1, 1, numFeatures)

# Input: 3D tensor with shape (batch_size, timesteps, input_dim).
model = Sequential()
model.add(
    LSTM(200, dropout=0.2, recurrent_dropout=0.2, batch_input_shape=inputShape, stateful=True, return_sequences=True))
# model.add(GRU(60, dropout = 0.2, recurrent_dropout = 0.2, batch_input_shape = inputShape, stateful = True, return_sequences = True))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(numActions))
model.add(Dropout(0.25))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=.00025), metrics=['accuracy'])

trainXLSTMShaped = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testXLSTMShaped = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model.fit(trainXLSTMShaped, trainY, epochs=2, batch_size=1, shuffle=False, verbose=1,
          validation_data=(testXLSTMShaped, testY))
