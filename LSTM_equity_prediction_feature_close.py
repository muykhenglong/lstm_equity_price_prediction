#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:02:38 2024

LSTM - Predict Equity Price using Close as Feature

@author: Muykheng Long
"""

import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import random
import tensorflow as tf

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

stocks = ['AMZN','MSFT','GOOG']

start = dt.datetime.today() - dt.timedelta(360)
end = dt.datetime.today()

data = {}

for ticker in stocks: 
    data[ticker] = yf.download(ticker,start,end)
    
# Data preprocessing
## add weekends and holidays using forward fill
for stock in stocks:
    data[stock] = data[stock].asfreq("D")
    data[stock].fillna(method='ffill', inplace=True)
    
# Work on AMZN first

stock_data = data['AMZN']
stock_data_close = data['AMZN']['Close']

plt.plot(stock_data_close)

# Data preprocessing
scaler = MinMaxScaler()
stock_data_close = scaler.fit_transform(np.array(stock_data_close).reshape(-1,1))

# Train test split
def create_rolling_window_dataset(dataset, timestep):
    """function to create input sequence and corresponding output value"""
    dataX, dataY = [], []
    for i in range(len(dataset) - timestep):
        dataX.append(dataset[i: (i+timestep)])
        dataY.append(dataset[(i+timestep)])
    return np.array(dataX), np.array(dataY)

train_data, test_data = stock_data_close[: int(len(stock_data_close)*.7)], stock_data_close[int(len(stock_data_close)*.7) :] # train size of 70%

timestep = 12
X_train, Y_train = create_rolling_window_dataset(train_data, timestep)
X_test, Y_test = create_rolling_window_dataset(test_data, timestep)

# Fit LSTM
def lstm_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units,return_sequences = False,input_shape = (X_train.shape[1],1)))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error',optimizer = optimizer)
    return model

# Hyperparameter tunning
model2 = KerasRegressor(model=lstm_model, verbose=0, units=32, random_state=42)


# Define the parameter grid
param_grid = {
    'units': [32, 64, 128],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100]
}

# Perform grid search
grid = GridSearchCV(estimator=model2, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)


print(f'Best {grid_result.best_score_, grid_result.best_params_}')

# Predict with best parameters
model = lstm_model(units=128, optimizer='adam') # Best (0.7016766874781273, {'batch_size': 32, 'epochs': 100, 'optimizer': 'adam', 'units': 128})
model.fit(X_train,Y_train,validation_data = (X_test,Y_test), epochs=100, batch_size=32) 

train_predict = model.predict(X_train)
train_predict = scaler.inverse_transform(train_predict)
    
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

print(math.sqrt(mean_squared_error(scaler.inverse_transform(Y_train),train_predict))) # 2.932386838446456
print(math.sqrt(mean_squared_error(scaler.inverse_transform(Y_test),test_predict))) # 2.93277721255082


# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(Y_true, Y_pred):
    return np.mean(np.abs((Y_true-Y_pred)/Y_true)) * 100

mape = calculate_mape(scaler.inverse_transform(Y_test), test_predict)
print(mape) # 1.1810387195167207

# Plot the true stock data and predicted
plt.plot(stock_data.index, scaler.inverse_transform(stock_data_close), label='True Data')

train_predict_plot = np.empty_like(stock_data_close)
train_predict_plot[:] = np.nan
train_predict_plot[timestep:int(len(stock_data_close)*0.7)] = train_predict

test_predict_plot = np.empty_like(stock_data_close)
test_predict_plot[:] = np.nan
test_predict_plot[int(len(stock_data_close)*0.7) + timestep:] = test_predict

plt.plot(stock_data.index, train_predict_plot, label='Train Predict', color='orange')
plt.plot(stock_data.index, test_predict_plot, label='Test Predict', color='green')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()