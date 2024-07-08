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

def create_rolling_window_dataset(dataset, timestep):
    """function to create input sequence and corresponding output value"""
    dataX, dataY = [], []
    for i in range(len(dataset) - timestep):
        dataX.append(dataset[i: (i+timestep)])
        dataY.append(dataset[(i+timestep)])
    return np.array(dataX), np.array(dataY)

def lstm_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units,return_sequences = False,input_shape = (X_train.shape[1],1)))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error',optimizer = optimizer)
    return model

def calculate_mape(Y_true, Y_pred):
    return np.mean(np.abs((Y_true-Y_pred)/Y_true)) * 100

def plot_true_predict(stock_data, stock_data_close, timestep, scaler, train_predict, test_predict):
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

def predict_future(model, recent_data, timestep, scaler):
    predictions = []
    input_data = recent_data[-timestep:].reshape(1,timestep,1)
    
    for _ in range(timestep):
        next_pred = model.predict(input_data) 
        predictions.append(next_pred)
        input_data = np.append(input_data[:,1:,:], next_pred.reshape((1,1,1)), axis=1)
        
    return scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

stocks = ['^SPX']

# Test on most recent data
end = dt.datetime.today()
start = end - dt.timedelta(360)


data = {}

for ticker in stocks: 
    data[ticker] = yf.download(ticker,start,end)
    
## Add weekends and holidays using forward fill
for stock in stocks:
    data[stock] = data[stock].asfreq("D")
    data[stock].fillna(method='ffill', inplace=True)
    
## Use only close price as feature
stock_data = data['^SPX']
stock_data_close = data['^SPX']['Close']

plt.plot(stock_data_close)

## Data preprocessing
scaler = MinMaxScaler()
stock_data_close = scaler.fit_transform(np.array(stock_data_close).reshape(-1,1))

## Train test split
train_size = .7
train_data, test_data = stock_data_close[: int(len(stock_data_close)*train_size)], stock_data_close[int(len(stock_data_close)*train_size) :]

timestep = 12
X_train, Y_train = create_rolling_window_dataset(train_data, timestep)
X_test, Y_test = create_rolling_window_dataset(test_data, timestep)

## Fit LSTM
### Hyperparameter tunning
model2 = KerasRegressor(model=lstm_model, verbose=0, units=32, random_state=42)

### Define the parameter grid
param_grid = {
    'units': [32, 64, 128],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100]
}

### Perform grid search
grid = GridSearchCV(estimator=model2, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)
print(f'Best {grid_result.best_score_, grid_result.best_params_}')

## Predict on Test set with best parameters
model = lstm_model(units=128, optimizer='adam') # Best (0.7803367735583784, {'batch_size': 32, 'epochs': 100, 'optimizer': 'adam', 'units': 128})
model.fit(X_train,Y_train,validation_data = (X_test,Y_test), epochs=100, batch_size=32) 

train_predict = model.predict(X_train)
train_predict = scaler.inverse_transform(train_predict)
    
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

### Print result
print(math.sqrt(mean_squared_error(scaler.inverse_transform(Y_train),train_predict))) # 43.27050997902599
print(math.sqrt(mean_squared_error(scaler.inverse_transform(Y_test),test_predict))) # 45.13355527829

print(calculate_mape(scaler.inverse_transform(Y_train), train_predict)) # 0.7303206858935632
print(calculate_mape(scaler.inverse_transform(Y_test), test_predict)) # 0.691487795188157

### Plot the true stock data and predicted
plot_true_predict(stock_data, stock_data_close, timestep, scaler, train_predict, test_predict)

## Predict into the future 
future_predictions = predict_future(model, np.array(stock_data_close), timestep, scaler)

futures_dates = pd.date_range(start=stock_data.index[-1] + dt.timedelta(1), periods=timestep, freq='B')
future_predictions = pd.DataFrame(future_predictions, index=futures_dates, columns=['Predicted Close'])

future_predictions.to_csv("future_predictions")