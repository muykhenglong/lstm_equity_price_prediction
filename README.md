# LSTM - Predict Equity Price using Close as Feature

This Python script demonstrates the use of a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical closing prices. The script leverages the Yahoo Finance API (via yfinance) to fetch historical stock data, preprocesses the data, trains an LSTM model, and evaluates its performance using various metrics.

## Features

Automated Data Fetching: Retrieves historical stock prices for specified stocks (AMZN, MSFT, GOOG).
Data Preprocessing: Adds missing data for weekends and holidays using forward fill, and normalizes the closing prices.
Dynamic LSTM Model Training: Utilizes GridSearchCV for hyperparameter tuning to optimize the LSTM model.
Performance Metrics: Computes Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) to evaluate the model's performance.
Visualization: Plots true stock prices against the model's predicted prices for visual comparison.

## Requirements

Python 3.8.12

## How It Works

Data Fetching: Downloads one year of daily stock data for AMZN, MSFT, and GOOG.
Data Preprocessing: Fills missing values for weekends and holidays using forward fill and normalizes the closing prices using MinMaxScaler.
Train-Test Split: Splits the normalized data into training (70%) and testing (30%) sets, and creates rolling window datasets for LSTM input.
LSTM Model Training: Uses GridSearchCV to find the best hyperparameters for the LSTM model. The model is then trained using the best parameters.
Performance Evaluation: Predicts the stock prices using the trained model, calculates RMSE and MAPE, and plots the true vs. predicted stock prices.

# Overview
## Long Short-Term Memory (LSTM) Architecture
LSTM networks are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies. They are designed to avoid the long-term dependency problem, making them suitable for time series forecasting.

## LSTM Cell Structure
An LSTM cell consists of three gates:

* Forget Gate (f_t): Decides what information to throw away from the cell state.
* Input Gate (i_t): Decides which values from the input to update the cell state.
* Output Gate (o_t): Decides what to output based on the cell state.
  
* The equations governing an LSTM cell are:
* Forget gate:
  $f_t = \sigma(W_f*[h_{t-1}, x_t] + b_f)$
* Input gate:
  $i_t = \sigma(W_i*[h_{t-1}, x_t] + b_i)$
  $\hat{C_t} = tanh(W_C*[h_{t-1}, x_t] + b_c)$
* Cell state update:
  $C_t = f_t * C_{t-1} + i_t * \hat{C_t}$
* Output gate:
  $o_t = \sigma(W_o*[h_{t-1}, x_t] + b_o)$
  $h_t = o_t * tanh(C_t)$

where 
* $\sigma$ is the sigmoid function.
* $x_t$ is the input at time step t.
* $h_{t-1}$ is the hidden state from the previous time step.
* $C_{t-1}$ is the cell state from the previous time step.
* $W_f, W_i, W_C, W_o$ are the weight matrices.
* $b_f, b_i, b_C, b_o$ are the bias vectors.
