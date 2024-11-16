#!/usr/bin/env python
# coding: utf-8

# 1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, LSTM
from sklearn.metrics import mean_absolute_error
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

# 2. Loading and Preprocessing Data
model_type = 'LSTM'  # Set to 'LSTM' for LSTM or 'GRU' for GRU
nifty_it = yf.download('^CNXIT', start='2010-01-01', end='2024-11-10', interval='1d')
nifty_it = nifty_it[['Close']].fillna(method='ffill')

# 3. Exploratory Data Analysis (EDA)
nifty_it.plot(figsize=(20, 7))
plt.xlabel('Date')
plt.ylabel(f'Nifty Close Price')
plt.title('Nifty IT Close Price Over Time')
plt.show()

results = seasonal_decompose(nifty_it['Close'], model='additive', period=253)
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(nifty_it['Close'], label='Original')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(results.trend, label='Trend')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(results.seasonal, label='Seasonal')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(results.resid, label='Residual')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Train-Test Split and Scaling
split_date_str = '2018-05-06'  # Replace this with user input if needed
split_date = pd.to_datetime(split_date_str)

train = nifty_it[nifty_it.index <= split_date]
test = nifty_it[nifty_it.index > split_date]

# Ensure enough data for predictions
if len(test) < 30:
    print("Warning: Not enough data for 30 days of predictions. Using available data.")
    prediction_length = len(test)
else:
    prediction_length = 30
    test = test.iloc[:30]

# Scaling
scaler = MinMaxScaler()
scaler.fit(nifty_it[['Close']])
scaled_train = scaler.transform(train[['Close']])
scaled_test = scaler.transform(test[['Close']])

# 5. Data Preparation for LSTM/GRU
def create_sequences(data, n_input):
    X, y = [], []
    for i in range(len(data) - n_input):
        X.append(data[i:i + n_input])
        y.append(data[i + n_input])
    return np.array(X), np.array(y)

n_input = 3  # Sequence length for training
X_train, y_train = create_sequences(scaled_train, n_input)
X_test, y_test = create_sequences(scaled_test, n_input)

# Reshape inputs for LSTM/GRU
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 6. Building the LSTM/GRU Model
model = Sequential()
if model_type == 'LSTM':
    model.add(LSTM(64, return_sequences=True, input_shape=(n_input, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
elif model_type == 'GRU':
    model.add(GRU(64, return_sequences=True, input_shape=(n_input, 1)))
    model.add(Dropout(0.2))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(64))
    model.add(Dropout(0.2))
else:
    raise ValueError("Invalid model type. Choose 'LSTM' or 'GRU'.")

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 7. Training the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# 8. Model Evaluation and Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Loss: {train_loss}")
print(f"Testing Loss: {test_loss}")

# 9. Function to Predict the Next Day
def predict_next_day(model, data, scaler, n_steps):
    if len(data) < n_steps:
        raise ValueError(f"Not enough data. Required: {n_steps}, Available: {len(data)}")
    input_seq = np.array(data[-n_steps:]).reshape((1, n_steps, 1))
    next_day_scaled = model.predict(input_seq, verbose=0)
    next_day_price = scaler.inverse_transform(next_day_scaled)
    return next_day_price[0, 0]

# Predict the next day price
next_day_price = predict_next_day(model, scaled_train, scaler, n_input)
print(f"Predicted price for the next day: {next_day_price}")

