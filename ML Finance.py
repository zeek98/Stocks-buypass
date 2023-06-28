#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the ticker symbols for the stocks
ticker_symbols = ["AAPL", "MSFT", "NFLX"]

#Historical data from Yahoo Finance
stock_data = yf.download(ticker_symbols, start="2010-01-01", end="2023-06-16")

#Preprocess the data
for ticker_symbol in ticker_symbols:
    stock_data[f"{ticker_symbol}_Return"] = stock_data["Close"][ticker_symbol].pct_change()  # Calculate the daily return
stock_data.dropna(inplace=True)  # Remove NaN values

#Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[[f"{ticker_symbol}_Return" for ticker_symbol in ticker_symbols]])

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Length of input sequence
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

#LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(ticker_symbols))))
model.add(LSTM(50, activation='relu'))
model.add(Dense(len(ticker_symbols)))
model.compile(optimizer='adam', loss='mse')

#Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

#Predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values to their original scales
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

#The best stock to invest in based on the maximum predicted value for each day
best_stock_symbols = [ticker_symbols[np.argmax(y_pred_inv[i])] for i in range(len(y_pred_inv))]

#The percentage comparison for each day and the best stock to invest in
for i, stock_symbol in enumerate(best_stock_symbols):
    percentage = np.max(y_pred_inv[i]) / (len(ticker_symbols) - 1) * 100
    print(f"Day {i+1}: Percentage Comparison - {percentage:.2f}%, Best Stock: {stock_symbol}")

#The overall best stock to invest in based on the most frequent occurrence in the best_stock_symbols list
overall_best_stock = max(set(best_stock_symbols), key=best_stock_symbols.count)

# Print the overall best stock to invest in
print("The overall best stock to invest in is:", overall_best_stock)


# In[ ]:


import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#The ticker symbols for the stocks
ticker_symbols = ["AAPL", "MSFT", "NFLX"]

#Historical data from Yahoo Finance
stock_data = yf.download(ticker_symbols, start="2010-01-01", end="2023-06-16")

# Preprocess
for ticker_symbol in ticker_symbols:
    stock_data[f"{ticker_symbol}_Return"] = stock_data["Close"][ticker_symbol].pct_change()  # Calculate the daily return
stock_data.dropna(inplace=True)  # Remove NaN values

#The standard deviation of each stock's returns
volatility = {}
for ticker_symbol in ticker_symbols:
    volatility[ticker_symbol] = stock_data[f"{ticker_symbol}_Return"].std()

#The stock with the lowest volatility
safest_stock = min(volatility, key=volatility.get)

#Volatility of each stock
for ticker_symbol in ticker_symbols:
    print(f"Volatility of {ticker_symbol}: {volatility[ticker_symbol]:.4f}")

#Safest stock to invest in
print("The safest stock to invest in is:", safest_stock)

