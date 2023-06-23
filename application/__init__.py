
import io
import keras.layers
import os
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, Response, send_file, json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


def get_historical_data(symbol, start_date = None):
    api_key = open(r'api_key.txt')
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
    df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1)
    if start_date:
        df = df[df.index >= start_date]
    df.to_csv("Data/%s.csv" % symbol)
    return df

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    ticker = data['ticker']
    valid_tickers = ['MVRS', 'AMZN', 'AAPL']
    if ticker not in valid_tickers:
        return Response(json.dumps(
            {'error': 'Invalid ticker symbol. Please enter MVRS for Meta, AMZN for Amazon, or AAPL for Apple.'}),
                        status=400)
    df = get_historical_data(ticker, "2019-01-01")
    average = np.average(df['open'])

    response = {
        'ticker': ticker,
        'predicted_stock_price': average
    }
    return Response(json.dumps(response))

@app.route('/accuracy', methods=['GET', 'POST'])
def accuracy():
    # to generate 'actual_stock_price' and 'predicted_stock_price'
    data = request.get_json(force=True)
    ticker = data['ticker']
    df = get_historical_data(ticker, "2019-01-01")
    dataset_train = df
    training_set = dataset_train.iloc[:,0:1].values # open
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_set = scaler.fit_transform(training_set)

    X_train = []
    y_train = []
    last_train_date = training_set.shape[0]-30

    for i in range(60, last_train_date):
       X_train.append(scaled_training_set[i-60:i, 0])
       y_train.append(scaled_training_set[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences= True, input_shape= (X_train.shape[1], 1)))
    regressor.add(Dropout(rate=0.2))
    regressor.add(LSTM(units = 50, return_sequences= True))
    regressor.add(Dropout(rate=0.2))
    regressor.add(LSTM(units = 50, return_sequences= True))
    regressor.add(Dropout(rate=0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(rate=0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 2, batch_size = 32)

    dataset_test = training_set[-30:]
    actual_stock_price = dataset_test[:,0]
    inputs =dataset_train['open'][len(dataset_train['open']) - len(dataset_test)-60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, 90):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # # Building the plot
    # plt.plot(actual_stock_price, color='red', label='Actual Stock Price')
    # plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    # plt.title('Stock Price Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Stock Price')
    # plt.legend()
    #
    # # Save it to a BytesIO object
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    mape = np.mean(np.abs((actual_stock_price - predicted_stock_price) / actual_stock_price)) * 100
    rmse = sqrt(mean_squared_error(actual_stock_price, predicted_stock_price))
    response = {
        'ticker': ticker,
        #'predicted_stock_price': predicted_stock_price.tolist(),
        'accuracy MAPE': 100 - mape,  # Accuracy is 100 - MAPE
        'accuracy RMSE' : rmse
    }
    return Response(json.dumps(response))


if __name__ == "__main__":
    app.run()
