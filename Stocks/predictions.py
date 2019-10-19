from keras import optimizers
from keras.utils import plot_model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from tensorflow import set_random_seed
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pylab import rcParams
from numpy.random import seed
import yfinance as yf
import numpy as np
import pandas as pd
import time
import math
import os
from datetime import date

from plotly.offline import plot
import plotly.graph_objs as go


# How many days in the past that we want to look at
TIME_STEPS = 5
BATCH_SIZE = 2


def build_timeseries(mat, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    print("Length of time-series i/o", x.shape, y.shape)
    return x, y


def trim_data(mat, batch_size):
    num_rows_dropped = mat.shape[0] % batch_size
    if(num_rows_dropped > 0):
        return mat[:-num_rows_dropped]
    else:
        return mat


class Prediction():

    def get_data(self, user_ticker):
        ticker = yf.Ticker(user_ticker)
        if ticker:
            today = date.today()
            formed_date = today.strftime("%Y-%m-%d")
            stock = ticker.history(period='max')
            stock = stock.loc['2015-09-01':formed_date]
            return stock
        else:
            return 'Please enter a valid ticker'

    def get_history_candlestick(self, stocks):

        trace1 = go.Candlestick(
            x=stocks.index,
            open=stocks.Open,
            high=stocks.High,
            low=stocks.Low,
            close=stocks.Close
        )
        layout = go.Layout(
            autosize=True,
            xaxis=dict(
                autorange=True
            ),
            yaxis=dict(
                autorange=True
            )
        )
        plot_data = [trace1]
        figure = go.Figure(data=plot_data, layout=layout)
        plot_div = plot(figure, output_type='div', include_plotlyjs=False)
        return plot_div

    def make_train_test(self, stock):
        train_cols = ['Open', 'High', 'Close', 'Low', 'Volume']
        stock_train, stock_test = train_test_split(
            stock, train_size=0.8, test_size=0.2, shuffle=False)

        x = stock_train.loc[:, train_cols].values
        scaler = MinMaxScaler()
        test = stock_test.loc[:, train_cols]
        x_train = scaler.fit_transform(x)
        x_test = scaler.fit_transform(test)

        return x_train, x_test, scaler

    def make_test_and_val(self, x_train, x_test):
        x_t, y_t = build_timeseries(x_train, 3)
        x_t = trim_data(x_t, BATCH_SIZE)
        y_t = trim_data(y_t, BATCH_SIZE)
        x_temp, y_temp = build_timeseries(x_test, 3)
        x_val, x_test_t = np.split(trim_data(x_temp, BATCH_SIZE), 2)
        y_val, y_test_t = np.split(trim_data(y_temp, BATCH_SIZE), 2)
        return x_t, y_t, x_val, y_val, x_test_t, y_test_t

    def train_model(self, _optimizer, x_t, y_t, x_val, y_val, _epochs):
        model = Sequential()
        model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0,
                       recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer=_optimizer)
        stock_history = model.fit(x_t, y_t, epochs=_epochs, verbose=2,
                                  batch_size=BATCH_SIZE, shuffle=False,
                                  validation_data=(trim_data(x_val, BATCH_SIZE), trim_data(y_val, BATCH_SIZE)))
        return model, stock_history

    def unscale_data(self, model, x_test_t, y_test_t, scaler):
        y_pred = model.predict(
            trim_data(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred - y_pred.flatten()
        y_test_t = trim_data(y_test_t, BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)

        # 3 being the amount of y_cols
        y_pred_org = (y_pred * scaler.data_range_[3] + scaler.data_min_[3])
        y_test_org_t = (y_test_t * scaler.data_range_[3] + scaler.data_min_[3])

        return y_pred_org, y_test_org_t

    def make_model_loss_chart(self, stock_history, name):
        trace1 = go.Scatter(
            y=stock_history.history['val_loss'],
            mode='lines',
            name='Model Accuracy'
        )
        trace2 = go.Scatter(
            y=stock_history.history['loss'],
            mode='lines',
            name='Test Data'
        )
        layout = go.Layout(
            autosize=True,
            xaxis=dict(
                autorange=True
            ),
            yaxis=dict(
                autorange=True
            )
        )
        plot_data = [trace1, trace2]
        figure = go.Figure(data=plot_data, layout=layout)
        plot_div = plot(figure, output_type='div', include_plotlyjs=False)
        return plot_div

    def make_prediction_chart(self, y_pred_org, y_test_org_t, name):

        pred = []
        for x in y_pred_org:
            pred.append(x[0])
        y_test_org_t.tolist()
        trace1 = go.Scatter(
            y=pred,
            mode='lines',
            name='Predictions'
        )
        trace2 = go.Scatter(
            y=y_test_org_t,
            mode='lines',
            name='Actual Values'
        )
        layout = go.Layout(
            autosize=True,
            xaxis=dict(
                autorange=True
            ),
            yaxis=dict(
                autorange=True
            )
        )
        plot_data = [trace1, trace2]
        figure = go.Figure(data=plot_data, layout=layout)
        plot_div = plot(figure, output_type='div', include_plotlyjs=False)
        return plot_div

    def predict_tomorrows_price(self, stock, model, scaler, y_test_t):
        next_pred = stock[-7:]
        train_cols = ['Open', 'High', 'Close', 'Low', 'Volume']
        x_pred_values = next_pred.loc[:, train_cols].values
        predx = scaler.fit_transform(x_pred_values)
        x_fut, y_fut = build_timeseries(predx, 3)

        y_test_t = trim_data(y_test_t, BATCH_SIZE)

        future_pred = model.predict(
            trim_data(x_fut, BATCH_SIZE), batch_size=BATCH_SIZE)

        future_pred - future_pred.flatten()
        y_fut = trim_data(y_fut, BATCH_SIZE)

        fut_pred_org = (
            future_pred * scaler.data_range_[3] + scaler.data_min_[3])
        y_fut = (y_fut * scaler.data_range_[3] + scaler.data_min_[3])
        return float("{0:.2f}".format(fut_pred_org[0][-1]))
