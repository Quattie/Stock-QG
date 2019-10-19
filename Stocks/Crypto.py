import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.random import seed
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

from plotly.offline import plot
import plotly.graph_objs as go

import matplotlib
matplotlib.use('Agg')

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


class Crypto:

    def __init__(self, coin):
        self.coin = coin
        if coin == 'bitcoin':
            self.tag = 'btc'
            print('you typed bitcoin')
        elif coin == 'ethereum':
            self.tag = 'eth'
        elif coin == 'litecoin':
            self.tag = 'ltc'
        elif coin == 'ltc':
            self.coin = 'litecoin'
            self.tag = 'ltc'
        elif coin == 'eth':
            self.coin = 'ethereum'
            self.tag = 'eth'
        elif coin == 'btc':
            self.coin = 'bitcoin'
            self.tag = 'btc'
        else:
            raise ValueError(
                'We only support Bitcoin, Ethereum and Litecoin right now')

    def get_market_data(self, market, tag=True):
        """
        market: the full name of the cryptocurrency as spelled on coinmarketcap.com. eg.: 'bitcoin'
        tag: eg.: 'btc', if provided it will add a tag to the name of every column.
        returns: panda DataFrame
        This function will use the coinmarketcap.com url for provided coin/token page.
        Reads the OHLCV and Market Cap.
        Converts the date format to be readable.
        Makes sure that the data is consistant by converting non_numeric values to a number very close to 0.
        And finally tags each columns if provided.
        """
        now = datetime.now()
        market_data = pd.read_html("https://coinmarketcap.com/currencies/" + market +
                                   "/historical-data/?start=20130428&end="+now.strftime("%Y%m%d"), flavor='html5lib')[0]
        market_data = market_data.assign(
            Date=pd.to_datetime(market_data['Date']))
        market_data['Volume'] = (pd.to_numeric(
            market_data['Volume'], errors='coerce').fillna(0))
        if tag:
            market_data.columns = [market_data.columns[0]] + \
                [tag + '_' + i for i in market_data.columns[1:]]
        return market_data

    def get_data(self):

        data = self.get_market_data(self.coin, self.tag)
        print(data.head())
        return data

    def get_crypto_candlestick(self, data):

        data = data.iloc[::-1]
        trace1 = go.Candlestick(
            x=data.index,
            open=data[data.columns[2]],
            high=data[data.columns[3]],
            low=data[data.columns[4]],
            close=data[data.columns[5]]
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

    def make_history_chart(self, data):
        crypt_fig = plt.figure()
        plt.style.use('dark_background')
        plt.plot(data["Date"], data[self.tag + "_Close**"])
        plt.title('History of Crypto')
        plt.ylabel('Price (USD)')
        plt.xlabel('Date')
        plt.legend(['Close'], loc='upper left')
        crypt_fig.autofmt_xdate()
        plt.savefig("static/Stocks/charts/CryptoHistory.png")

    def make_model(self, data):
        # Make training set 80% of the data
        train_cols = [self.tag + '_Open*', self.tag + '_High', self.tag + '_Low',
                      self.tag + '_Close**', self.tag + '_Volume', self.tag + '_Market Cap']
        btc_train, btc_test = train_test_split(
            data, train_size=0.8, test_size=0.2, shuffle=False)
        x = btc_train.loc[:, train_cols].values
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x)
        x_test = scaler.fit_transform(data.loc[:, train_cols])

        x_t, y_t = build_timeseries(x_train, 3)
        x_t = trim_data(x_t, BATCH_SIZE)
        y_t = trim_data(y_t, BATCH_SIZE)
        x_temp, y_temp = build_timeseries(x_test, 3)
        x_val, x_test_t = np.split(trim_data(x_temp, BATCH_SIZE), 2)
        y_val, y_test_t = np.split(trim_data(y_temp, BATCH_SIZE), 2)
        print(x_test_t.shape)

        model = Sequential()
        model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0,
                       recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()

        btc_history = model.fit(x_t, y_t, epochs=30, verbose=2,
                                batch_size=BATCH_SIZE, shuffle=False,
                                validation_data=(trim_data(x_val, BATCH_SIZE), trim_data(y_val, BATCH_SIZE)))

        y_pred = model.predict(
            trim_data(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        print(x_test_t.shape)
        y_pred - y_pred.flatten()
        y_test_t = trim_data(y_test_t, BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is ", error, y_pred.shape, y_test_t.shape)

        y_pred_org = (y_pred * scaler.data_range_[3] + scaler.data_min_[3])
        y_test_org_t = (y_test_t * scaler.data_range_[3] + scaler.data_min_[3])

        print(y_pred_org.shape)
        print(y_test_org_t.shape)

        model_fig = plt.figure()
        plt.plot(btc_history.history['loss'])
        plt.plot(btc_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'])
        ax = plt.gca()
        ax.invert_xaxis()
        model_fig.savefig("static/Stocks/charts/ModelLossCrypto.png")

        accuracy_fig = plt.figure()
        plt.plot(y_pred_org)
        plt.plot(y_test_org_t)
        plt.title('Prediction vs. Real Stock Price')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend(['Prediction', 'Real'])
        ax = plt.gca()
        ax.invert_xaxis()
        accuracy_fig.savefig(
            "static/Stocks/charts/PredictionVSReal.png")

        next_pred = data[-7:]
        train_cols = [self.tag + '_Open*', self.tag + '_High', self.tag + '_Low',
                      self.tag + '_Close**', self.tag + '_Volume', self.tag + '_Market Cap']
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

        print(next_pred)
        print(y_fut)
        print(fut_pred_org)
        return float("{0:.2f}".format(fut_pred_org[0][-1]))
