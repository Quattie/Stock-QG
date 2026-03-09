import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from tqdm.auto import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM

from plotly.offline import plot
import plotly.graph_objs as go

TIME_STEPS = 5
BATCH_SIZE = 2

# Map common names/tags to yfinance crypto tickers
CRYPTO_MAP = {
    'bitcoin': 'BTC-USD', 'btc': 'BTC-USD',
    'ethereum': 'ETH-USD', 'eth': 'ETH-USD',
    'litecoin': 'LTC-USD', 'ltc': 'LTC-USD',
    'solana': 'SOL-USD', 'sol': 'SOL-USD',
    'dogecoin': 'DOGE-USD', 'doge': 'DOGE-USD',
    'cardano': 'ADA-USD', 'ada': 'ADA-USD',
    'xrp': 'XRP-USD', 'ripple': 'XRP-USD',
}


def build_timeseries(mat, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    for i in tqdm(range(dim_0)):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    return x, y


def trim_data(mat, batch_size):
    num_rows_dropped = mat.shape[0] % batch_size
    if num_rows_dropped > 0:
        return mat[:-num_rows_dropped]
    return mat


class Crypto:

    def __init__(self, coin):
        coin_lower = coin.strip().lower()
        if coin_lower in CRYPTO_MAP:
            self.ticker_symbol = CRYPTO_MAP[coin_lower]
            self.display_name = coin_lower.capitalize()
        elif coin.upper().endswith('-USD'):
            self.ticker_symbol = coin.upper()
            self.display_name = coin.upper().replace('-USD', '')
        else:
            # Try as-is with -USD appended
            self.ticker_symbol = f"{coin.upper()}-USD"
            self.display_name = coin.upper()

    def get_data(self):
        ticker = yf.Ticker(self.ticker_symbol)
        data = ticker.history(period='max')
        return data

    def get_crypto_info(self):
        """Fetch crypto metadata for dashboard."""
        ticker = yf.Ticker(self.ticker_symbol)
        info = ticker.info
        return {
            'name': info.get('shortName', self.display_name),
            'symbol': self.ticker_symbol,
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice', 'N/A'),
            'previous_close': info.get('previousClose', 'N/A'),
            'day_high': info.get('dayHigh') or info.get('regularMarketDayHigh', 'N/A'),
            'day_low': info.get('dayLow') or info.get('regularMarketDayLow', 'N/A'),
            'volume': info.get('volume') or info.get('regularMarketVolume', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
        }

    def get_crypto_candlestick(self, data):
        trace1 = go.Candlestick(
            x=data.index,
            open=data.Open,
            high=data.High,
            low=data.Low,
            close=data.Close,
            increasing_line_color='#00c853',
            decreasing_line_color='#ff1744',
        )
        layout = go.Layout(
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(autorange=True, rangeslider=dict(visible=False), gridcolor='#333'),
            yaxis=dict(autorange=True, gridcolor='#333'),
            margin=dict(l=50, r=20, t=20, b=40),
        )
        figure = go.Figure(data=[trace1], layout=layout)
        return plot(figure, output_type='div', include_plotlyjs=False)

    def make_model(self, data):
        train_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        btc_train, btc_test = train_test_split(
            data, train_size=0.8, test_size=0.2, shuffle=False)

        x = btc_train.loc[:, train_cols].values
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x)
        x_test = scaler.transform(data.loc[:, train_cols])

        x_t, y_t = build_timeseries(x_train, 3)
        x_t = trim_data(x_t, BATCH_SIZE)
        y_t = trim_data(y_t, BATCH_SIZE)
        x_temp, y_temp = build_timeseries(x_test, 3)
        x_val, x_test_t = np.split(trim_data(x_temp, BATCH_SIZE), 2)
        y_val, y_test_t = np.split(trim_data(y_temp, BATCH_SIZE), 2)

        model = Sequential([
            Input(shape=(TIME_STEPS, x_t.shape[2])),
            LSTM(100, kernel_initializer='random_uniform'),
            Dropout(0.5),
            Dense(20, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')

        history = model.fit(
            x_t, y_t, epochs=30, verbose=2,
            batch_size=BATCH_SIZE, shuffle=False,
            validation_data=(trim_data(x_val, BATCH_SIZE),
                             trim_data(y_val, BATCH_SIZE)))

        y_pred = model.predict(
            trim_data(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_data(y_test_t, BATCH_SIZE)

        y_pred_org = (y_pred * scaler.data_range_[3] + scaler.data_min_[3])
        y_test_org_t = (y_test_t * scaler.data_range_[3] + scaler.data_min_[3])

        # Build plotly charts instead of static images
        model_loss_chart = self._make_loss_chart(history)
        prediction_chart = self._make_prediction_chart(y_pred_org, y_test_org_t)

        # Tomorrow's prediction
        next_pred = data[-7:]
        x_pred_values = next_pred.loc[:, train_cols].values
        predx = scaler.transform(x_pred_values)
        x_fut, y_fut = build_timeseries(predx, 3)

        future_pred = model.predict(
            trim_data(x_fut, BATCH_SIZE), batch_size=BATCH_SIZE)
        future_pred = future_pred.flatten()

        fut_pred_org = (future_pred * scaler.data_range_[3] + scaler.data_min_[3])
        tom_price = float(f"{fut_pred_org[-1]:.2f}")

        return tom_price, model_loss_chart, prediction_chart

    def _make_loss_chart(self, history):
        traces = [
            go.Scatter(y=history.history['loss'], mode='lines',
                       name='Training Loss', line=dict(color='#00e5ff')),
            go.Scatter(y=history.history['val_loss'], mode='lines',
                       name='Validation Loss', line=dict(color='#ff6e40')),
        ]
        layout = go.Layout(
            title='Model Loss',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Epoch', gridcolor='#333'),
            yaxis=dict(title='Loss', gridcolor='#333'),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return plot(go.Figure(data=traces, layout=layout),
                    output_type='div', include_plotlyjs=False)

    def _make_prediction_chart(self, y_pred_org, y_test_org_t):
        traces = [
            go.Scatter(y=y_pred_org.tolist(), mode='lines',
                       name='Prediction', line=dict(color='#7c4dff')),
            go.Scatter(y=y_test_org_t.tolist(), mode='lines',
                       name='Actual', line=dict(color='#00c853')),
        ]
        layout = go.Layout(
            title='Prediction vs Actual',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Days', gridcolor='#333'),
            yaxis=dict(title='Price ($)', gridcolor='#333'),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return plot(go.Figure(data=traces, layout=layout),
                    output_type='div', include_plotlyjs=False)
