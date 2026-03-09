import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from tqdm.auto import tqdm
import time as _time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from plotly.offline import plot
import plotly.graph_objs as go

TIME_STEPS = 30
BATCH_SIZE = 32  # No longer constrained by stateful LSTM

MARKET_TICKERS = [
    ('SPY',     'S&P 500'),
    ('QQQ',     'Nasdaq 100'),
    ('DIA',     'Dow Jones'),
    ('IWM',     'Russell 2000'),
    ('VTI',     'Total Market'),
    ('BTC-USD', 'Bitcoin'),
]

_market_cache = {'data': None, 'ts': 0.0}
_news_cache = {'data': None, 'ts': 0.0}
_MARKET_CACHE_TTL = 300  # seconds


def build_timeseries(mat, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    for i in tqdm(range(dim_0)):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    return x, y


class Prediction:

    def get_data(self, user_ticker):
        ticker = yf.Ticker(user_ticker)
        today = date.today()
        formed_date = today.strftime("%Y-%m-%d")
        stock = ticker.history(period='max')
        return stock.loc['2015-09-01':formed_date]

    def get_stock_info(self, user_ticker):
        ticker = yf.Ticker(user_ticker)
        info = ticker.info
        return {
            'name': info.get('shortName', user_ticker),
            'symbol': info.get('symbol', user_ticker.upper()),
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice', 'N/A'),
            'previous_close': info.get('previousClose', 'N/A'),
            'open_price': info.get('open') or info.get('regularMarketOpen', 'N/A'),
            'day_high': info.get('dayHigh') or info.get('regularMarketDayHigh', 'N/A'),
            'day_low': info.get('dayLow') or info.get('regularMarketDayLow', 'N/A'),
            'volume': info.get('volume') or info.get('regularMarketVolume', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'description': info.get('longBusinessSummary', ''),
        }

    def get_price_change(self, info):
        try:
            current = float(info['current_price'])
            prev = float(info['previous_close'])
            change = current - prev
            pct = (change / prev) * 100
            return {
                'change': round(change, 2),
                'pct': round(pct, 2),
                'direction': 'up' if change >= 0 else 'down',
            }
        except (ValueError, TypeError, ZeroDivisionError):
            return {'change': 0, 'pct': 0, 'direction': 'neutral'}

    def get_market_overview(self):
        now = _time.time()
        if _market_cache['data'] is not None and (now - _market_cache['ts']) < _MARKET_CACHE_TTL:
            return _market_cache['data']

        results = []
        for symbol, display_name in MARKET_TICKERS:
            try:
                info = yf.Ticker(symbol).info
                current = info.get('currentPrice') or info.get('regularMarketPrice')
                prev    = info.get('previousClose') or info.get('regularMarketPreviousClose')
                if current is None or prev is None:
                    continue
                current, prev = float(current), float(prev)
                change = round(current - prev, 2)
                pct    = round((change / prev) * 100, 2) if prev else 0.0
                results.append({
                    'symbol': symbol, 'name': display_name,
                    'price': round(current, 2), 'change': change,
                    'pct': pct, 'direction': 'up' if change >= 0 else 'down',
                })
            except Exception:
                continue

        _market_cache['data'] = results
        _market_cache['ts'] = _time.time()
        return results

    def get_market_news(self):
        now = _time.time()
        if _news_cache['data'] is not None and (now - _news_cache['ts']) < _MARKET_CACHE_TTL:
            return _news_cache['data']

        articles = []
        seen_titles = set()

        for symbol in ['SPY', 'QQQ']:
            try:
                news = yf.Ticker(symbol).news or []
                for item in news[:8]:
                    content = item.get('content', {})
                    title = content.get('title', '')
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    provider = content.get('provider', {})
                    publisher = provider.get('displayName') if isinstance(provider, dict) else str(provider)

                    click_url = content.get('clickThroughUrl', '')
                    if isinstance(click_url, dict):
                        url = click_url.get('url', '')
                    else:
                        url = str(click_url) if click_url else ''

                    articles.append({
                        'title': title,
                        'publisher': publisher or 'Unknown',
                        'date': (content.get('pubDate') or '')[:10],
                        'url': url,
                    })
            except Exception:
                continue

        articles.sort(key=lambda x: x['date'], reverse=True)
        articles = articles[:8]

        _news_cache['data'] = articles
        _news_cache['ts'] = _time.time()
        return articles

    def get_history_candlestick(self, stocks):
        trace = go.Candlestick(
            x=stocks.index,
            open=stocks.Open, high=stocks.High,
            low=stocks.Low, close=stocks.Close,
            increasing_line_color='#00c853',
            decreasing_line_color='#ff1744',
        )
        layout = go.Layout(
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(autorange=True, rangeslider=dict(visible=False), gridcolor='#333'),
            yaxis=dict(autorange=True, gridcolor='#333'),
            margin=dict(l=50, r=20, t=20, b=40),
        )
        return plot(go.Figure(data=[trace], layout=layout), output_type='div', include_plotlyjs=False)

    def get_volume_chart(self, stocks):
        colors = ['#00c853' if c >= o else '#ff1744'
                  for o, c in zip(stocks.Open, stocks.Close)]
        trace = go.Bar(x=stocks.index, y=stocks.Volume, marker_color=colors)
        layout = go.Layout(
            autosize=True, height=200,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(autorange=True, gridcolor='#333'),
            yaxis=dict(autorange=True, gridcolor='#333'),
            margin=dict(l=50, r=20, t=20, b=40),
        )
        return plot(go.Figure(data=[trace], layout=layout), output_type='div', include_plotlyjs=False)

    def get_moving_averages_chart(self, stocks):
        df = stocks.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        traces = [
            go.Scatter(x=df.index, y=df.Close, mode='lines', name='Close', line=dict(color='#7c4dff', width=1)),
            go.Scatter(x=df.index, y=df.MA20, mode='lines', name='MA20', line=dict(color='#00e5ff', width=1)),
            go.Scatter(x=df.index, y=df.MA50, mode='lines', name='MA50', line=dict(color='#ffd740', width=1)),
            go.Scatter(x=df.index, y=df.MA200, mode='lines', name='MA200', line=dict(color='#ff6e40', width=1)),
        ]
        layout = go.Layout(
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(autorange=True, gridcolor='#333'),
            yaxis=dict(autorange=True, gridcolor='#333'),
            margin=dict(l=50, r=20, t=20, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        return plot(go.Figure(data=traces, layout=layout), output_type='div', include_plotlyjs=False)

    def make_train_test(self, stock):
        train_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        stock_train, stock_test = train_test_split(stock, train_size=0.8, test_size=0.2, shuffle=False)
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(stock_train.loc[:, train_cols].values)
        x_test = scaler.transform(stock_test.loc[:, train_cols])
        return x_train, x_test, scaler

    def make_test_and_val(self, x_train, x_test):
        x_t, y_t = build_timeseries(x_train, 3)
        x_temp, y_temp = build_timeseries(x_test, 3)
        split = len(x_temp) // 2
        x_val, x_test_t = x_temp[:split], x_temp[split:]
        y_val, y_test_t = y_temp[:split], y_temp[split:]
        return x_t, y_t, x_val, y_val, x_test_t, y_test_t

    def train_model(self, _optimizer, x_t, y_t, x_val, y_val, _epochs):
        n_features = x_t.shape[2]
        model = Sequential([
            Input(shape=(TIME_STEPS, n_features)),
            LSTM(100, kernel_initializer='random_uniform'),
            Dropout(0.15),
            Dense(20, activation='relu'),
            Dense(1),
        ])
        model.compile(loss='mean_squared_error', optimizer=_optimizer)
        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        stock_history = model.fit(
            x_t, y_t, epochs=_epochs, verbose=2,
            batch_size=BATCH_SIZE, shuffle=False,
            validation_data=(x_val, y_val),
            callbacks=[early_stop])
        return model, stock_history

    def unscale_data(self, model, x_test_t, y_test_t, scaler):
        y_pred = model.predict(x_test_t).flatten()
        y_pred_org = y_pred * scaler.data_range_[3] + scaler.data_min_[3]
        y_test_org_t = y_test_t * scaler.data_range_[3] + scaler.data_min_[3]
        return y_pred_org, y_test_org_t

    def make_model_loss_chart(self, stock_history, name):
        traces = [
            go.Scatter(y=stock_history.history['val_loss'], mode='lines',
                       name='Validation Loss', line=dict(color='#ff6e40')),
            go.Scatter(y=stock_history.history['loss'], mode='lines',
                       name='Training Loss', line=dict(color='#00e5ff')),
        ]
        layout = go.Layout(
            title=f'Model Loss - {name}',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Epoch', gridcolor='#333'),
            yaxis=dict(title='Loss', gridcolor='#333'),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return plot(go.Figure(data=traces, layout=layout), output_type='div', include_plotlyjs=False)

    def make_prediction_chart(self, y_pred_org, y_test_org_t, name):
        traces = [
            go.Scatter(y=list(y_pred_org), mode='lines', name='Predictions', line=dict(color='#7c4dff')),
            go.Scatter(y=list(y_test_org_t), mode='lines', name='Actual Values', line=dict(color='#00c853')),
        ]
        layout = go.Layout(
            title=f'Prediction vs Actual - {name}',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Days', gridcolor='#333'),
            yaxis=dict(title='Price ($)', gridcolor='#333'),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return plot(go.Figure(data=traces, layout=layout), output_type='div', include_plotlyjs=False)

    def predict_tomorrows_price(self, stock, model, scaler, y_test_t):
        train_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        next_pred = stock[-(TIME_STEPS + 2):]
        predx = scaler.transform(next_pred.loc[:, train_cols].values)
        x_fut, _ = build_timeseries(predx, 3)
        future_pred = model.predict(x_fut).flatten()
        fut_pred_org = future_pred * scaler.data_range_[3] + scaler.data_min_[3]
        return float(f"{fut_pred_org[-1]:.2f}")
