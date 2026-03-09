import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score

from plotly.offline import plot
import plotly.graph_objs as go


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bollinger(series, period=20, std_dev=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    width = (upper - lower) / ma
    pct_b = (series - lower) / (upper - lower)
    return width, pct_b


# ---------------------------------------------------------------------------
# Feature columns used by the model
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'rsi_14', 'rsi_7',
    'macd', 'macd_signal', 'macd_hist',
    'atr_14', 'volatility_20',
    'bb_width', 'bb_pct',
    'ma_5_dist', 'ma_10_dist', 'ma_20_dist', 'ma_50_dist', 'ma_200_dist',
    'volume_ratio', 'obv_slope',
    'high_low_ratio', 'open_close_ratio',
    'day_of_week', 'month',
]


# ---------------------------------------------------------------------------
# XGBoost stock model
# ---------------------------------------------------------------------------

class XGBStockModel:

    def get_data(self, ticker_str):
        ticker = yf.Ticker(ticker_str)
        today = date.today().strftime("%Y-%m-%d")
        return ticker.history(start='2000-01-01', end=today)

    def engineer_features(self, data):
        df = data.copy()

        # Returns
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)

        # Momentum
        df['rsi_14'] = compute_rsi(df['Close'], 14)
        df['rsi_7'] = compute_rsi(df['Close'], 7)
        macd, macd_signal, macd_hist = compute_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        # Volatility
        df['atr_14'] = compute_atr(df['High'], df['Low'], df['Close'], 14)
        df['volatility_20'] = df['return_1d'].rolling(20).std()
        bb_width, bb_pct = compute_bollinger(df['Close'])
        df['bb_width'] = bb_width
        df['bb_pct'] = bb_pct

        # Distance from moving averages (relative)
        for period in [5, 10, 20, 50, 200]:
            ma = df['Close'].rolling(period).mean()
            df[f'ma_{period}_dist'] = (df['Close'] - ma) / ma

        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_slope'] = df['obv'].pct_change(5)

        # Price patterns
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']

        # Calendar
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # Target: next-day direction (1 = up, 0 = down)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Drop rows with NaN features and the last row (no target)
        df.dropna(inplace=True)
        df = df.iloc[:-1]

        return df

    def train(self, df):
        X = df[FEATURE_COLS]
        y = df['target']

        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        # Backtest: strategy returns on test set
        fwd = df['Close'].pct_change().shift(-1)
        test_fwd = fwd.iloc[split:split + len(y_pred)].fillna(0).values
        signal = np.where(y_pred == 1, 1.0, -1.0)
        strat = signal * test_fwd

        cumulative = pd.Series((1 + strat).cumprod(), index=X_test.index)
        buy_hold = pd.Series((1 + test_fwd).cumprod(), index=X_test.index)

        # Tomorrow's prediction
        latest = X.iloc[[-1]]
        tomorrow_pred = model.predict(latest)[0]
        tomorrow_prob = model.predict_proba(latest)[0]
        confidence = float(max(tomorrow_prob)) * 100

        # Feature importance
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))

        return {
            'accuracy': round(accuracy * 100, 1),
            'precision': round(precision * 100, 1),
            'direction': 'up' if tomorrow_pred == 1 else 'down',
            'confidence': round(confidence, 1),
            'feature_importance': importance,
            'cumulative': cumulative,
            'buy_hold': buy_hold,
        }

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def make_feature_importance_chart(self, feature_importance):
        sorted_feats = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        names = [f[0] for f in reversed(sorted_feats)]
        values = [f[1] for f in reversed(sorted_feats)]

        trace = go.Bar(x=values, y=names, orientation='h', marker_color='#7c4dff')
        layout = go.Layout(
            title='Top 15 Feature Importance',
            autosize=True, height=450,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333'),
            margin=dict(l=140, r=20, t=50, b=40),
        )
        return plot(go.Figure(data=[trace], layout=layout),
                    output_type='div', include_plotlyjs=False)

    def make_backtest_chart(self, cumulative, buy_hold):
        traces = [
            go.Scatter(
                x=cumulative.index, y=cumulative.values,
                mode='lines', name='XGBoost Strategy',
                line=dict(color='#7c4dff', width=2),
            ),
            go.Scatter(
                x=buy_hold.index, y=buy_hold.values,
                mode='lines', name='Buy & Hold',
                line=dict(color='#484f58', width=1, dash='dash'),
            ),
        ]
        layout = go.Layout(
            title='Strategy vs Buy & Hold',
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Date', gridcolor='#333'),
            yaxis=dict(title='Cumulative Return', gridcolor='#333'),
            margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
        )
        return plot(go.Figure(data=traces, layout=layout),
                    output_type='div', include_plotlyjs=False)
