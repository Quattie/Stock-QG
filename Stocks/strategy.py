"""
Multi-horizon XGBoost strategy engine.

Trains separate models for 1d, 5d, and 20d horizons — both classifiers
(direction) and regressors (expected return %). Synthesizes into an
actionable strategy with price targets and conviction levels.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from plotly.offline import plot
import plotly.graph_objs as go

from .XGBModel import XGBStockModel, FEATURE_COLS, compute_rsi, compute_macd, \
    compute_atr, compute_bollinger


HORIZONS = [
    {'name': '1-Day', 'days': 1, 'key': '1d'},
    {'name': '5-Day', 'days': 5, 'key': '5d'},
    {'name': '20-Day', 'days': 20, 'key': '20d'},
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
)


class MultiHorizonStrategy:

    def __init__(self):
        self.xgb_helper = XGBStockModel()

    # ------------------------------------------------------------------
    # Feature engineering (extends base with multi-horizon targets)
    # ------------------------------------------------------------------

    def engineer_features(self, data):
        """Build features + multi-horizon targets."""
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

        # Distance from moving averages
        for period in [5, 10, 20, 50, 200]:
            ma = df['Close'].rolling(period).mean()
            df[f'ma_{period}_dist'] = (df['Close'] - ma) / ma

        # Volume
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_slope'] = df['obv'].pct_change(5)

        # Price patterns
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']

        # Calendar
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # --- Multi-horizon targets ---
        for h in HORIZONS:
            n = h['days']
            key = h['key']
            future_return = df['Close'].shift(-n) / df['Close'] - 1
            df[f'target_dir_{key}'] = (future_return > 0).astype(int)
            df[f'target_ret_{key}'] = future_return

        df.dropna(inplace=True)

        # Drop last N rows where we don't have the furthest target
        max_horizon = max(h['days'] for h in HORIZONS)
        df = df.iloc[:-max_horizon]

        return df

    # ------------------------------------------------------------------
    # Train models for all horizons
    # ------------------------------------------------------------------

    def train_all(self, df):
        """Train classifier + regressor for each horizon. Returns dict of results."""
        X = df[FEATURE_COLS]
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]

        horizon_results = []

        for h in HORIZONS:
            key = h['key']

            # --- Classifier (direction) ---
            y_dir = df[f'target_dir_{key}']
            y_dir_train, y_dir_test = y_dir.iloc[:split], y_dir.iloc[split:]

            clf = xgb.XGBClassifier(**XGB_PARAMS, eval_metric='logloss')
            clf.fit(X_train, y_dir_train,
                    eval_set=[(X_test, y_dir_test)], verbose=False)

            dir_pred = clf.predict(X_test)
            dir_prob = clf.predict_proba(X_test)
            dir_accuracy = float((dir_pred == y_dir_test.values).mean() * 100)

            # --- Regressor (expected return) ---
            y_ret = df[f'target_ret_{key}']
            y_ret_train, y_ret_test = y_ret.iloc[:split], y_ret.iloc[split:]

            reg = xgb.XGBRegressor(**XGB_PARAMS, eval_metric='rmse')
            reg.fit(X_train, y_ret_train,
                    eval_set=[(X_test, y_ret_test)], verbose=False)

            ret_pred = reg.predict(X_test)

            # --- Forward prediction (latest row) ---
            latest = X.iloc[[-1]]
            fwd_dir = clf.predict(latest)[0]
            fwd_prob = clf.predict_proba(latest)[0]
            fwd_confidence = float(max(fwd_prob)) * 100
            fwd_ret = float(reg.predict(latest)[0])

            # --- Backtest: strategy returns on test set ---
            fwd_returns = df[f'target_ret_{key}'].iloc[split:split + len(dir_pred)].values
            signal = np.where(dir_pred == 1, 1.0, -1.0)
            strat_returns = signal * fwd_returns
            cumulative = pd.Series(
                (1 + strat_returns).cumprod(), index=X_test.index)
            buy_hold = pd.Series(
                (1 + fwd_returns).cumprod(), index=X_test.index)

            horizon_results.append({
                'horizon': h,
                'accuracy': round(dir_accuracy, 1),
                'direction': 'up' if fwd_dir == 1 else 'down',
                'confidence': round(fwd_confidence, 1),
                'expected_return': round(fwd_ret * 100, 2),
                'cumulative': cumulative,
                'buy_hold': buy_hold,
                'feature_importance': dict(zip(FEATURE_COLS, clf.feature_importances_)),
                # For regression quality
                'reg_r2': round(float(1 - np.sum((y_ret_test.values - ret_pred)**2) /
                                       np.sum((y_ret_test.values - y_ret_test.mean())**2)) * 100, 1),
            })

        return horizon_results

    # ------------------------------------------------------------------
    # Strategy synthesis
    # ------------------------------------------------------------------

    def synthesize_strategy(self, horizon_results, current_price):
        """Combine multi-horizon signals into a strategy recommendation."""
        signals = {}
        price_targets = {}

        for hr in horizon_results:
            key = hr['horizon']['key']
            name = hr['horizon']['name']
            direction = hr['direction']
            confidence = hr['confidence']
            expected_return = hr['expected_return']

            # Signed signal: positive = bullish, negative = bearish
            signed = (confidence / 100) * (1 if direction == 'up' else -1)
            signals[key] = {
                'direction': direction,
                'confidence': confidence,
                'signed': signed,
                'expected_return': expected_return,
            }
            price_targets[key] = round(current_price * (1 + expected_return / 100), 2)

        # Determine strategy
        dirs = [s['direction'] for s in signals.values()]
        confs = [s['confidence'] for s in signals.values()]
        avg_confidence = sum(confs) / len(confs)

        short_term = signals['1d']
        medium_term = signals['5d']
        long_term = signals['20d']

        all_bullish = all(d == 'up' for d in dirs)
        all_bearish = all(d == 'down' for d in dirs)
        short_bearish_long_bullish = (
            short_term['direction'] == 'down' and long_term['direction'] == 'up')
        short_bullish_long_bearish = (
            short_term['direction'] == 'up' and long_term['direction'] == 'down')

        if all_bullish and avg_confidence >= 58:
            action = 'Strong Buy'
            rationale = 'All horizons bullish with solid conviction across timeframes.'
            color = 'success'
        elif all_bullish:
            action = 'Buy'
            rationale = 'All horizons agree on upside, but conviction is moderate.'
            color = 'success'
        elif all_bearish and avg_confidence >= 58:
            action = 'Strong Sell'
            rationale = 'All horizons bearish with solid conviction. Risk is elevated.'
            color = 'danger'
        elif all_bearish:
            action = 'Sell / Avoid'
            rationale = 'All horizons point down, though conviction is moderate.'
            color = 'danger'
        elif short_bearish_long_bullish:
            action = 'Accumulate on Dips'
            rationale = (
                f'Short-term weakness expected ({short_term["expected_return"]:+.1f}% 1d), '
                f'but {long_term["expected_return"]:+.1f}% upside over 20d. '
                'Consider scaling in on pullbacks.'
            )
            color = 'info'
        elif short_bullish_long_bearish:
            action = 'Short-term Trade Only'
            rationale = (
                f'Near-term upside ({short_term["expected_return"]:+.1f}% 1d), '
                f'but longer-term outlook is negative ({long_term["expected_return"]:+.1f}% 20d). '
                'Take profits quickly.'
            )
            color = 'warning'
        else:
            # Mixed signals
            weighted = 0.2 * short_term['signed'] + 0.3 * medium_term['signed'] + \
                       0.5 * long_term['signed']
            if weighted > 0.05:
                action = 'Lean Bullish'
                rationale = 'Mixed signals but weight of evidence tilts positive.'
                color = 'success'
            elif weighted < -0.05:
                action = 'Lean Bearish'
                rationale = 'Mixed signals but weight of evidence tilts negative.'
                color = 'danger'
            else:
                action = 'Neutral / No Edge'
                rationale = 'Conflicting signals across timeframes. No clear edge — sit tight.'
                color = 'secondary'

        return {
            'action': action,
            'rationale': rationale,
            'color': color,
            'signals': signals,
            'price_targets': price_targets,
            'avg_confidence': round(avg_confidence, 1),
        }

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def make_multi_horizon_chart(self, horizon_results):
        """Backtest performance chart for all horizons overlaid."""
        colors = ['#7c4dff', '#00e5ff', '#ff9100']
        traces = []

        for i, hr in enumerate(horizon_results):
            key = hr['horizon']['key']
            name = hr['horizon']['name']
            traces.append(go.Scatter(
                x=hr['cumulative'].index, y=hr['cumulative'].values,
                mode='lines', name=f'{name} Strategy',
                line=dict(color=colors[i], width=2),
            ))

        # Add buy-and-hold from longest horizon
        bh = horizon_results[-1]['buy_hold']
        traces.append(go.Scatter(
            x=bh.index, y=bh.values,
            mode='lines', name='Buy & Hold',
            line=dict(color='#484f58', width=1, dash='dash'),
        ))

        layout = go.Layout(
            title='Multi-Horizon Strategy Backtest',
            autosize=True, height=450,
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

    def make_price_target_chart(self, current_price, price_targets, strategy):
        """Visual price target ladder."""
        labels = ['Now'] + [h['name'] for h in HORIZONS]
        prices = [current_price] + [price_targets[h['key']] for h in HORIZONS]
        colors_map = {
            'up': '#00c853',
            'down': '#ff1744',
        }

        bar_colors = ['#7c4dff']  # current = purple
        for h in HORIZONS:
            sig = strategy['signals'][h['key']]
            bar_colors.append(colors_map.get(sig['direction'], '#888'))

        trace = go.Bar(
            x=labels, y=prices,
            marker_color=bar_colors,
            text=[f'${p:.2f}' for p in prices],
            textposition='outside',
            textfont=dict(color='#e0e0e0'),
        )
        layout = go.Layout(
            title='Price Targets by Horizon',
            autosize=True, height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(title='Price ($)', gridcolor='#333',
                       range=[min(prices) * 0.97, max(prices) * 1.03]),
            margin=dict(l=60, r=20, t=50, b=40),
        )
        return plot(go.Figure(data=[trace], layout=layout),
                    output_type='div', include_plotlyjs=False)

    def make_confidence_radar(self, horizon_results):
        """Radar chart of accuracy and confidence per horizon."""
        labels = [hr['horizon']['name'] for hr in horizon_results]
        accuracy = [hr['accuracy'] for hr in horizon_results]
        confidence = [hr['confidence'] for hr in horizon_results]

        traces = [
            go.Scatterpolar(
                r=accuracy + [accuracy[0]],
                theta=labels + [labels[0]],
                fill='toself', name='Backtest Accuracy',
                line=dict(color='#7c4dff'),
                fillcolor='rgba(124,77,255,0.2)',
            ),
            go.Scatterpolar(
                r=confidence + [confidence[0]],
                theta=labels + [labels[0]],
                fill='toself', name='Current Confidence',
                line=dict(color='#00e5ff'),
                fillcolor='rgba(0,229,255,0.2)',
            ),
        ]
        layout = go.Layout(
            autosize=True, height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[40, 100],
                                gridcolor='#333', linecolor='#333'),
                angularaxis=dict(gridcolor='#333', linecolor='#333'),
            ),
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
        )
        return plot(go.Figure(data=traces, layout=layout),
                    output_type='div', include_plotlyjs=False)
