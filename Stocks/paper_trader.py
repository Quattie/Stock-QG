import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump as joblib_dump, load as joblib_load

from plotly.offline import plot
import plotly.graph_objs as go

from .XGBModel import XGBStockModel, FEATURE_COLS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


class PaperTrader:
    """File-backed paper trading simulator with XGBoost integration."""

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.portfolio = self._load('portfolio.json', {
            'cash': 10000.0,
            'initial_cash': 10000.0,
            'positions': {},
        })
        self.trades = self._load('trades.json', [])
        self.snapshots = self._load('snapshots.json', [])
        self.last_backtest = self._load('last_backtest.json', None)

    def _path(self, filename):
        return os.path.join(DATA_DIR, filename)

    def _load(self, filename, default):
        path = self._path(filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    def _save(self):
        for name, data in [
            ('portfolio.json', self.portfolio),
            ('trades.json', self.trades),
            ('snapshots.json', self.snapshots),
            ('last_backtest.json', self.last_backtest),
        ]:
            with open(self._path(name), 'w') as f:
                json.dump(data, f, indent=2, default=str)

    def reset(self):
        self.portfolio = {'cash': 10000.0, 'initial_cash': 10000.0, 'positions': {}}
        self.trades = []
        self.snapshots = []
        self.last_backtest = None
        self._save()

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run_backtest(self, ticker, initial_cash=10000.0, confidence_threshold=55.0):
        """Run full historical backtest using XGBoost predictions."""
        xgb_model = XGBStockModel()
        data = xgb_model.get_data(ticker)
        df = xgb_model.engineer_features(data)

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
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # --- simulate day-by-day trading ---
        cash = initial_cash
        shares = 0
        avg_cost = 0.0
        trades = []
        snapshots = []
        peak_value = initial_cash

        for i in range(len(X_test)):
            date_str = X_test.index[i].strftime('%Y-%m-%d')
            price = float(df['Close'].iloc[split + i])
            direction = 'up' if y_pred[i] == 1 else 'down'
            confidence = float(max(y_prob[i])) * 100
            actual = 'up' if y_test.iloc[i] == 1 else 'down'

            # Buy signal
            if direction == 'up' and shares == 0 and confidence >= confidence_threshold:
                max_shares = int((cash * 0.95) / price)
                if max_shares > 0:
                    cost = max_shares * price
                    cash -= cost
                    shares = max_shares
                    avg_cost = price
                    trades.append({
                        'date': date_str, 'action': 'BUY', 'ticker': ticker,
                        'shares': shares, 'price': round(price, 2),
                        'total': round(cost, 2),
                        'confidence': round(confidence, 1),
                        'signal': direction,
                    })

            # Sell signal
            elif direction == 'down' and shares > 0:
                revenue = shares * price
                pnl = revenue - (shares * avg_cost)
                pnl_pct = (pnl / (shares * avg_cost)) * 100
                cash += revenue
                trades.append({
                    'date': date_str, 'action': 'SELL', 'ticker': ticker,
                    'shares': shares, 'price': round(price, 2),
                    'total': round(revenue, 2),
                    'pnl': round(pnl, 2),
                    'return_pct': round(pnl_pct, 2),
                    'confidence': round(confidence, 1),
                    'signal': direction,
                })
                shares = 0
                avg_cost = 0.0

            # Daily snapshot
            portfolio_value = cash + shares * price
            peak_value = max(peak_value, portfolio_value)
            drawdown = ((peak_value - portfolio_value) / peak_value) * 100

            snapshots.append({
                'date': date_str,
                'value': round(portfolio_value, 2),
                'cash': round(cash, 2),
                'shares': shares,
                'price': round(price, 2),
                'drawdown': round(drawdown, 2),
                'signal': direction,
                'correct': direction == actual,
            })

        # Close remaining position at final price
        if shares > 0:
            final_price = snapshots[-1]['price']
            revenue = shares * final_price
            pnl = revenue - (shares * avg_cost)
            cash += revenue
            trades.append({
                'date': snapshots[-1]['date'], 'action': 'SELL (CLOSE)',
                'ticker': ticker, 'shares': shares,
                'price': round(final_price, 2),
                'total': round(revenue, 2),
                'pnl': round(pnl, 2),
                'return_pct': round((pnl / (shares * avg_cost)) * 100, 2),
                'confidence': 0, 'signal': 'close',
            })
            shares = 0

        final_value = cash

        # Buy-and-hold comparison
        bh_start_price = float(df['Close'].iloc[split])
        bh_end_price = snapshots[-1]['price']
        bh_return = ((bh_end_price - bh_start_price) / bh_start_price) * 100

        # Win rate
        sell_trades = [t for t in trades if 'SELL' in t['action']]
        winning = [t for t in sell_trades if t.get('pnl', 0) > 0]
        win_rate = (len(winning) / len(sell_trades) * 100) if sell_trades else 0

        # Signal accuracy
        correct_signals = sum(1 for s in snapshots if s['correct'])
        signal_accuracy = (correct_signals / len(snapshots) * 100) if snapshots else 0

        # Max drawdown
        max_drawdown = max((s['drawdown'] for s in snapshots), default=0)

        # Avg win / avg loss
        wins = [t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in sell_trades if t.get('pnl', 0) <= 0]
        avg_win = round(sum(wins) / len(wins), 2) if wins else 0
        avg_loss = round(sum(losses) / len(losses), 2) if losses else 0

        summary = {
            'ticker': ticker,
            'initial_cash': initial_cash,
            'final_value': round(final_value, 2),
            'total_return': round(((final_value - initial_cash) / initial_cash) * 100, 2),
            'buy_hold_return': round(bh_return, 2),
            'total_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'signal_accuracy': round(signal_accuracy, 1),
            'max_drawdown': round(max_drawdown, 2),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'confidence_threshold': confidence_threshold,
            'test_start': snapshots[0]['date'] if snapshots else '',
            'test_end': snapshots[-1]['date'] if snapshots else '',
        }

        # Persist
        self.portfolio = {
            'cash': round(cash, 2),
            'initial_cash': initial_cash,
            'positions': {},
        }
        self.trades = trades
        self.snapshots = snapshots
        self.last_backtest = summary
        self._save()

        return summary

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def make_performance_chart(self, snapshots, initial_cash, trades=None):
        if not snapshots:
            return ''

        dates = [s['date'] for s in snapshots]
        values = [s['value'] for s in snapshots]

        # Buy-and-hold line
        start_price = snapshots[0]['price']
        bh_values = [initial_cash * (s['price'] / start_price) for s in snapshots]

        traces = [
            go.Scatter(
                x=dates, y=values, mode='lines', name='XGBoost Strategy',
                line=dict(color='#7c4dff', width=2)),
            go.Scatter(
                x=dates, y=bh_values, mode='lines', name='Buy & Hold',
                line=dict(color='#484f58', width=1, dash='dash')),
        ]

        # Buy / sell markers
        if trades:
            date_to_value = {s['date']: s['value'] for s in snapshots}
            buys = [t for t in trades if t['action'] == 'BUY']
            sells = [t for t in trades if 'SELL' in t['action']]

            if buys:
                traces.append(go.Scatter(
                    x=[t['date'] for t in buys],
                    y=[date_to_value.get(t['date'], 0) for t in buys],
                    mode='markers', name='Buy',
                    marker=dict(color='#00c853', symbol='triangle-up', size=12),
                ))
            if sells:
                traces.append(go.Scatter(
                    x=[t['date'] for t in sells],
                    y=[date_to_value.get(t['date'], 0) for t in sells],
                    mode='markers', name='Sell',
                    marker=dict(color='#ff1744', symbol='triangle-down', size=12),
                ))

        layout = go.Layout(
            autosize=True, height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Date', gridcolor='#333'),
            yaxis=dict(title='Portfolio Value ($)', gridcolor='#333'),
            margin=dict(l=60, r=20, t=20, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
        )
        return plot(go.Figure(data=traces, layout=layout),
                    output_type='div', include_plotlyjs=False)

    def make_drawdown_chart(self, snapshots):
        if not snapshots:
            return ''

        dates = [s['date'] for s in snapshots]
        drawdowns = [-s['drawdown'] for s in snapshots]

        trace = go.Scatter(
            x=dates, y=drawdowns, fill='tozeroy', mode='lines',
            name='Drawdown',
            line=dict(color='#ff1744', width=1),
            fillcolor='rgba(255, 23, 68, 0.2)',
        )
        layout = go.Layout(
            autosize=True, height=200,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(title='Drawdown (%)', gridcolor='#333'),
            margin=dict(l=60, r=20, t=10, b=40),
        )
        return plot(go.Figure(data=[trace], layout=layout),
                    output_type='div', include_plotlyjs=False)


# ======================================================================
# Live Tracker — forward-looking paper trading with real market data
# ======================================================================

class LiveTrader:
    """Tracks XGBoost predictions day-by-day against real market data.

    On each page load, checks for new trading days since last update,
    evaluates prior predictions, executes trades, and makes new predictions.
    All state persists to data/ as JSON. Model persists via joblib.
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.config = self._load('live_config.json', None)
        self.portfolio = self._load('live_portfolio.json', {
            'cash': 0, 'shares': 0, 'avg_cost': 0,
        })
        self.predictions = self._load('live_predictions.json', [])
        self.snapshots = self._load('live_snapshots.json', [])
        self.trades = self._load('live_trades.json', [])

    def _path(self, filename):
        return os.path.join(DATA_DIR, filename)

    def _load(self, filename, default):
        path = self._path(filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    def _save(self):
        for name, data in [
            ('live_config.json', self.config),
            ('live_portfolio.json', self.portfolio),
            ('live_predictions.json', self.predictions),
            ('live_snapshots.json', self.snapshots),
            ('live_trades.json', self.trades),
        ]:
            with open(self._path(name), 'w') as f:
                json.dump(data, f, indent=2, default=str)

    @property
    def is_active(self):
        return self.config is not None and self.config.get('active', False)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self, ticker, initial_cash, confidence_threshold=55.0):
        """Train XGBoost on all data, make first prediction, initialize portfolio."""
        xgb_model = XGBStockModel()
        data = xgb_model.get_data(ticker)
        df = xgb_model.engineer_features(data)

        X = df[FEATURE_COLS]
        y = df['target']

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='logloss',
        )
        model.fit(X, y, verbose=False)

        # Persist model
        joblib_dump(model, self._path('live_model.joblib'))

        current_price = round(float(data['Close'].iloc[-1]), 2)
        last_date = df.index[-1].strftime('%Y-%m-%d')

        # First prediction
        pred = model.predict(X.iloc[[-1]])[0]
        prob = model.predict_proba(X.iloc[[-1]])[0]

        self.config = {
            'ticker': ticker,
            'initial_cash': initial_cash,
            'confidence_threshold': confidence_threshold,
            'start_date': last_date,
            'active': True,
        }
        self.portfolio = {'cash': initial_cash, 'shares': 0, 'avg_cost': 0.0}
        self.predictions = [{
            'date': last_date,
            'direction': 'up' if pred == 1 else 'down',
            'confidence': round(float(max(prob)) * 100, 1),
            'price_at_prediction': current_price,
        }]
        self.snapshots = [{
            'date': last_date,
            'value': initial_cash,
            'cash': initial_cash,
            'shares': 0,
            'price': current_price,
        }]
        self.trades = []
        self._save()

    # ------------------------------------------------------------------
    # Update — called every page load
    # ------------------------------------------------------------------

    def update(self):
        """Process any new trading days since last update."""
        if not self.is_active:
            return

        ticker = self.config['ticker']
        threshold = self.config.get('confidence_threshold', 55.0)

        xgb_model = XGBStockModel()
        data = xgb_model.get_data(ticker)
        df = xgb_model.engineer_features(data)

        # Determine new trading days
        last_date = pd.Timestamp(self.snapshots[-1]['date'])
        new_dates = df.index[df.index > last_date]

        if len(new_dates) == 0:
            return

        # Load model
        model_path = self._path('live_model.joblib')
        if not os.path.exists(model_path):
            return
        model = joblib_load(model_path)

        for dt in new_dates:
            date_str = dt.strftime('%Y-%m-%d')
            price = round(float(df.loc[dt, 'Close']), 2)

            # --- evaluate yesterday's prediction ---
            if self.predictions:
                prev = self.predictions[-1]
                if 'actual_direction' not in prev:
                    actual = 'up' if price > prev['price_at_prediction'] else 'down'
                    prev['actual_direction'] = actual
                    prev['actual_price'] = price
                    prev['correct'] = prev['direction'] == actual

                    # --- execute trade based on that prediction ---
                    direction = prev['direction']
                    confidence = prev['confidence']

                    if direction == 'up' and self.portfolio['shares'] == 0 \
                            and confidence >= threshold:
                        max_shares = int((self.portfolio['cash'] * 0.95) / price)
                        if max_shares > 0:
                            cost = max_shares * price
                            self.portfolio['cash'] -= cost
                            self.portfolio['shares'] = max_shares
                            self.portfolio['avg_cost'] = price
                            self.trades.append({
                                'date': date_str, 'action': 'BUY',
                                'ticker': ticker, 'shares': max_shares,
                                'price': price, 'total': round(cost, 2),
                                'confidence': confidence, 'signal': direction,
                            })

                    elif direction == 'down' and self.portfolio['shares'] > 0:
                        shares = self.portfolio['shares']
                        revenue = shares * price
                        pnl = revenue - (shares * self.portfolio['avg_cost'])
                        pnl_pct = (pnl / (shares * self.portfolio['avg_cost'])) * 100
                        self.portfolio['cash'] += revenue
                        self.portfolio['shares'] = 0
                        self.portfolio['avg_cost'] = 0.0
                        self.trades.append({
                            'date': date_str, 'action': 'SELL',
                            'ticker': ticker, 'shares': shares,
                            'price': price, 'total': round(revenue, 2),
                            'pnl': round(pnl, 2),
                            'return_pct': round(pnl_pct, 2),
                            'confidence': confidence, 'signal': direction,
                        })

            # --- snapshot ---
            portfolio_value = self.portfolio['cash'] + self.portfolio['shares'] * price
            self.snapshots.append({
                'date': date_str,
                'value': round(portfolio_value, 2),
                'cash': round(self.portfolio['cash'], 2),
                'shares': self.portfolio['shares'],
                'price': price,
            })

            # --- new prediction for tomorrow ---
            features = df.loc[[dt], FEATURE_COLS]
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            self.predictions.append({
                'date': date_str,
                'direction': 'up' if pred == 1 else 'down',
                'confidence': round(float(max(prob)) * 100, 1),
                'price_at_prediction': price,
            })

        self._save()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def retrain(self):
        """Retrain the model on latest data without resetting portfolio."""
        if not self.is_active:
            return
        ticker = self.config['ticker']
        xgb_model = XGBStockModel()
        data = xgb_model.get_data(ticker)
        df = xgb_model.engineer_features(data)
        X = df[FEATURE_COLS]
        y = df['target']
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='logloss',
        )
        model.fit(X, y, verbose=False)
        joblib_dump(model, self._path('live_model.joblib'))

    def reset(self):
        """Wipe all live tracking data."""
        self.config = None
        self.portfolio = {'cash': 0, 'shares': 0, 'avg_cost': 0}
        self.predictions = []
        self.snapshots = []
        self.trades = []
        for f in ('live_model.joblib',):
            path = self._path(f)
            if os.path.exists(path):
                os.remove(path)
        self._save()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self):
        """Build status dict for template rendering."""
        if not self.config:
            return None

        initial = self.config['initial_cash']
        current_value = self.snapshots[-1]['value'] if self.snapshots else initial
        total_return = ((current_value - initial) / initial) * 100

        resolved = [p for p in self.predictions if 'correct' in p]
        correct = sum(1 for p in resolved if p['correct'])
        accuracy = (correct / len(resolved) * 100) if resolved else 0

        # Current (unresolved) prediction
        last_pred = self.predictions[-1] if self.predictions else None
        current_prediction = None
        if last_pred and 'actual_direction' not in last_pred:
            current_prediction = {
                'direction': last_pred['direction'],
                'confidence': last_pred['confidence'],
            }

        return {
            'ticker': self.config['ticker'],
            'initial_cash': initial,
            'current_value': round(current_value, 2),
            'total_return': round(total_return, 2),
            'total_trades': len(self.trades),
            'accuracy': round(accuracy, 1),
            'days_tracked': len(self.snapshots),
            'start_date': self.config['start_date'],
            'active': self.config.get('active', False),
            'confidence_threshold': self.config.get('confidence_threshold', 55),
            'current_prediction': current_prediction,
            'shares': self.portfolio['shares'],
            'cash': round(self.portfolio['cash'], 2),
        }

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def make_performance_chart(self):
        if not self.snapshots:
            return ''

        initial = self.config['initial_cash']
        dates = [s['date'] for s in self.snapshots]
        values = [s['value'] for s in self.snapshots]
        start_price = self.snapshots[0]['price']
        bh_values = [initial * (s['price'] / start_price) for s in self.snapshots]

        traces = [
            go.Scatter(x=dates, y=values, mode='lines', name='Live Strategy',
                       line=dict(color='#00e5ff', width=2)),
            go.Scatter(x=dates, y=bh_values, mode='lines', name='Buy & Hold',
                       line=dict(color='#484f58', width=1, dash='dash')),
        ]

        # Buy / sell markers
        date_to_value = {s['date']: s['value'] for s in self.snapshots}
        buys = [t for t in self.trades if t['action'] == 'BUY']
        sells = [t for t in self.trades if 'SELL' in t['action']]
        if buys:
            traces.append(go.Scatter(
                x=[t['date'] for t in buys],
                y=[date_to_value.get(t['date'], 0) for t in buys],
                mode='markers', name='Buy',
                marker=dict(color='#00c853', symbol='triangle-up', size=12)))
        if sells:
            traces.append(go.Scatter(
                x=[t['date'] for t in sells],
                y=[date_to_value.get(t['date'], 0) for t in sells],
                mode='markers', name='Sell',
                marker=dict(color='#ff1744', symbol='triangle-down', size=12)))

        layout = go.Layout(
            autosize=True, height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Date', gridcolor='#333'),
            yaxis=dict(title='Portfolio Value ($)', gridcolor='#333'),
            margin=dict(l=60, r=20, t=20, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
        )
        return plot(go.Figure(data=traces, layout=layout),
                    output_type='div', include_plotlyjs=False)
