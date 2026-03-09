import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from plotly.offline import plot
import plotly.graph_objs as go


class RFClassifier:

    def get_data(self, user_ticker):
        today = date.today()
        formed_date = today.strftime("%Y-%m-%d")
        ticker = yf.Ticker(user_ticker)
        data = ticker.history(start='2000-01-01', end=formed_date)
        return data

    def make_features(self, data):
        data['Open-Close'] = (data.Open - data.Close) / data.Open
        data['High-Low'] = (data.High - data.Low) / data.Low
        data['percent_change'] = data['Close'].pct_change()
        data['std_5'] = data['percent_change'].rolling(5).std()
        data['ret_5'] = data['percent_change'].rolling(5).mean()
        data.dropna(inplace=True)

        X = data[['Open-Close', 'High-Low', 'std_5', 'ret_5', 'Volume']]
        y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
        return X, y

    def split_data(self, X, y, data):
        dataset_length = data.shape[0]
        split = int(dataset_length * 0.75)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        return X_train, X_test, y_train, y_test

    def make_model(self, X_train, X_test, y_train, y_test):
        clf = RandomForestClassifier(
            random_state=5, n_estimators=500,
            criterion='entropy', max_depth=10)
        model = clf.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test), normalize=True) * 100.0
        report = classification_report(y_test, model.predict(X_test))
        print(f'Accuracy: {accuracy:.1f}%')
        print(report)
        return model

    def make_charts(self, model, data, X, stock_name):
        dataset_length = data.shape[0]
        split = int(dataset_length * 0.75)
        data['strategy_returns'] = data['percent_change'].shift(-1) * model.predict(X)
        predictions = model.predict(X)

        histogram_chart = self._make_histogram(data, split, stock_name)
        returns_chart = self._make_returns_chart(data, split, stock_name)
        return histogram_chart, returns_chart

    def _make_histogram(self, data, split, stock_name):
        trace = go.Histogram(
            x=data.strategy_returns[split:],
            nbinsx=15,
            marker_color='#7c4dff',
        )
        layout = go.Layout(
            title=f'Strategy Returns Distribution - {stock_name}',
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Returns (%)', gridcolor='#333'),
            yaxis=dict(title='Frequency', gridcolor='#333'),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        figure = go.Figure(data=[trace], layout=layout)
        return plot(figure, output_type='div', include_plotlyjs=False)

    def _make_returns_chart(self, data, split, stock_name):
        cumulative = (data.strategy_returns[split:] + 1).cumprod()
        trace = go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00c853'),
        )
        layout = go.Layout(
            title=f'Cumulative Strategy Returns - {stock_name}',
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(title='Date', gridcolor='#333'),
            yaxis=dict(title='Returns', gridcolor='#333'),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        figure = go.Figure(data=[trace], layout=layout)
        return plot(figure, output_type='div', include_plotlyjs=False)

    def predict_tomorrow(self, model, X):
        predictions = model.predict(X)
        return predictions[-1]
