import numpy as np
import quantrautil as q
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')


class RFClassifier():

    def get_data(self, user_ticker):

        today = date.today()
        formed_date = today.strftime("%Y-%m-%d")
        data = q.get_data(user_ticker, '2000-1-1', formed_date)
        print(data.tail())
        return data

    def make_features(self, data):

        # Features construction
        data['Open-Close'] = (data.Open - data.Close)/data.Open
        data['High-Low'] = (data.High - data.Low)/data.Low
        data['percent_change'] = data['Adj Close'].pct_change()
        data['std_5'] = data['percent_change'].rolling(5).std()
        data['ret_5'] = data['percent_change'].rolling(5).mean()
        data.dropna(inplace=True)

        # X is the input variable
        X = data[['Open-Close', 'High-Low', 'std_5', 'ret_5',
                  'Volume']]

        # Y is the target or output variable
        y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)
        print(X.shape)
        print(y.shape)
        return X, y

    def split_data(self, X, y, data):
        # Total dataset length
        dataset_length = data.shape[0]

        # Training dataset length
        split = int(dataset_length * 0.75)

        # Splitiing the X and y into train and test datasets
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Print the size of the train and test dataset
        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)

        return X_train, X_test, y_train, y_test

    def make_model(self, X_train, X_test, y_train, y_test):
        clf = RandomForestClassifier(
            random_state=5, n_estimators=500, criterion='entropy', max_depth=10)

        # Create the model on train dataset
        model = clf.fit(X_train, y_train)
        print('Correct Prediction (%): ', accuracy_score(
            y_test, model.predict(X_test), normalize=True)*100.0)
        # Run the code to view the classification report metrics
        from sklearn.metrics import classification_report
        report = classification_report(y_test, model.predict(X_test))
        print(report)
        return model

    def make_charts(self, model, data, X, stock_name):

        plt.style.use('dark_background')
        dataset_length = data.shape[0]
        split = int(dataset_length * 0.75)
        data['strategy_returns'] = data['percent_change'].shift(
            -1) * model.predict(X)
        predictions = model.predict(X)

        fig3 = plt.figure()
        fig3.autofmt_xdate()
        plt.hist(data.strategy_returns[split:], density=1, bins=15)
        plt.xlabel('Strategy Returns (%)')
        plt.grid()

        fig2 = plt.figure()
        fig2.autofmt_xdate()
        (data.strategy_returns[split:]+1).cumprod().plot()
        plt.ylabel('Strategy returns (%)')
        plt.title('Returns of ' + stock_name + ' Stocks')

        plt.savefig("Stocks/static/Stocks/charts/" + "ReturnsOf" +
                    stock_name + ".png")

    def predict_tomorrow(self, model, X):
        predictions = model.predict(X)
        print(predictions[-1])
        return predictions[-1]
