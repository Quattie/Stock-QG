from django.shortcuts import render
from .forms import TickerForm, EpochForm, CryptoForm
from .predictions import Prediction
from .RFClassifier import RFClassifier
from .XGBModel import XGBStockModel
from .paper_trader import PaperTrader, LiveTrader
from .Crypto import Crypto
from .sentiment import analyze_sentiment


def home(request):
    ticker = TickerForm()
    history_chart = ''
    volume_chart = ''
    ma_chart = ''
    stock_info = None
    price_change = None
    sentiment = None

    pred = Prediction()
    market_overview = pred.get_market_overview()
    market_news = pred.get_market_news()

    if request.method == "POST":
        ticker = TickerForm(request.POST)
        if ticker.is_valid():
            ticker_str = ticker.cleaned_data['ticker'].upper()

            try:
                stock_info = pred.get_stock_info(ticker_str)
                price_change = pred.get_price_change(stock_info)
                stock_data = pred.get_data(ticker_str)
                history_chart = pred.get_history_candlestick(stock_data)
                volume_chart = pred.get_volume_chart(stock_data)
                ma_chart = pred.get_moving_averages_chart(stock_data)
                sentiment = analyze_sentiment(ticker_str, stock_info.get('name', ''))
            except Exception as e:
                stock_info = {'error': str(e)}
                sentiment = None

            request.session['ticker_str'] = ticker_str
            ticker = TickerForm()

    return render(request, 'Stocks/home.html', context={
        'form': ticker,
        'market_overview': market_overview,
        'market_news': market_news,
        'history_chart': history_chart,
        'volume_chart': volume_chart,
        'ma_chart': ma_chart,
        'stock_info': stock_info,
        'price_change': price_change,
        'sentiment': sentiment,
    })


def about(request):
    return render(request, 'Stocks/about.html', {'title': 'About'})


def recurrent(request):
    ticker_str = request.session.get('ticker_str')
    if not ticker_str:
        return render(request, 'Stocks/home.html', {
            'form': TickerForm(),
            'error': 'Please enter a stock ticker first.',
        })

    epochs = 100  # EarlyStopping will halt training when val_loss plateaus
    pred = Prediction()
    stock_info = pred.get_stock_info(ticker_str)
    stock_data = pred.get_data(ticker_str)
    x_train, x_test, scaler = pred.make_train_test(stock_data)
    x_t, y_t, x_val, y_val, x_test_t, y_test_t = pred.make_test_and_val(x_train, x_test)

    model, stock_history = pred.train_model('adam', x_t, y_t, x_val, y_val, epochs)
    model_loss_chart = pred.make_model_loss_chart(stock_history, ticker_str)

    prediction, real_data = pred.unscale_data(model, x_test_t, y_test_t, scaler)
    prediction_chart = pred.make_prediction_chart(prediction, real_data, ticker_str)

    tom_price = pred.predict_tomorrows_price(stock_data, model, scaler, y_test_t)

    return render(request, 'Stocks/recurrent.html', context={
        'title': 'Recurrent',
        'epochs': epochs,
        'stock_name': ticker_str,
        'stock_info': stock_info,
        'tom_price': tom_price,
        'model_loss_chart': model_loss_chart,
        'prediction_chart': prediction_chart,
    })


def randomForest(request):
    ticker_str = request.session.get('ticker_str')
    if not ticker_str:
        return render(request, 'Stocks/home.html', {
            'form': TickerForm(),
            'error': 'Please enter a stock ticker first.',
        })

    rf = RFClassifier()
    rf_data = rf.get_data(ticker_str)
    X, y = rf.make_features(rf_data)
    X_train, X_test, y_train, y_test = rf.split_data(X, y, rf_data)
    model = rf.make_model(X_train, X_test, y_train, y_test)
    histogram_chart, returns_chart = rf.make_charts(model, rf_data, X, ticker_str)
    tom_price = rf.predict_tomorrow(model, X)

    return render(request, 'Stocks/random-forests.html', context={
        'title': 'Random Forest Classifier',
        'stock_name': ticker_str,
        'tom_price': tom_price,
        'histogram_chart': histogram_chart,
        'returns_chart': returns_chart,
    })


def xgboost(request):
    ticker_str = request.session.get('ticker_str')
    if not ticker_str:
        return render(request, 'Stocks/home.html', {
            'form': TickerForm(),
            'error': 'Please enter a stock ticker first.',
        })

    xgb_model = XGBStockModel()
    data = xgb_model.get_data(ticker_str)
    df = xgb_model.engineer_features(data)
    results = xgb_model.train(df)

    feature_chart = xgb_model.make_feature_importance_chart(results['feature_importance'])
    backtest_chart = xgb_model.make_backtest_chart(
        results['cumulative'], results['buy_hold'])

    current_price = round(float(data['Close'].iloc[-1]), 2)
    avg_move = df['return_1d'].abs().tail(20).mean()
    if results['direction'] == 'up':
        implied_price = round(current_price * (1 + avg_move), 2)
    else:
        implied_price = round(current_price * (1 - avg_move), 2)

    # Sentiment + ensemble signal (if API key configured)
    sentiment = None
    ensemble = None
    try:
        sentiment = analyze_sentiment(ticker_str)
        if sentiment:
            xgb_signal = (results['confidence'] / 100) * (1 if results['direction'] == 'up' else -1)
            sent_signal = sentiment['sentiment_score'] / 100
            combined = 0.7 * xgb_signal + 0.3 * sent_signal
            sent_direction = 'up' if sentiment['sentiment_score'] > 0 else 'down'
            ensemble = {
                'direction': 'up' if combined > 0 else 'down',
                'confidence': round(min(abs(combined) * 100, 100), 1),
                'agreement': results['direction'] == sent_direction,
            }
    except Exception:
        pass

    return render(request, 'Stocks/xgboost.html', context={
        'title': 'XGBoost',
        'stock_name': ticker_str,
        'direction': results['direction'],
        'confidence': results['confidence'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'current_price': current_price,
        'implied_price': implied_price,
        'feature_chart': feature_chart,
        'backtest_chart': backtest_chart,
        'sentiment': sentiment,
        'ensemble': ensemble,
    })


def paper_trading(request):
    trader = PaperTrader()
    live = LiveTrader()
    results = None
    performance_chart = ''
    drawdown_chart = ''
    trades = []
    active_tab = 'backtest'

    if request.method == 'POST':
        action = request.POST.get('action')

        # --- backtest actions ---
        if action == 'reset':
            trader.reset()
        elif action == 'simulate':
            ticker = request.POST.get('ticker', '').upper().strip()
            initial_cash = float(request.POST.get('initial_cash', 10000))
            confidence = float(request.POST.get('confidence_threshold', 55))

            try:
                results = trader.run_backtest(ticker, initial_cash, confidence)
                trades = trader.trades
                performance_chart = trader.make_performance_chart(
                    trader.snapshots, initial_cash, trades)
                drawdown_chart = trader.make_drawdown_chart(trader.snapshots)
            except Exception as e:
                return render(request, 'Stocks/paper-trading.html', {
                    'title': 'Paper Trading',
                    'error': str(e),
                })

        # --- live tracking actions ---
        elif action == 'live_start':
            active_tab = 'live'
            ticker = request.POST.get('live_ticker', '').upper().strip()
            cash = float(request.POST.get('live_cash', 10000))
            conf = float(request.POST.get('live_confidence', 55))
            try:
                live.start(ticker, cash, conf)
            except Exception as e:
                return render(request, 'Stocks/paper-trading.html', {
                    'title': 'Paper Trading',
                    'error': f'Live tracking error: {e}',
                })

        elif action == 'live_retrain':
            active_tab = 'live'
            try:
                live.retrain()
            except Exception as e:
                pass  # non-critical

        elif action == 'live_reset':
            active_tab = 'live'
            live.reset()

    else:
        # GET — reload backtest if available
        if trader.last_backtest and trader.snapshots:
            results = trader.last_backtest
            trades = trader.trades
            performance_chart = trader.make_performance_chart(
                trader.snapshots, trader.portfolio['initial_cash'], trades)
            drawdown_chart = trader.make_drawdown_chart(trader.snapshots)

    # --- always update live tracker on page load ---
    live_status = None
    live_chart = ''
    if live.is_active:
        try:
            live.update()
        except Exception:
            pass  # don't crash the page if live update fails
        live_status = live.get_status()
        live_chart = live.make_performance_chart()
        if active_tab != 'live' and not results:
            active_tab = 'live'
    elif live.config:
        live_status = live.get_status()

    return render(request, 'Stocks/paper-trading.html', context={
        'title': 'Paper Trading',
        'active_tab': active_tab,
        'results': results,
        'performance_chart': performance_chart,
        'drawdown_chart': drawdown_chart,
        'trades': trades,
        'live_status': live_status,
        'live_chart': live_chart,
        'live_predictions': list(reversed(live.predictions[-20:])),
        'live_trades': list(reversed(live.trades[-20:])),
    })


def crypto(request):
    crypto_form = CryptoForm()
    crypto_history_chart = ''
    crypto_info = None

    if request.method == "POST":
        crypto_form = CryptoForm(request.POST)
        if crypto_form.is_valid():
            crypto_str = crypto_form.cleaned_data['crypto']
            try:
                coin = Crypto(crypto_str)
                data = coin.get_data()
                crypto_info = coin.get_crypto_info()
                crypto_history_chart = coin.get_crypto_candlestick(data)
                request.session['crypto_str'] = crypto_str
            except Exception as e:
                crypto_info = {'error': str(e)}
            crypto_form = CryptoForm()

    return render(request, 'Stocks/crypto.html', context={
        'title': 'Crypto',
        'form': crypto_form,
        'crypto_history_chart': crypto_history_chart,
        'crypto_info': crypto_info,
    })


def cryptoModel(request):
    crypto_str = request.session.get('crypto_str')
    if not crypto_str:
        return render(request, 'Stocks/crypto.html', {
            'form': CryptoForm(),
            'error': 'Please enter a cryptocurrency first.',
        })

    coin = Crypto(crypto_str)
    data = coin.get_data()
    tom_price, model_loss_chart, prediction_chart = coin.make_model(data)

    return render(request, 'Stocks/crypto-model.html', context={
        'title': 'Crypto Model',
        'crypto_name': crypto_str,
        'tom_price': tom_price,
        'model_loss_chart': model_loss_chart,
        'prediction_chart': prediction_chart,
    })
