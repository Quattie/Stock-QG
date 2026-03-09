# Stawks - Quantitative Stock Analysis & Trading Platform

## Vision
Build a retail quant platform: ML-driven market signals, automated trade execution,
and portfolio management — all in a Django web app.

## Tech Stack
- **Backend**: Django 5.1, Python 3.11+
- **ML**: XGBoost (primary), LSTM/Keras (legacy), scikit-learn
- **Data**: yfinance (market data), Anthropic Claude API (sentiment)
- **Charts**: Plotly (interactive, dark theme)
- **Broker**: Alpaca Markets (planned — paper trading first, then live)

## Key Files
- `Stocks/predictions.py` — LSTM model (legacy), market overview, stock data
- `Stocks/XGBModel.py` — XGBoost classifier with technical indicator features
- `Stocks/RFClassifier.py` — Random Forest classifier (legacy)
- `Stocks/paper_trader.py` — Paper trading simulator with XGBoost backtest engine
- `Stocks/Crypto.py` — Crypto LSTM model
- `Stocks/sentiment.py` — Claude-powered news sentiment analysis
- `Stocks/views.py` — All view functions
- `Stocks/urls.py` — URL routing
- `data/` — Paper trading state (portfolio.json, trades.json, snapshots.json, last_backtest.json)

## Architecture
```
User → Django views → { yfinance data, ML models, Claude sentiment }
                     → Plotly charts → Templates → Browser
```

Models run synchronously in the request cycle. Long-running training (LSTM)
blocks the web worker — async execution is on the backlog.

## Dev Notes
- Market overview data is cached in-process for 5 minutes (`_market_cache`)
- LSTM was fixed: sigmoid→linear output, dropout 0.5→0.15, timesteps 5→30,
  column order bug (was predicting Low instead of Close)
- XGBoost is the primary model — classification (up/down) with confidence %
- Paper trading simulator runs day-by-day backtests, saves state to `data/` as JSON
- Paper trader uses confidence threshold (default 55%) to filter low-conviction signals
- **Live Tracker** (`LiveTrader`): trains XGBoost on all data, makes forward predictions
  daily, auto-updates on each page load, evaluates accuracy against actual market data.
  Model persisted via joblib. Retrain button refreshes model with latest data.
- `ANTHROPIC_API_KEY` must come from console.anthropic.com (separate from claude.ai subscription)

## Broker Integration Notes
Fidelity does not offer a public retail trading API. Best alternatives:
- **Alpaca Markets** — Free, commission-free, paper trading mode, clean REST + Python SDK
- **Interactive Brokers** — TWS API, institutional-grade, more complex setup
Recommendation: Start with Alpaca paper trading, then graduate to live.

---

## Backlog (prioritized)

### P0 — Next Up
- [ ] Make market overview cards clickable → auto-submit search for that ticker
- [ ] Add 5-day sparkline mini-charts to market overview cards
- [ ] Walk-forward cross-validation for XGBoost (rolling retrain windows)
- [ ] Show XGBoost prediction confidence history (how confident was it over time?)
- [ ] Alpaca paper trading integration — place real paper trades via API
- [ ] Multi-ticker paper trading — simulate portfolio across multiple stocks
- [ ] Add transaction costs / slippage to paper trader backtest
- [ ] Sharpe ratio calculation on paper trading results

### P1 — Short Term
- [ ] Sector ETF heatmap panel (XLK, XLF, XLE, XLV, etc.)
- [ ] Session-based watchlist (add tickers, persist in session/DB)
- [ ] "Top movers" section using yfinance screener data
- [ ] Ensemble model: combine XGBoost + sentiment + LSTM signals
- [ ] LSTM: add technical indicator features to input (RSI, MACD, etc.)
- [ ] LSTM: predict returns instead of price levels
- [ ] Position sizing logic (Kelly criterion or fixed fractional)

### P2 — Medium Term
- [ ] Alpaca live trading with risk management (stop losses, daily loss limits)
- [ ] Portfolio tracker with Django ORM model (positions, P&L, history)
- [ ] AJAX/HTMX auto-refresh market prices every 60s (no full page reload)
- [ ] Async ML training via Celery so long runs don't block web worker
- [ ] Economic calendar / macro data panel
- [ ] Multi-ticker comparison view
- [ ] XGBoost hyperparameter tuning (Optuna or grid search)
- [ ] Multiple prediction horizons (1d, 5d, 20d)

### P3 — Larger Refactors
- [ ] Django cache framework (Redis) to replace module-level dict cache
- [ ] User accounts + persistent watchlists/portfolios
- [ ] Real-time websocket price streaming
- [ ] Options flow data integration
- [ ] Backtesting framework with transaction costs and slippage
- [ ] Model registry — save/load trained models, track performance over time
- [ ] Alerting system (email/SMS when model signals trigger)
