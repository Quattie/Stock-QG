"""
Alpaca Markets integration for paper trading via API.

Wraps the Alpaca SDK to place orders, fetch positions, and track
account status. Connects to XGBoost strategy signals.
"""

import os
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


def _get_client():
    """Build TradingClient from env vars. Returns None if not configured."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET')
    if not api_key or not secret:
        return None
    paper = os.getenv('ALPACA_PAPER', 'true').lower() in ('true', '1', 'yes')
    return TradingClient(api_key, secret, paper=paper)


def _get_data_client():
    """Build StockHistoricalDataClient for market data."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET')
    if not api_key or not secret:
        return None
    return StockHistoricalDataClient(api_key, secret)


class AlpacaTrader:

    def __init__(self):
        self.client = _get_client()
        self.data_client = _get_data_client()

    @property
    def is_configured(self):
        return self.client is not None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_summary(self):
        """Return account info dict for templates."""
        if not self.client:
            return None
        account = self.client.get_account()
        return {
            'status': str(account.status).split('.')[-1],
            'cash': round(float(account.cash), 2),
            'portfolio_value': round(float(account.portfolio_value), 2),
            'buying_power': round(float(account.buying_power), 2),
            'equity': round(float(account.equity), 2),
            'long_market_value': round(float(account.long_market_value), 2),
            'short_market_value': round(float(account.short_market_value), 2),
            'pnl': round(float(account.equity) - 100000, 2),  # paper starts at 100k
            'pnl_pct': round((float(account.equity) - 100000) / 100000 * 100, 2),
            'daytrade_count': account.daytrade_count,
            'paper': os.getenv('ALPACA_PAPER', 'true').lower() in ('true', '1', 'yes'),
        }

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self):
        """Return list of open positions as dicts."""
        if not self.client:
            return []
        positions = self.client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                'symbol': p.symbol,
                'qty': int(float(p.qty)),
                'side': str(p.side).split('.')[-1].lower(),
                'avg_entry': round(float(p.avg_entry_price), 2),
                'current_price': round(float(p.current_price), 2),
                'market_value': round(float(p.market_value), 2),
                'unrealized_pnl': round(float(p.unrealized_pl), 2),
                'unrealized_pnl_pct': round(float(p.unrealized_plpc) * 100, 2),
                'change_today': round(float(p.change_today) * 100, 2),
            })
        return result

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, ticker, qty, side='buy'):
        """Place a market order. Returns order dict or error string."""
        if not self.client:
            return {'error': 'Alpaca not configured'}

        order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL

        order_data = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(order_data)
        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': str(order.qty),
            'side': side,
            'status': str(order.status).split('.')[-1],
            'submitted_at': str(order.submitted_at),
        }

    def get_recent_orders(self, limit=20):
        """Fetch recent orders (filled + pending)."""
        if not self.client:
            return []

        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=limit,
        )
        orders = self.client.get_orders(request)
        result = []
        for o in orders:
            filled_price = None
            if o.filled_avg_price:
                filled_price = round(float(o.filled_avg_price), 2)

            result.append({
                'id': str(o.id),
                'symbol': o.symbol,
                'qty': str(o.qty),
                'filled_qty': str(o.filled_qty) if o.filled_qty else '0',
                'side': str(o.side).split('.')[-1].lower(),
                'status': str(o.status).split('.')[-1].lower(),
                'filled_price': filled_price,
                'submitted_at': o.submitted_at.strftime('%Y-%m-%d %H:%M') if o.submitted_at else '',
                'filled_at': o.filled_at.strftime('%Y-%m-%d %H:%M') if o.filled_at else '',
            })
        return result

    def close_position(self, ticker):
        """Close all shares of a position."""
        if not self.client:
            return {'error': 'Alpaca not configured'}
        self.client.close_position(ticker.upper())
        return {'status': 'closed', 'symbol': ticker.upper()}

    def close_all_positions(self):
        """Liquidate everything."""
        if not self.client:
            return {'error': 'Alpaca not configured'}
        self.client.close_all_positions(cancel_orders=True)
        return {'status': 'all positions closed'}

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    def get_latest_quote(self, ticker):
        """Get current bid/ask for a ticker."""
        if not self.data_client:
            return None
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker.upper())
        quotes = self.data_client.get_stock_latest_quote(request)
        q = quotes[ticker.upper()]
        return {
            'bid': round(float(q.bid_price), 2),
            'ask': round(float(q.ask_price), 2),
            'bid_size': q.bid_size,
            'ask_size': q.ask_size,
            'midpoint': round((float(q.bid_price) + float(q.ask_price)) / 2, 2),
        }

    # ------------------------------------------------------------------
    # Strategy-driven order sizing
    # ------------------------------------------------------------------

    def calculate_order(self, ticker, strategy_action, current_price,
                        max_pct_of_buying_power=0.25):
        """
        Given a strategy signal, calculate how many shares to trade.
        Returns dict with suggested order details or None if no action.
        """
        if not self.client:
            return None

        account = self.client.get_account()
        buying_power = float(account.buying_power)
        positions = {p.symbol: p for p in self.client.get_all_positions()}

        has_position = ticker.upper() in positions

        if strategy_action in ('Strong Buy', 'Buy') and not has_position:
            budget = buying_power * max_pct_of_buying_power
            qty = int(budget / current_price)
            if qty < 1:
                return None
            return {
                'side': 'buy',
                'qty': qty,
                'estimated_cost': round(qty * current_price, 2),
                'pct_of_buying_power': round(qty * current_price / buying_power * 100, 1),
            }

        elif strategy_action in ('Strong Sell', 'Sell / Avoid') and has_position:
            pos = positions[ticker.upper()]
            qty = int(float(pos.qty))
            return {
                'side': 'sell',
                'qty': qty,
                'estimated_value': round(qty * current_price, 2),
                'unrealized_pnl': round(float(pos.unrealized_pl), 2),
            }

        elif strategy_action == 'Accumulate on Dips' and not has_position:
            # Smaller position — half the normal size
            budget = buying_power * max_pct_of_buying_power * 0.5
            qty = int(budget / current_price)
            if qty < 1:
                return None
            return {
                'side': 'buy',
                'qty': qty,
                'estimated_cost': round(qty * current_price, 2),
                'pct_of_buying_power': round(qty * current_price / buying_power * 100, 1),
                'note': 'Half-size position (accumulate strategy)',
            }

        return None
