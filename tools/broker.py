"""
tools/broker.py

This is NOT an agent. It's the harness — the reliable, boring layer
that talks to Alpaca. Agents will call these functions as "tools".

Why separate this out? Two reasons that matter once things get more complex:
1. You can test/trust this layer on its own, independent of any LLM.
2. Every agent (research, risk, execution) shares the exact same
   connection instead of each reinventing it.
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. "
        "Create a .env file (see .env.example) with your PAPER account keys."
    )

# paper=True is critical — this must never point at a live account
# until you've decided that deliberately, with your eyes open.
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_account():
    """Return basic account state: cash, buying power, portfolio value."""
    account = trading_client.get_account()
    return {
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
    }


def get_quote(symbol: str):
    """Return the latest bid/ask quote for a symbol."""
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data_client.get_stock_latest_quote(request)[symbol]
    return {
        "symbol": symbol,
        "bid": float(quote.bid_price),
        "ask": float(quote.ask_price),
    }


def place_market_order(symbol: str, qty: float, side: str):
    """
    Place a paper market order. side must be 'buy' or 'sell'.
    Deliberately the ONLY function in this file that touches money —
    every agent that wants to trade has to come through here, which
    is where we'll later force every order through the risk agent.
    """
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(order_request)
    return {
        "id": str(order.id),
        "symbol": order.symbol,
        "qty": order.qty,
        "side": order.side.value,
        "status": order.status.value,
    }


if __name__ == "__main__":
    # Quick smoke test — run this file directly to confirm the
    # connection works before building anything on top of it.
    print("Account:", get_account())
    print("Quote AAPL:", get_quote("AAPL"))
