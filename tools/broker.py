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
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
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


def get_positions():
    """
    Return currently held positions: symbol, quantity, current value.
    Quantity stays a float — Alpaca reports fractional shares (e.g.
    from partial fills), and rounding here would misstate holdings.
    """
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": float(p.market_value),
        }
        for p in trading_client.get_all_positions()
    ]


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
    One of only two functions in this file that touch money (the
    other is place_bracket_order below) — every agent that wants to
    trade has to come through here, which is where we'll later force
    every order through the risk agent.
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


def place_bracket_order(
    symbol: str,
    qty: int,
    take_profit_price: float,
    stop_loss_price: float,
):
    """
    Buy `qty` shares at market with an attached take-profit (limit
    sell) and stop-loss (stop sell) in one atomic submission. Alpaca
    holds and manages the exit legs server-side: whichever triggers
    first fills and cancels the other (OCO). No agent needs to watch
    the position after entry — that's the point.

    Notes that shape the signature:
      - qty is an int: Alpaca does not allow fractional bracket orders.
      - exit prices are absolute dollars, not percentages. Converting
        "take profit +4%" into a price requires a reference price and
        rounding decisions, and that judgment belongs to the caller
        (the execution agent), not the harness.
      - long-only entry (buy), matching the system design.
      - GTC, not DAY: the exit legs must outlive the trading day.
        With DAY, an unfilled take-profit/stop-loss would expire at
        the close, leaving the position held overnight with no
        protection and no one watching. GTC keeps both legs live
        until one of them fills (Alpaca caps GTC at 90 days).
    """
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
    )
    order = trading_client.submit_order(order_request)
    return {
        "id": str(order.id),
        "symbol": order.symbol,
        "qty": order.qty,
        "side": order.side.value,
        "status": order.status.value,
        "take_profit": round(take_profit_price, 2),
        "stop_loss": round(stop_loss_price, 2),
        "legs": [str(leg.id) for leg in (order.legs or [])],
    }


if __name__ == "__main__":
    # Quick smoke test — run this file directly to confirm the
    # connection works before building anything on top of it.
    print("Account:", get_account())
    print("Quote AAPL:", get_quote("AAPL"))
