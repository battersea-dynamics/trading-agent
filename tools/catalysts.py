"""
tools/catalysts.py

Stage 2 of the daily pipeline: context for the shortlist only.

The scanner (stage 1) answers "what is moving?" — this module answers
"is there a *reason* it's moving, or about to?" Three sources:

  earnings   - Finnhub earnings calendar. An earnings date inside the
               next few days is the single most common cause of a big
               overnight gap — for or against you. The signal agent
               needs to know it's there.
  dividends  - Alpaca's corporate-actions endpoint. An ex-dividend
               date matters for an intraday system because the price
               mechanically drops by the dividend on the ex date, which
               can trip a stop-loss that had nothing to do with the
               trade thesis.
  news       - Finnhub company news headlines. Raw text — no scoring
               here; interpreting headlines is precisely the judgment
               call we're saving the LLM for.

This runs only on the 10-20 shortlisted names, not the whole universe:
Finnhub's free tier is rate-limited (60 calls/min) and needs one call
per symbol per endpoint. 143 tickers would blow through that; 15 fits
comfortably.

Plain dicts out. Like the scanner, there is deliberately no LLM here —
gathering evidence and judging evidence are separate stages.
"""

import os
import sys
import time
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

from alpaca.data.enums import CorporateActionsType
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import CorporateActionsRequest

load_dotenv()

FINNHUB_BASE = "https://finnhub.io/api/v1"
EARNINGS_AHEAD_DAYS = 14
NEWS_BACK_DAYS = 7
MAX_HEADLINES = 5

_corp_actions_client = CorporateActionsClient(
    os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
)


# Finnhub's free tier allows 60 calls/min but also enforces a burst
# limit - fire 30 calls back-to-back and it starts resetting
# connections (found empirically: WinError 10054 mid-report). All
# Finnhub traffic funnels through this one helper, so pacing and
# retries live here and every caller inherits them.
_MIN_CALL_INTERVAL = 1.1   # seconds; ~55 calls/min, under the 60 cap
_RETRIES = 3
_last_call_time = 0.0


def _finnhub_get(path: str, params: dict) -> dict | list:
    global _last_call_time
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing FINNHUB_API_KEY - get a free key at "
            "https://finnhub.io/register and add it to .env"
        )

    for attempt in range(_RETRIES):
        wait = _MIN_CALL_INTERVAL - (time.monotonic() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()

        try:
            response = requests.get(
                f"{FINNHUB_BASE}/{path}",
                params={**params, "token": key},
                timeout=15,
            )
            if response.status_code == 429 and attempt < _RETRIES - 1:
                print(f"[catalysts] 429 from finnhub/{path}, backing off",
                      file=sys.stderr)
                time.sleep(10 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            if attempt == _RETRIES - 1:
                raise
            print(f"[catalysts] connection reset on finnhub/{path}, retrying",
                  file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def get_upcoming_earnings(symbol: str) -> list[dict]:
    """Earnings events for `symbol` in the next EARNINGS_AHEAD_DAYS."""
    today = date.today()
    data = _finnhub_get("calendar/earnings", {
        "symbol": symbol,
        "from": today.isoformat(),
        "to": (today + timedelta(days=EARNINGS_AHEAD_DAYS)).isoformat(),
    })
    return [
        {
            "date": e.get("date"),
            "hour": e.get("hour"),          # bmo = before open, amc = after close
            "eps_estimate": e.get("epsEstimate"),
            "revenue_estimate": e.get("revenueEstimate"),
        }
        for e in data.get("earningsCalendar", [])
    ]


def get_recent_news(symbol: str) -> list[dict]:
    """Most recent headlines for `symbol`, newest first, capped."""
    today = date.today()
    articles = _finnhub_get("company-news", {
        "symbol": symbol,
        "from": (today - timedelta(days=NEWS_BACK_DAYS)).isoformat(),
        "to": today.isoformat(),
    })
    articles = sorted(articles, key=lambda a: a.get("datetime", 0), reverse=True)
    return [
        {
            "headline": a.get("headline"),
            "source": a.get("source"),
            "date": date.fromtimestamp(a["datetime"]).isoformat()
            if a.get("datetime") else None,
        }
        for a in articles[:MAX_HEADLINES]
    ]


def get_upcoming_dividends(symbols: list[str]) -> dict[str, list[dict]]:
    """
    Cash dividends with ex-dates in the next two weeks, for all
    shortlisted symbols in one batched call (this endpoint accepts a
    symbol list, unlike the Finnhub ones).

    Subtlety found by testing: Alpaca's start/end filter applies to the
    *process date* (roughly the payable date), which trails the ex-date
    by weeks. So we query a wide future process window and filter on
    ex_date ourselves — querying start=today, end=today+14 directly
    would miss nearly every upcoming ex-date.
    """
    today = date.today()
    request = CorporateActionsRequest(
        symbols=symbols,
        types=[CorporateActionsType.CASH_DIVIDEND],
        start=today,
        end=today + timedelta(days=75),
    )
    data = _corp_actions_client.get_corporate_actions(request).data
    dividends: dict[str, list[dict]] = {}
    for action in data.get("cash_dividends", []):
        if action.ex_date is None or not (
            today <= action.ex_date <= today + timedelta(days=14)
        ):
            continue
        dividends.setdefault(action.symbol, []).append({
            "ex_date": action.ex_date.isoformat() if action.ex_date else None,
            "payable_date": action.payable_date.isoformat()
            if action.payable_date else None,
            "rate": action.rate,
        })
    return dividends


def prescan_earnings(
    universe: list[str],
    days_ahead: int = 3,
) -> dict[str, dict]:
    """
    PRE-SCAN mode: which of the (possibly thousands of) universe
    symbols report earnings in the next `days_ahead` days?

    This is a different animal from get_upcoming_earnings above, and
    the difference is why both exist. The shortlist-stage functions ask
    Finnhub about ONE symbol in depth — fine for 15 names, impossible
    for 2,000+ on a 60-calls/min free tier. But the earnings-calendar
    endpoint called WITHOUT a symbol returns every company reporting
    in the window in a single response, so the pre-scan is: one bulk
    call, then intersect with our universe in memory. Rate limits stop
    being a concern entirely.

    Returns {symbol: {"date", "hour", "days_until"}} for flagged
    symbols only. The scanner uses the keys as its boost set; the
    values ride along into the shortlist file so the signal agent can
    later see *why* something was flagged.
    """
    today = date.today()
    data = _finnhub_get("calendar/earnings", {
        "from": today.isoformat(),
        "to": (today + timedelta(days=days_ahead)).isoformat(),
    })
    universe_set = set(universe)
    flagged: dict[str, dict] = {}
    for event in data.get("earningsCalendar", []):
        symbol = event.get("symbol")
        if symbol not in universe_set or not event.get("date"):
            continue
        event_date = date.fromisoformat(event["date"])
        # keep the soonest event per symbol
        existing = flagged.get(symbol)
        if existing and existing["date"] <= event["date"]:
            continue
        flagged[symbol] = {
            "date": event["date"],
            "hour": event.get("hour"),  # bmo/amc/dmh or blank
            "days_until": (event_date - today).days,
        }
    return flagged


def build_catalyst_report(symbols: list[str]) -> dict[str, dict]:
    """
    One dict per symbol: {"earnings": [...], "dividends": [...],
    "news": [...]}. This is the blob the signal agent gets alongside
    the scanner metrics.
    """
    dividends = get_upcoming_dividends(symbols)
    report = {}
    for symbol in symbols:
        report[symbol] = {
            "earnings": get_upcoming_earnings(symbol),
            "dividends": dividends.get(symbol, []),
            "news": get_recent_news(symbol),
        }
    return report


if __name__ == "__main__":
    import json
    import sys

    symbols = sys.argv[1:] or ["AAPL", "NVDA"]
    print(json.dumps(build_catalyst_report(symbols), indent=2))
