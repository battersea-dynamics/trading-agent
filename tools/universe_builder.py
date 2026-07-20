"""
tools/universe_builder.py

Dynamic replacement for the static config/universe.py list. Instead of
143 hand-picked names, ask Alpaca for every tradable US equity and
keep whatever currently clears a liquidity bar. A mover can now be
caught anywhere in the market — the trade-off is that we must filter
hard, because most of the ~13,000 tradable symbols are exactly the
illiquid junk the static list existed to avoid.

Two-stage filter, cheapest first:

  1. Metadata (free, one API call): active + tradable, listed on
     NASDAQ/NYSE/AMEX (ARCA and BATS listings are overwhelmingly ETFs,
     OTC is untouchable), and a plain alphabetic symbol — dots mark
     preferred shares, units, and warrants (MS.PRO, AAC.U), which
     aren't day-tradable equities in any useful sense.
  2. Price/volume (costs batched bar requests): last close >= $3
     (sub-$3 names have spreads and halts that eat any edge) and
     average daily volume >= 500k shares (we need to enter AND exit
     without moving the price).

The result is cached to data/universe.json with a build date. Building
takes ~20 batched bar calls over ~8,000 symbols; the cache means the
rest of the pipeline can reload it instantly, and a scheduled daily_scan
simply rebuilds when the cache isn't from today.
"""

import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from tools.market_data import sip_safe_end

load_dotenv()

MIN_PRICE = 3.0
MIN_AVG_VOLUME = 500_000
ALLOWED_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}

# ETPs listed on NASDAQ/NYSE slip past the exchange filter (AAPD is a
# NASDAQ-listed 1x-inverse-AAPL fund). They're poison for a scanner
# that hunts unusual moves — a leveraged fund moves "unusually" every
# single day by construction. No is_etf flag exists in Alpaca's asset
# metadata, so filter on fund-ish names; spot-checked against the full
# asset list, this catches ~5,300 ETPs with no operating-company false
# positives (Ball Corp, Gold.com etc. survive).
FUND_NAME_PATTERN = re.compile(
    r"ETF|ETN|Fund|Trust, Series|ProShares|Direxion|GraniteShares"
    r"|iShares|Leverage|\b[123](\.5)?X\b|Daily (Bull|Bear|Long|Short)",
    re.IGNORECASE,
)
LIQUIDITY_WINDOW_DAYS = 15   # calendar days of bars for the ADV check
CHUNK_SIZE = 500             # symbols per bars request (keeps URLs sane)
CACHE_PATH = Path("data/universe.json")

_trading_client = TradingClient(
    os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True
)
_data_client = StockHistoricalDataClient(
    os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def list_candidate_symbols() -> list[str]:
    """Stage 1: metadata filter. One API call, no market data needed."""
    assets = _trading_client.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    )
    return sorted(
        a.symbol for a in assets
        if a.tradable
        and a.exchange.value in ALLOWED_EXCHANGES
        and a.symbol.isalpha()
        and not FUND_NAME_PATTERN.search(a.name or "")
    )


def filter_by_liquidity(symbols: list[str]) -> list[dict]:
    """
    Stage 2: keep symbols whose last close >= MIN_PRICE and whose
    average daily volume over the window >= MIN_AVG_VOLUME.

    Bars are fetched in chunks of CHUNK_SIZE symbols: one request for
    8,000 symbols would blow past URL length limits, while 8,000
    individual requests would take forever and hammer the rate limit.
    ~16 chunked calls is the middle ground. alpaca-py handles result
    pagination within each chunk transparently.
    """
    survivors = []
    start = datetime.now() - timedelta(days=LIQUIDITY_WINDOW_DAYS)
    for chunk in _chunked(symbols, CHUNK_SIZE):
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start,
            end=sip_safe_end(),   # free-tier SIP limit, see tools/market_data.py
            adjustment=Adjustment.ALL,
        )
        bars_by_symbol = _data_client.get_stock_bars(request).data
        for symbol, bars in bars_by_symbol.items():
            if len(bars) < 5:  # too few trading days = stale/halted
                continue
            close = bars[-1].close
            avg_volume = sum(b.volume for b in bars) / len(bars)
            if close >= MIN_PRICE and avg_volume >= MIN_AVG_VOLUME:
                survivors.append({
                    "symbol": symbol,
                    "close": close,
                    "avg_volume": int(avg_volume),
                })
    return survivors


def build_universe(cache_path: Path = CACHE_PATH) -> list[str]:
    """Full rebuild: metadata filter -> liquidity filter -> cache."""
    candidates = list_candidate_symbols()
    survivors = filter_by_liquidity(candidates)
    symbols = sorted(s["symbol"] for s in survivors)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "built_on": date.today().isoformat(),
        "criteria": {
            "min_price": MIN_PRICE,
            "min_avg_volume": MIN_AVG_VOLUME,
            "exchanges": sorted(ALLOWED_EXCHANGES),
        },
        "candidates_screened": len(candidates),
        "symbols": symbols,
    }, indent=2))
    return symbols


def load_universe(
    cache_path: Path = CACHE_PATH,
    max_age_days: int = 1,
) -> list[str]:
    """
    Return the cached universe if it was built recently, else rebuild.
    This is the entry point the pipeline uses: on a scheduled daily
    run the first call of the day pays the rebuild cost and everything
    after reads the file.
    """
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        built_on = date.fromisoformat(cached["built_on"])
        if (date.today() - built_on).days < max_age_days and cached["symbols"]:
            return cached["symbols"]
    return build_universe(cache_path)


if __name__ == "__main__":
    import time
    t0 = time.time()
    symbols = build_universe()
    print(f"universe: {len(symbols)} symbols in {time.time() - t0:.0f}s")
    print("sample:", symbols[:10], "...", symbols[-5:])
