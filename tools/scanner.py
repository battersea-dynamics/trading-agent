"""
tools/scanner.py

Stage 1 of the daily pipeline: narrow ~143 tickers down to the 10-20
that are behaving unusually today. This is pure arithmetic — no LLM.

Why no LLM here? Cost and fit. Screening 143 stocks with a language
model would be slow, expensive, and worse: ranking numbers is exactly
what plain code is good at. The LLM's judgment is saved for stage 3,
where the question ("given this volume spike AND earnings on Thursday,
is this a buy?") actually requires reading and weighing context.

Metrics per stock, from daily bars:
  rel_volume   - latest day's volume / average of the prior 20 days.
                 2.0 means "trading at twice its normal volume".
  pct_change   - latest close vs previous close, in percent.
  ma_distance  - how far the latest close sits from its own 20-day
                 moving average, in percent. Stretched = interesting.

"Unusualness" score: each metric is z-scored across the universe (so
they're comparable despite different units) and the absolute values
are summed. Direction doesn't matter for the shortlist — a crash is as
much of a candidate as a spike; the signal agent decides what to do
with it.

Caveat worth knowing: if you run this during market hours, the latest
daily bar is partial — its volume only covers the session so far, so
rel_volume understates morning activity. Fine for after-close or
pre-open runs against yesterday's completed bar; just know what you're
comparing.
"""

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from dotenv import load_dotenv

from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.universe import ALL_TICKERS, SECTOR_OF

load_dotenv()

LOOKBACK_DAYS = 20          # window for avg volume and moving average
CALENDAR_BUFFER_DAYS = 45   # calendar days to fetch so we get ~20+ trading days

_data_client = StockHistoricalDataClient(
    os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
)


@dataclass
class ScanResult:
    symbol: str
    sector: str
    close: float
    rel_volume: float     # 1.0 = normal volume
    pct_change: float     # day-over-day close, percent
    ma_distance: float    # percent above (+) / below (-) the 20d MA
    score: float          # summed |z-scores|; higher = more unusual

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_bars(symbols: list[str]) -> dict[str, list]:
    """
    One batched request for daily bars across the whole universe.
    alpaca-py accepts a list of symbols and returns them keyed by
    symbol — 143 tickers is one HTTP call, not 143.
    """
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=CALENDAR_BUFFER_DAYS),
        # Split/dividend-adjusted bars. Without this a 2:1 split shows
        # up as a -50% "move" and dominates the ranking.
        adjustment=Adjustment.ALL,
    )
    barset = _data_client.get_stock_bars(request)
    return barset.data  # dict: symbol -> list of Bar objects


def compute_metrics(bars_by_symbol: dict[str, list]) -> list[ScanResult]:
    results = []
    for symbol, bars in bars_by_symbol.items():
        # Need the latest bar, the one before it, and a 20-day history
        # behind that. Skip anything with a short history (new listing,
        # halted, bad data) rather than computing garbage.
        if len(bars) < LOOKBACK_DAYS + 2:
            continue

        latest = bars[-1]
        previous = bars[-2]
        window = bars[-(LOOKBACK_DAYS + 1):-1]  # 20 bars before latest

        avg_volume = sum(b.volume for b in window) / len(window)
        moving_avg = sum(b.close for b in window) / len(window)
        if avg_volume == 0 or moving_avg == 0 or previous.close == 0:
            continue

        results.append(ScanResult(
            symbol=symbol,
            sector=SECTOR_OF.get(symbol, "unknown"),
            close=latest.close,
            rel_volume=latest.volume / avg_volume,
            pct_change=(latest.close - previous.close) / previous.close * 100,
            ma_distance=(latest.close - moving_avg) / moving_avg * 100,
            score=0.0,  # filled in below, needs the whole universe first
        ))
    return results


def _zscores(values: list[float]) -> list[float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def rank(results: list[ScanResult], top_n: int = 15) -> list[ScanResult]:
    """
    Score = |z(rel_volume)| + |z(pct_change)| + |z(ma_distance)|.

    Z-scoring first means a metric only counts for how far it deviates
    from the rest of the universe *today* — so on a day when everything
    is up 2%, being up 2% scores near zero, but on a flat day it
    stands out. Absolute values because unusual is unusual in either
    direction.
    """
    if not results:
        return []
    z_vol = _zscores([r.rel_volume for r in results])
    z_chg = _zscores([r.pct_change for r in results])
    z_ma = _zscores([r.ma_distance for r in results])
    for r, zv, zc, zm in zip(results, z_vol, z_chg, z_ma):
        r.score = abs(zv) + abs(zc) + abs(zm)
    return sorted(results, key=lambda r: r.score, reverse=True)[:top_n]


def scan(top_n: int = 15) -> list[ScanResult]:
    bars = fetch_bars(ALL_TICKERS)
    return rank(compute_metrics(bars), top_n=top_n)


if __name__ == "__main__":
    shortlist = scan()
    print(f"{'SYM':6s} {'SECTOR':15s} {'CLOSE':>9s} {'RVOL':>6s} "
          f"{'CHG%':>7s} {'vsMA%':>7s} {'SCORE':>6s}")
    for r in shortlist:
        print(f"{r.symbol:6s} {r.sector:15s} {r.close:9.2f} "
              f"{r.rel_volume:6.2f} {r.pct_change:+7.2f} "
              f"{r.ma_distance:+7.2f} {r.score:6.2f}")
