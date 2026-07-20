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

"Unusualness" score, three layers:

  1. z-scores — each metric standardized across the universe and the
     absolute values summed. Relative: "is this stock extreme
     compared to everything else *today*?" Direction doesn't matter —
     a crash is as much of a candidate as a spike.
  2. volume kicker — a reward for rel_volume crossing an *absolute*
     threshold (>1.2x, saturating at 4x). This is the
     earlier-detection layer: z-scores only ever surface the day's
     tail, so a stock quietly building 1.5-2x volume gets drowned out
     by whatever is doing 8x. The kicker scores "unusual for itself"
     independent of the cross-section, catching moves while they're
     accelerating instead of after they've peaked.
  3. catalyst boost — a flat bonus for symbols the catalyst pre-scan
     flagged (earnings within days). Volume creeping up *into* a
     known event is a fundamentally better lead than volume with no
     known cause, so flagged names should make the shortlist on
     softer numbers.

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
CHUNK_SIZE = 500            # symbols per bars request

# Sensitivity knobs for the score layers described above.
VOLUME_KICKER_FLOOR = 1.2   # rel_volume where the kicker starts paying
VOLUME_KICKER_CAP = 4.0     # ...and where it stops (z-scores own the tail)
VOLUME_KICKER_WEIGHT = 1.5  # 1.5x vol -> +0.45, 2x -> +1.2, >=4x -> +4.2
CATALYST_BOOST = 2.0        # flat bonus for pre-scan-flagged symbols

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
    score: float          # layered score; higher = more unusual
    catalyst: dict | None = None  # pre-scan flag (earnings info), if any

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_bars(symbols: list[str]) -> dict[str, list]:
    """
    Batched requests for daily bars across the whole universe, in
    chunks of CHUNK_SIZE symbols. alpaca-py accepts a symbol list per
    request (so 143 tickers was literally one HTTP call), but symbols
    travel in the URL query string — a 2,300-symbol universe has to be
    split. ~5 chunked calls, not 2,300 individual ones.
    """
    bars: dict[str, list] = {}
    start = datetime.now() - timedelta(days=CALENDAR_BUFFER_DAYS)
    for i in range(0, len(symbols), CHUNK_SIZE):
        request = StockBarsRequest(
            symbol_or_symbols=symbols[i:i + CHUNK_SIZE],
            timeframe=TimeFrame.Day,
            start=start,
            # Split/dividend-adjusted bars. Without this a 2:1 split
            # shows up as a -50% "move" and dominates the ranking.
            adjustment=Adjustment.ALL,
        )
        bars.update(_data_client.get_stock_bars(request).data)
    return bars  # dict: symbol -> list of Bar objects


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
            # Placeholder only. SECTOR_OF is the retired static
            # 143-ticker map; the dynamic universe's symbols mostly
            # aren't in it, so nearly everything would resolve to
            # "unknown" here. Real sector is filled in on the
            # shortlist by _enrich_sectors (see scan()).
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


def rank(
    results: list[ScanResult],
    top_n: int = 15,
    catalyst_flags: dict[str, dict] | None = None,
) -> list[ScanResult]:
    """
    Score = |z(rel_volume)| + |z(pct_change)| + |z(ma_distance)|
            + volume kicker + catalyst boost.

    Z-scoring means a metric only counts for how far it deviates from
    the rest of the universe *today* — on a day when everything is up
    2%, being up 2% scores near zero. Absolute values because unusual
    is unusual in either direction. The kicker and boost layers are
    explained in the module docstring: absolute-threshold volume
    detection (catch 1.5-2x builds early) and pre-scan catalyst
    awareness (volume into a known event beats volume from nowhere).
    """
    if not results:
        return []
    catalyst_flags = catalyst_flags or {}
    z_vol = _zscores([r.rel_volume for r in results])
    z_chg = _zscores([r.pct_change for r in results])
    z_ma = _zscores([r.ma_distance for r in results])
    for r, zv, zc, zm in zip(results, z_vol, z_chg, z_ma):
        kicker = VOLUME_KICKER_WEIGHT * max(
            0.0, min(r.rel_volume, VOLUME_KICKER_CAP) - VOLUME_KICKER_FLOOR
        )
        r.score = abs(zv) + abs(zc) + abs(zm) + kicker
        if r.symbol in catalyst_flags:
            r.catalyst = catalyst_flags[r.symbol]
            r.score += CATALYST_BOOST
    return sorted(results, key=lambda r: r.score, reverse=True)[:top_n]


def _enrich_sectors(results: list[ScanResult]) -> None:
    """
    Fill in a real sector for each shortlisted result, in place.

    Sector is resolved HERE (post-ranking, ~15 names) and not on the
    whole universe because it costs one Finnhub call per symbol —
    fine for a shortlist, prohibitive for ~2,300 symbols. Finnhub is
    the source because Alpaca's asset metadata has no sector field
    (see tools.catalysts.get_sector). The static SECTOR_OF map is a
    free fallback for the handful of big names it still covers, used
    only when Finnhub returns nothing (e.g. it's down or rate-limited).
    Best-effort throughout: a labelling gap never breaks the scan.

    Lazy import keeps the pure-math core (compute_metrics/rank) free of
    any Finnhub dependency — only this enrichment step reaches out.
    """
    from tools.catalysts import get_sector

    for r in results:
        sector = get_sector(r.symbol)
        if sector == "unknown":
            sector = SECTOR_OF.get(r.symbol, "unknown")
        r.sector = sector


def scan(
    top_n: int = 15,
    symbols: list[str] | None = None,
    catalyst_flags: dict[str, dict] | None = None,
) -> list[ScanResult]:
    """
    `symbols` and `catalyst_flags` are injected by the pipeline (the
    dynamic universe and the pre-scan output). Defaults fall back to
    the static list with no flags, so the scanner still runs standalone
    for a quick manual look.
    """
    bars = fetch_bars(symbols if symbols is not None else ALL_TICKERS)
    shortlist = rank(compute_metrics(bars), top_n=top_n,
                     catalyst_flags=catalyst_flags)
    _enrich_sectors(shortlist)
    return shortlist


if __name__ == "__main__":
    shortlist = scan()
    print(f"{'SYM':6s} {'SECTOR':15s} {'CLOSE':>9s} {'RVOL':>6s} "
          f"{'CHG%':>7s} {'vsMA%':>7s} {'SCORE':>6s}")
    for r in shortlist:
        flag = f" [earnings {r.catalyst['date']}]" if r.catalyst else ""
        print(f"{r.symbol:6s} {r.sector:15s} {r.close:9.2f} "
              f"{r.rel_volume:6.2f} {r.pct_change:+7.2f} "
              f"{r.ma_distance:+7.2f} {r.score:6.2f}{flag}")
