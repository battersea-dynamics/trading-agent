"""
tools/premarket_scanner.py

Pre-market twin of tools/scanner.py. No LLM — pure math.

Why a separate scanner instead of a flag on the existing one? The
baselines are different animals. Regular-session relative volume
compares a day to 20 prior days; pre-market volume is 1-2% of daily
volume and wildly skewed, so "3x normal pre-market volume" only means
something against a PRE-MARKET baseline (10-14 prior sessions of
4:00-9:30 activity). Mixing the two baselines in one module invites
exactly the kind of subtle unit error that ranks garbage.

Cost design — two phases, cheap filter first (same philosophy as
universe_builder):

  Phase 1: today's pre-market minute bars for the WHOLE universe.
           Sounds heavy, but pre-market is thin: most symbols print
           few or zero bars, so the response is small and the
           emptiness itself is the first filter. Keep the top
           CANDIDATE_POOL by naive activity (gap% x volume).
  Phase 2: only for those candidates, fetch 14 calendar days of
           minute bars and build the real per-symbol pre-market
           baseline. 100 symbols x 14 days is affordable; 2,500
           would not be.

Scoring mirrors the regular scanner deliberately: z-scores across
today's candidates (|z(gap)| + |z(rel_pm_volume)|) plus a raw
kicker for genuinely elevated pre-market volume. Same shape of
math -> comparable intuition when reading both shortlists.
"""

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from tools.broker import data_client
from tools.datapaths import list_path
from tools.market_calendar import ET, is_market_open_today, is_trading_day
from tools.universe_builder import load_universe

load_dotenv()

PM_START = dtime(4, 0)    # ET; Alpaca's earliest extended-hours prints
PM_END = dtime(9, 30)     # regular session open
BASELINE_CALENDAR_DAYS = 14   # ~10 trading days of pre-market history
CANDIDATE_POOL = 100
TOP_N = 12
CHUNK_SIZE = 500
MIN_PM_VOLUME = 5_000     # ignore symbols with basically no PM prints
VOLUME_KICKER_CAP = 5.0


@dataclass
class PremarketScanResult:
    symbol: str
    prev_close: float
    last_pm_price: float
    pm_gap_pct: float        # last PM price vs yesterday's close
    pm_volume: int           # today's 4:00-9:30 volume
    avg_pm_volume: int       # baseline: mean PM volume, prior sessions
    rel_pm_volume: float     # today / baseline
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _zscores(values: list[float]) -> list[float]:
    # Duplicated from tools/scanner.py on purpose: 8 lines of pure
    # math is cheaper than a dependency between two components that
    # the brief wants independently runnable and separately evolvable.
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def _pm_window(day: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(day, PM_START, tzinfo=ET),
        datetime.combine(day, PM_END, tzinfo=ET),
    )


def _fetch_pm_bars(symbols: list[str], start: datetime, end: datetime) -> dict:
    """Minute bars for [start, end), chunked, keyed by symbol."""
    bars: dict[str, list] = {}
    for chunk in _chunked(symbols, CHUNK_SIZE):
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        bars.update(data_client.get_stock_bars(request).data)
    return bars


def _prev_closes(symbols: list[str], today: date) -> dict[str, float]:
    """Yesterday's daily close per symbol (the gap reference)."""
    closes: dict[str, float] = {}
    for chunk in _chunked(symbols, CHUNK_SIZE):
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=datetime.combine(today - timedelta(days=7), dtime(0, 0), tzinfo=ET),
            end=datetime.combine(today, dtime(0, 0), tzinfo=ET),
        )
        for symbol, bars in data_client.get_stock_bars(request).data.items():
            if bars:
                closes[symbol] = bars[-1].close
    return closes


def run_premarket_scan(
    target_date: date | None = None,
    output_path: Path | None = None,
    top_n: int = TOP_N,
) -> list[PremarketScanResult]:
    """
    Entry point. `target_date` defaults to today (ET); pointing it at
    a past session is what makes this component testable at any hour
    — pre-market data is only "live" for a few hours a day, but
    history is always there.
    """
    if target_date is None:
        if not is_market_open_today():
            print("premarket_scan: market closed today - nothing to do")
            return []
        target_date = datetime.now(ET).date()
    elif not is_trading_day(target_date):
        print(f"premarket_scan: {target_date} was not a trading day")
        return []

    if output_path is None:
        output_path = list_path("premarket_scan.json", target_date)

    universe = load_universe()

    # ---- Phase 1: today's pre-market, whole universe ----
    pm_start, pm_end = _pm_window(target_date)
    today_bars = _fetch_pm_bars(universe, pm_start, pm_end)
    prev_closes = _prev_closes(list(today_bars.keys()), target_date)

    candidates = []
    for symbol, bars in today_bars.items():
        volume = int(sum(b.volume for b in bars))  # Alpaca reports float
        prev = prev_closes.get(symbol)
        if volume < MIN_PM_VOLUME or not prev or not bars:
            continue
        last_price = bars[-1].close
        gap_pct = (last_price - prev) / prev * 100
        candidates.append((symbol, last_price, prev, gap_pct, volume))

    # Naive activity ranking picks the candidate pool: |gap| weighted
    # by log-ish volume via simple product. Crude is fine here - phase
    # 2 re-ranks properly; this only decides who gets a baseline.
    candidates.sort(key=lambda c: abs(c[3]) * c[4], reverse=True)
    candidates = candidates[:CANDIDATE_POOL]
    pool = [c[0] for c in candidates]

    # ---- Phase 2: pre-market baseline for the candidate pool ----
    baseline_start = datetime.combine(
        target_date - timedelta(days=BASELINE_CALENDAR_DAYS),
        PM_START, tzinfo=ET,
    )
    history = _fetch_pm_bars(pool, baseline_start, pm_start)

    avg_pm: dict[str, float] = {}
    for symbol, bars in history.items():
        by_day: dict[date, int] = {}
        for b in bars:
            ts = b.timestamp.astimezone(ET)
            if PM_START <= ts.time() < PM_END:
                by_day[ts.date()] = by_day.get(ts.date(), 0) + b.volume
        if by_day:
            avg_pm[symbol] = sum(by_day.values()) / len(by_day)

    results = []
    for symbol, last_price, prev, gap_pct, volume in candidates:
        baseline = avg_pm.get(symbol, 0.0)
        if baseline <= 0:
            # No pre-market history at all -> today's activity is
            # infinitely unusual, which is exactly the kind of stock
            # (fresh catalyst on a normally-dead name) this scan
            # exists to find. Cap rather than divide by zero.
            rel = VOLUME_KICKER_CAP
        else:
            rel = volume / baseline
        results.append(PremarketScanResult(
            symbol=symbol, prev_close=prev, last_pm_price=last_price,
            pm_gap_pct=round(gap_pct, 2), pm_volume=volume,
            avg_pm_volume=int(baseline), rel_pm_volume=round(rel, 2),
            score=0.0,
        ))

    if results:
        z_gap = _zscores([r.pm_gap_pct for r in results])
        z_vol = _zscores([r.rel_pm_volume for r in results])
        for r, zg, zv in zip(results, z_gap, z_vol):
            kicker = max(0.0, min(r.rel_pm_volume, VOLUME_KICKER_CAP) - 1.2)
            r.score = round(abs(zg) + abs(zv) + kicker, 3)
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "session_date": target_date.isoformat(),
        "universe_size": len(universe),
        "phase1_active": len(candidates),
        "shortlist": [r.to_dict() for r in results],
    }, indent=2))
    print(f"premarket_scan: {len(universe)} symbols -> {len(results)} "
          f"shortlisted -> {output_path}")
    return results


if __name__ == "__main__":
    import sys

    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    shortlist = run_premarket_scan(target_date=target)
    for r in shortlist:
        print(f"{r.symbol:6s} gap {r.pm_gap_pct:+6.2f}%  "
              f"pmvol {r.pm_volume:>10,}  rel {r.rel_pm_volume:5.2f}x  "
              f"score {r.score:6.2f}")
