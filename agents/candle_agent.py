"""
agents/candle_agent.py

Reads two candles per stock — yesterday's completed daily candle and
today's aggregated pre-market candle — and judges the pattern:
accumulation, exhaustion, reversal, continuation, or indecision.

Numeric OHLC in, judgment out. Why this is an LLM job at all: the
individual facts (gapped above yesterday's high, closed near the top
of its range, range expanded 3x) are arithmetic, but naming the
PATTERN those facts add up to is interpretation over several
interacting signals — the kind of "read" a chart-literate human does
at a glance and a rule tree does badly.

Design decision — the code does the division, the model does the
reading. Every ratio the pattern judgment needs (gap position
relative to yesterday's range, where the PM close sits inside the PM
range, PM range vs yesterday's range, upper/lower wick proportions)
is pre-computed and handed over as a labeled number. LLMs are
unreliable calculators; making the model derive ratios from raw OHLC
would spend its capacity on the part it's worst at. Give it the
facts, ask it only for the verdict.

Conviction uses the same anchored 0-1 scale as the bull/bear agents
(0.9+ textbook / 0.5 could-be-anything / <0.3 forced) — downstream
consumers read all these numbers side by side, so they must be on
one ruler.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Literal

from crewai import Agent, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agents.llm_runner import gemini_llm, run_task
from alpaca.data.enums import Adjustment
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from tools.broker import data_client
from tools.datapaths import list_path
from tools.market_data import sip_safe_end
from tools.market_calendar import ET, is_market_open_today, is_trading_day

load_dotenv()

PM_START = dtime(4, 0)
PM_END = dtime(9, 30)


class CandleRead(BaseModel):
    symbol: str
    pattern: str = Field(
        description="Named pattern, e.g. accumulation, exhaustion, "
                    "reversal, continuation, indecision",
    )
    bias: Literal["bullish", "bearish", "neutral"]
    conviction: float = Field(ge=0.0, le=1.0)


@dataclass
class CandlePair:
    symbol: str
    daily: dict     # yesterday's OHLCV
    premarket: dict # today's aggregated PM OHLCV
    derived: dict   # pre-computed ratios the model reasons over


def _fetch_candle_pair(symbol: str, session: date) -> CandlePair | None:
    daily_bars = data_client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.combine(session - timedelta(days=7), dtime(0, 0), tzinfo=ET),
        # Capped to the free plan's SIP entitlement (tools/market_data.py)
        end=sip_safe_end(datetime.combine(session, dtime(0, 0), tzinfo=ET)),
        adjustment=Adjustment.ALL,
    )).data.get(symbol, [])
    pm_bars = data_client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime.combine(session, PM_START, tzinfo=ET),
        end=sip_safe_end(datetime.combine(session, PM_END, tzinfo=ET)),
    )).data.get(symbol, [])
    if not daily_bars or not pm_bars:
        return None

    y = daily_bars[-1]  # yesterday (last completed daily before session)
    pm = {
        "open": pm_bars[0].open,
        "high": max(b.high for b in pm_bars),
        "low": min(b.low for b in pm_bars),
        "close": pm_bars[-1].close,
        "volume": int(sum(b.volume for b in pm_bars)),
    }
    daily = {"open": y.open, "high": y.high, "low": y.low,
             "close": y.close, "volume": int(y.volume)}

    y_range = max(y.high - y.low, 1e-9)
    pm_range = max(pm["high"] - pm["low"], 1e-9)
    derived = {
        # >1: PM opened above yesterday's high; <0: below its low
        "pm_open_vs_yesterday_range": round((pm["open"] - y.low) / y_range, 2),
        # 1.0 = PM closed at the top of its own range, 0.0 = at the bottom
        "pm_close_position_in_pm_range": round((pm["close"] - pm["low"]) / pm_range, 2),
        "pm_range_vs_yesterday_range": round(pm_range / y_range, 2),
        "pm_change_vs_yesterday_close_pct": round((pm["close"] - y.close) / y.close * 100, 2),
        "yesterday_close_position_in_its_range": round((y.close - y.low) / y_range, 2),
    }
    return CandlePair(symbol=symbol, daily=daily, premarket=pm, derived=derived)


def _build_candle_agent() -> Agent:
    return Agent(
        role="Candle Structure Analyst",
        goal=(
            "Read a two-candle sequence (yesterday's daily, today's "
            "pre-market) and name the pattern it forms, with an honest "
            "conviction score."
        ),
        backstory=(
            "You are a price-action specialist. You read candles as a "
            "record of who was in control and whether that control "
            "held: where a candle closes within its range, whether the "
            "next session's gap builds on or rejects the prior move, "
            "whether range expansion comes with follow-through or "
            "wick. You name what the structure shows — including "
            "'indecision' when that is the honest read. Your "
            "conviction score is not enthusiasm; it is how cleanly "
            "the evidence matches the pattern you named."
        ),
        llm=gemini_llm(),
        tools=[],
        allow_delegation=False,
        verbose=True,
    )


def _build_candle_task(agent: Agent, pair: CandlePair) -> Task:
    return Task(
        description=(
            f"Read the candle structure for {pair.symbol}.\n\n"
            f"Yesterday's completed daily candle (OHLCV):\n"
            f"{json.dumps(pair.daily, indent=2)}\n\n"
            f"Today's pre-market candle, aggregated 04:00-09:30 ET "
            f"(OHLCV):\n{json.dumps(pair.premarket, indent=2)}\n\n"
            f"Pre-computed structure (use these, don't re-derive):\n"
            f"{json.dumps(pair.derived, indent=2)}\n\n"
            "Name the pattern this two-candle sequence forms "
            "(accumulation / exhaustion / reversal / continuation / "
            "indecision, or a more precise standard name if one fits "
            "better), state the directional bias it implies for "
            "today's open, and rate your conviction:\n"
            "  0.9+  textbook - every element of the structure agrees\n"
            "  0.7   clear - the pattern is there with minor noise\n"
            "  0.5   could be anything - mixed or contradictory signals\n"
            "  <0.3  forced - you had to squint to name anything"
        ),
        expected_output=(
            "JSON: symbol, pattern (short name), bias "
            "(bullish/bearish/neutral), conviction (0.0-1.0 on the "
            "anchored scale)."
        ),
        agent=agent,
        output_pydantic=CandleRead,
    )


def run_candle_agent(
    symbols: list[str] | None = None,
    target_date: date | None = None,
    output_path: Path | None = None,
) -> dict[str, dict]:
    """
    Entry point. `symbols=None` reads the pre-market scan's shortlist
    (file seam, same pattern as the rest of the system); passing an
    explicit list keeps the component testable without the scanner.
    """
    if target_date is None:
        if not is_market_open_today():
            print("candle_agent: market closed today - nothing to do")
            return {}
        target_date = datetime.now(ET).date()
    elif not is_trading_day(target_date):
        print(f"candle_agent: {target_date} was not a trading day")
        return {}

    if output_path is None:
        output_path = list_path("premarket_candles.json", target_date)
    if symbols is None:
        scan_path = list_path("premarket_scan.json", target_date)
        if not scan_path.exists():
            raise SystemExit(f"{scan_path} not found - run the "
                             f"premarket scanner first, or pass symbols")
        payload = json.loads(scan_path.read_text())
        symbols = [r["symbol"] for r in payload["shortlist"]]

    reads: dict[str, dict] = {}
    for symbol in symbols:
        pair = _fetch_candle_pair(symbol, target_date)
        if pair is None:
            print(f"[candle] {symbol}: no usable candle data - skipped")
            continue
        agent = _build_candle_agent()
        task = _build_candle_task(agent, pair)
        read = run_task(agent, task, label="candle", symbol=symbol)
        if read is not None:
            reads[symbol] = read.model_dump()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "session_date": target_date.isoformat(),
        "reads": reads,
    }, indent=2))
    print(f"candle_agent: {len(reads)}/{len(symbols)} read -> {output_path}")
    return reads


if __name__ == "__main__":
    import sys

    symbols = sys.argv[1:] or None
    for symbol, read in run_candle_agent(symbols).items():
        print(f"{symbol:6s} {read['bias']:8s} {read['conviction']:.2f}  "
              f"{read['pattern']}")
