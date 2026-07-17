"""
tools/premarket_trader.py

Deterministic referee for the pre-market debate. No LLM — and the
reason that's even possible is the file seam: the bull and bear
runners each wrote their judgment to disk before this runs. This
module is pure policy over two JSON files.

Formula is IDENTICAL to tools/trader.py, constants included:

  net = bull_confidence - bear_risk ; buy at net >= 0.2
  confidence = (net + 1) / 2   (0.2 net == 0.6 confidence, which is
                                exactly the execution gate)
  TP/SL from the bull, shrunk up to half as bear_risk -> 1, clamped
  to schema bounds.

The formula is re-implemented rather than imported from
tools/trader.py: that module's decide() is typed against the
regular-session BullCase/BearCase classes, and this brief keeps
existing files untouched — so the ~15 lines are duplicated with the
constants asserted equal in spirit. If the formula ever changes,
change BOTH (the comment in each file points at the other).

Decisions are validated through the same SignalDecision schema the
regular pipeline uses — one decision shape everywhere means the
execution layer needs exactly one code path.
"""

import json
from datetime import datetime
from pathlib import Path

from agents.signal_agent import SignalDecision
from tools.market_calendar import ET, is_market_open_today

BULL_PATH = Path("data/premarket_bull_cases.json")
BEAR_PATH = Path("data/premarket_bear_cases.json")
OUTPUT_PATH = Path("data/premarket_decisions.json")

# Keep in sync with tools/trader.py (see module docstring)
BUY_THRESHOLD = 0.2
MAX_TEMPERING = 0.5


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def decide_premarket_trades(output_path: Path = OUTPUT_PATH) -> list[SignalDecision]:
    """
    Entry point. Reads both case files, combines per symbol, writes
    final decisions. Symbols present in only one file are skipped —
    one-sided evidence must not become a trade (same rule as the
    regular signal stage).
    """
    if not is_market_open_today():
        print("premarket_trader: market closed today - nothing to do")
        return []
    for path in (BULL_PATH, BEAR_PATH):
        if not path.exists():
            raise SystemExit(f"{path} not found - run the premarket "
                             f"bull/bear agents first")

    bulls = json.loads(BULL_PATH.read_text())["cases"]
    bears = json.loads(BEAR_PATH.read_text())["cases"]

    decisions: list[SignalDecision] = []
    for symbol in sorted(bulls.keys() | bears.keys()):
        bull, bear = bulls.get(symbol), bears.get(symbol)
        if bull is None or bear is None:
            missing = "bull" if bull is None else "bear"
            print(f"[pm-trader] {symbol}: skipped - no {missing} case")
            continue

        net = bull["bull_confidence"] - bear["bear_risk"]
        temper = 1 - MAX_TEMPERING * bear["bear_risk"]
        decisions.append(SignalDecision(
            symbol=symbol,
            signal="buy" if net >= BUY_THRESHOLD else "hold",
            confidence=round((net + 1) / 2, 3),
            take_profit_pct=round(_clamp(bull["take_profit_pct"] * temper, 0.5, 20.0), 2),
            stop_loss_pct=round(_clamp(bull["stop_loss_pct"] * temper, 0.5, 10.0), 2),
            reasoning=(
                f"net {net:+.2f} = bull {bull['bull_confidence']:.2f} - "
                f"bear {bear['bear_risk']:.2f} (buy at >= "
                f"{BUY_THRESHOLD:+.2f}). BULL: {bull['bull_case']} "
                f"BEAR: {bear['bear_case']}"
            ),
        ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "decisions": [d.model_dump() for d in decisions],
    }, indent=2))
    buys = sum(1 for d in decisions if d.signal == "buy")
    print(f"premarket_trader: {len(decisions)} decisions ({buys} buys) "
          f"-> {output_path}")
    return decisions


if __name__ == "__main__":
    for d in decide_premarket_trades():
        print(f"{d.symbol:6s} {d.signal:4s} conf {d.confidence:.2f}  "
              f"tp {d.take_profit_pct:.1f}% sl {d.stop_loss_pct:.1f}%")
