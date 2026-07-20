"""
tools/premarket_execution.py

Last pre-market component: turn the trader's decisions file into
bracket orders. Deliberately almost no code of its own — it
rehydrates the decisions into SignalDecision objects and hands them
to the existing, tested execute_signals(), which brings along the
whole safety apparatus for free: the 0.6 confidence gate, the
position cap (MAX_POSITION_PCT of live account value), the exit
ceilings (take-profit clamped at MAX_TAKE_PROFIT_PCT; stop-loss
above MAX_STOP_LOSS_PCT skips the trade), whole-share sizing, the
dead-quote guard, dry-run-by-default, and place_bracket_order (GTC
brackets) at the end. Those constants live in
agents/execution_agent.py ON PURPOSE - both pipelines share the one
sizing path, so a calibration change lands in both by editing one
file. New execution code would mean re-testing all of that; reused
code means it stays tested.

Timing is explicitly NOT this module's job. Bracket orders can't be
submitted for extended-hours execution (Alpaca limitation we
accepted when choosing brackets), so this must run once the regular
session is open — but WHEN to call it belongs to the future
orchestrator. A component that both decided timing and acted on it
would be two responsibilities in one name.

The calendar gate still applies (no orders on a holiday even if
invoked by mistake), plus one freshness guard: decisions from a
previous session must never execute today — a gap thesis is dead by
the next day, and a stale file silently becoming orders is exactly
the kind of failure a file-seam design has to defend against.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from agents.execution_agent import execute_signals
from agents.signal_agent import SignalDecision
from tools.broker import get_quote
from tools.datapaths import list_path
from tools.market_calendar import ET, is_market_open_today


# Price deviation guard: a pre-market decision was argued against a
# specific price (the scan's last_pm_price). If the live ask has
# moved more than this far from it - EITHER direction - the stock
# the debate judged no longer exists and the decision doesn't
# transfer. Symmetric on purpose: a drop isn't "more room to rise",
# it's evidence something changed since the thesis was formed.
MAX_PRICE_DEVIATION_PCT = 2.0


def _reference_prices() -> dict[str, float]:
    """last_pm_price per symbol from the scan the debate argued over."""
    scan_path = list_path("premarket_scan.json")
    if not scan_path.exists():
        return {}
    scan = json.loads(scan_path.read_text())
    return {r["symbol"]: r["last_pm_price"] for r in scan["shortlist"]}


def _apply_deviation_guard(
    decisions: list[SignalDecision],
) -> tuple[list[SignalDecision], list[dict]]:
    """
    Split decisions into (pass-through, guard-report-entries).
    Only buys are price-checked - holds never execute anyway. A buy
    with no findable reference price is dropped too: an unverifiable
    price basis fails safe, same philosophy as the dead-quote guard.
    A dead ask (<= 0) passes through so execute_signals can report
    it with its own, more precise reason.
    """
    references = _reference_prices()
    survivors, guard_entries = [], []
    for decision in decisions:
        if decision.signal != "buy":
            survivors.append(decision)
            continue
        reference = references.get(decision.symbol)
        if not reference or reference <= 0:
            guard_entries.append({
                "symbol": decision.symbol, "action": "skipped",
                "reason": "price guard: no reference price in "
                          "the session scan for this symbol",
            })
            continue
        ask = get_quote(decision.symbol)["ask"]
        if not ask or ask <= 0:
            survivors.append(decision)  # dead-quote guard's job
            continue
        deviation_pct = (ask - reference) / reference * 100
        if abs(deviation_pct) > MAX_PRICE_DEVIATION_PCT:
            guard_entries.append({
                "symbol": decision.symbol, "action": "skipped",
                "reason": (
                    f"price guard: ask ${ask:.2f} is "
                    f"{deviation_pct:+.2f}% vs decision basis "
                    f"${reference:.2f} (limit "
                    f"±{MAX_PRICE_DEVIATION_PCT}%)"
                ),
            })
            continue
        survivors.append(decision)
    return survivors, guard_entries


def execute_premarket_decisions(
    input_path: Path | None = None,
    submit: bool = False,
) -> list[dict]:
    """
    Entry point. Assumes regular market hours have begun (the
    orchestrator's promise); dry-run unless submit=True (paper orders).
    """
    if not is_market_open_today():
        print("premarket_execution: market closed today - nothing to do")
        return []
    if input_path is None:
        input_path = list_path("premarket_decisions.json")
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found - run the premarket "
                         f"trader first")

    payload = json.loads(input_path.read_text())
    generated = datetime.fromisoformat(payload["generated_at"])
    if generated.date() != datetime.now(ET).date():
        raise SystemExit(
            f"premarket_execution: decisions file is from "
            f"{generated.date()}, today is {datetime.now(ET).date()} - "
            f"refusing to execute a stale gap thesis. Re-run the "
            f"pre-market pipeline."
        )

    decisions = [SignalDecision(**d) for d in payload["decisions"]]
    decisions, guard_entries = _apply_deviation_guard(decisions)
    report = guard_entries + execute_signals(decisions, submit=submit)

    print(json.dumps(report, indent=2))
    if not submit:
        print("\n(dry run - pass --submit / submit=True to submit paper "
              "orders)", file=sys.stderr)
    return report


if __name__ == "__main__":
    execute_premarket_decisions(submit="--submit" in sys.argv)
