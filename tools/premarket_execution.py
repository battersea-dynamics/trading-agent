"""
tools/premarket_execution.py

Last pre-market component: turn the trader's decisions file into
bracket orders. Deliberately almost no code of its own — it
rehydrates the decisions into SignalDecision objects and hands them
to the existing, tested execute_signals(), which brings along the
whole safety apparatus for free: the 0.6 confidence gate, the
per-position dollar cap, whole-share sizing, the dead-quote guard,
dry-run-by-default, and place_bracket_order (GTC brackets) at the
end. New execution code would mean re-testing all of that; reused
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
from tools.market_calendar import ET, is_market_open_today

DECISIONS_PATH = Path("data/premarket_decisions.json")


def execute_premarket_decisions(
    input_path: Path = DECISIONS_PATH,
    live: bool = False,
) -> list[dict]:
    """
    Entry point. Assumes regular market hours have begun (the
    orchestrator's promise); dry-run unless live=True.
    """
    if not is_market_open_today():
        print("premarket_execution: market closed today - nothing to do")
        return []
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
    report = execute_signals(decisions, live=live)

    print(json.dumps(report, indent=2))
    if not live:
        print("\n(dry run - pass --live / live=True to submit paper "
              "orders)", file=sys.stderr)
    return report


if __name__ == "__main__":
    execute_premarket_decisions(live="--live" in sys.argv)
