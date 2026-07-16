"""
agents/execution_agent.py

The last stage: turn approved buy signals into Alpaca bracket orders.

Deliberately NOT a CrewAI agent — and that's the architecture lesson
of this file. An "agent" earns an LLM when the step requires judgment
over unstructured input. Execution is the opposite: the judgment was
already made upstream (signal + confidence + exit levels), and what
remains is arithmetic and API calls that must be *boringly
deterministic*. An LLM here could round a price creatively, size a
position generously, or hallucinate a symbol — and unlike a bad
opinion, a bad order costs money. The rule of thumb: LLMs decide,
code executes.

The asymmetry with the signal agent is the security model:
  signal agent    - judgment, no order access
  execution agent - order access, no judgment
Neither can do the other's job, so no single failure (bad prompt,
model outage, hallucination) can both invent and place a trade.

What "no judgment" still includes — mechanical policy, applied
uniformly:
  - only signal == "buy" with confidence >= MIN_CONFIDENCE
  - fixed dollar cap per position (MAX_POSITION_USD), whole shares
    only (Alpaca forbids fractional bracket orders)
  - skip anything that can't be sized or quoted, rather than improvise
  - dry-run by default; pass live=True (or --live on the CLI) to
    actually submit
"""

import json
import math

from dotenv import load_dotenv

from agents.signal_agent import SignalDecision
from tools.broker import get_account, get_quote, place_bracket_order

load_dotenv()

MIN_CONFIDENCE = 0.6
MAX_POSITION_USD = 1_000.0   # cap per position, not per day
BUYING_POWER_HEADROOM = 0.95  # never commit the last 5% of buying power


def execute_signals(
    decisions: list[SignalDecision],
    live: bool = False,
) -> list[dict]:
    """
    Filter, size, and (if live) submit one bracket order per approved
    buy. Returns a report of what was done or would be done — the
    dry-run output is the exact order that live mode would submit.
    """
    account = get_account()
    available = account["buying_power"] * BUYING_POWER_HEADROOM
    report = []

    for decision in decisions:
        entry = {"symbol": decision.symbol, "action": "skipped"}
        report.append(entry)

        if decision.signal != "buy":
            entry["reason"] = f"signal is '{decision.signal}'"
            continue
        if decision.confidence < MIN_CONFIDENCE:
            entry["reason"] = (
                f"confidence {decision.confidence:.2f} < {MIN_CONFIDENCE}"
            )
            continue

        # Reference price for converting the agent's percentages into
        # absolute bracket prices: the current ask, i.e. roughly what
        # a market buy would actually pay right now.
        quote = get_quote(decision.symbol)
        ask = quote["ask"]
        if not ask or ask <= 0:
            entry["reason"] = f"no usable ask price (got {ask!r})"
            continue

        qty = math.floor(min(MAX_POSITION_USD, available) / ask)
        if qty < 1:
            entry["reason"] = (
                f"can't afford 1 share at ${ask:.2f} within "
                f"${MAX_POSITION_USD:.0f} cap / remaining buying power"
            )
            continue

        take_profit = ask * (1 + decision.take_profit_pct / 100)
        stop_loss = ask * (1 - decision.stop_loss_pct / 100)
        available -= qty * ask

        order = {
            "symbol": decision.symbol,
            "qty": qty,
            "est_cost": round(qty * ask, 2),
            "entry_ref": ask,
            "take_profit": round(take_profit, 2),
            "stop_loss": round(stop_loss, 2),
            "confidence": decision.confidence,
        }

        if live:
            result = place_bracket_order(
                decision.symbol, qty, take_profit, stop_loss
            )
            entry.update(action="submitted", order=order, broker=result)
        else:
            entry.update(action="dry_run", order=order)

    return report


if __name__ == "__main__":
    import sys

    from tools.catalysts import build_catalyst_report
    from tools.scanner import scan
    from agents.signal_agent import analyze_shortlist

    live = "--live" in sys.argv
    top_n = 5

    shortlist = scan(top_n=top_n)
    catalysts = build_catalyst_report([s.symbol for s in shortlist])
    decisions = analyze_shortlist(shortlist, catalysts)

    report = execute_signals(decisions, live=live)
    print(json.dumps(report, indent=2))
    if not live:
        print("\n(dry run - re-run with --live to submit paper orders)",
              file=sys.stderr)
