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
  - stop-loss ceiling: > MAX_STOP_LOSS_PCT skips the trade (never
    clamp a stop tighter); take-profit ceiling: > MAX_TAKE_PROFIT_PCT
    clamps down and proceeds
  - position cap: MAX_POSITION_PCT of live account value, whole
    shares only (Alpaca forbids fractional bracket orders); if one
    share busts the cap, skip
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
BUYING_POWER_HEADROOM = 0.95  # never commit the last 5% of buying power

# Position sizing: 20% of live account value per position, not a
# fixed dollar cap - scales automatically as the account grows or
# shrinks, no manual updates. If even one whole share busts the cap
# (brackets can't be fractional), the trade is skipped outright.
MAX_POSITION_PCT = 0.20

# Asymmetric exit ceilings, applied to the trader's numbers at the
# last moment before submission. Asymmetric on purpose:
#   TP > 12%  -> CLAMP to 12% and proceed. Capping upside never
#                makes a trade less safe.
#   SL > 5%   -> SKIP the trade entirely, never clamp tighter. A
#                wide stop is the bear agent's honest read of real
#                volatility; forcing it tighter just converts normal
#                noise into stop-outs, which defeats the stop.
MAX_TAKE_PROFIT_PCT = 12.0
MAX_STOP_LOSS_PCT = 5.0


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
    position_cap = account["portfolio_value"] * MAX_POSITION_PCT
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

        # Stop-loss ceiling: skip, never tighten (see constants).
        if decision.stop_loss_pct > MAX_STOP_LOSS_PCT:
            entry["reason"] = (
                f"stop-loss ceiling exceeded, skipped: trader set "
                f"{decision.stop_loss_pct:.1f}% > {MAX_STOP_LOSS_PCT:.0f}% "
                f"max (wide stop = honest volatility read; not clamping)"
            )
            continue

        # Take-profit ceiling: clamp and proceed (capping upside
        # never makes a trade less safe).
        take_profit_pct = decision.take_profit_pct
        tp_clamped = take_profit_pct > MAX_TAKE_PROFIT_PCT
        if tp_clamped:
            take_profit_pct = MAX_TAKE_PROFIT_PCT

        # Reference price for converting the agent's percentages into
        # absolute bracket prices: the current ask, i.e. roughly what
        # a market buy would actually pay right now.
        quote = get_quote(decision.symbol)
        ask = quote["ask"]
        if not ask or ask <= 0:
            entry["reason"] = f"no usable ask price (got {ask!r})"
            continue

        if ask > position_cap:
            entry["reason"] = (
                f"position size cap exceeded, skipped: 1 share at "
                f"${ask:.2f} > {MAX_POSITION_PCT:.0%} of account value "
                f"(${position_cap:.2f})"
            )
            continue
        qty = math.floor(min(position_cap, available) / ask)
        if qty < 1:
            entry["reason"] = (
                f"can't afford 1 share at ${ask:.2f} with remaining "
                f"buying power (${available:.2f})"
            )
            continue

        take_profit = ask * (1 + take_profit_pct / 100)
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
        if tp_clamped:
            order["take_profit_clamped"] = (
                f"trader wanted {decision.take_profit_pct:.1f}%, "
                f"capped at {MAX_TAKE_PROFIT_PCT:.0f}%"
            )

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
