"""
agents/premarket_case_format.py

Shared evidence block for the pre-market bull/bear debate — same
rationale as agents/case_format.py (both sides must argue from
identical bytes), kept as a separate module because the evidence
shape differs: pre-market metrics + a candle read, no catalyst
report.

That absence is deliberate and worth understanding: the pre-market
debate's inputs are the scanner row and the candle agent's read.
Neither side is told WHAT the news is — only what its footprint
looks like (gap size, volume vs pre-market baseline, candle
structure). The prompts therefore ask the agents to reason about
whether the footprint implies a real catalyst, not to name one —
an agent asked to discuss news it was never given will invent some.
(Extension point: pass tools.catalysts.build_catalyst_report output
in here and add it to the block if the debate should see headlines.)

Works on plain dicts (rows from the JSON files) rather than
dataclasses: pre-market components exchange data through files, and
re-hydrating classes just to format text would couple this module to
every producer's internals.
"""

import json


def format_premarket_evidence(scan_row: dict, candle: dict | None) -> str:
    candle_block = (
        json.dumps(candle, indent=2) if candle
        else "(no candle read available for this stock)"
    )
    return (
        f"Pre-market evidence for {scan_row['symbol']} "
        f"(session {scan_row.get('session_date', 'today')}):\n\n"
        f"Scanner metrics (baseline = this stock's own pre-market "
        f"sessions, ~2 weeks):\n"
        f"  Yesterday's close: ${scan_row['prev_close']:.2f}\n"
        f"  Last pre-market price: ${scan_row['last_pm_price']:.2f}\n"
        f"  Pre-market gap: {scan_row['pm_gap_pct']:+.2f}%\n"
        f"  Pre-market volume: {scan_row['pm_volume']:,} shares "
        f"(vs {scan_row['avg_pm_volume']:,} baseline avg = "
        f"{scan_row['rel_pm_volume']:.2f}x)\n\n"
        f"Candle agent's structure read (yesterday's daily + today's "
        f"pre-market candle):\n{candle_block}"
    )
