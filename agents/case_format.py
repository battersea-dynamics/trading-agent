"""
agents/case_format.py

The one shared evidence block both debate agents receive. Kept in a
single function so the bull and bear literally see the same bytes —
if each agent formatted its own view of the data, a score difference
between them could come from presentation (one saw the headlines
first, one saw them last) rather than judgment. Symmetric inputs are
what make the bull-minus-bear subtraction meaningful.
"""

import json

from tools.scanner import ScanResult


def format_evidence(scan: ScanResult, catalysts: dict) -> str:
    return (
        f"Evidence for {scan.symbol} ({scan.sector}):\n\n"
        f"Scanner metrics (vs its own 20-day history):\n"
        f"  Close: ${scan.close:.2f}\n"
        f"  Relative volume: {scan.rel_volume:.2f}x normal\n"
        f"  Day change: {scan.pct_change:+.2f}%\n"
        f"  Distance from 20-day MA: {scan.ma_distance:+.2f}%\n\n"
        f"Catalyst report (earnings within 14 days, ex-dividend dates "
        f"within 14 days, headlines from the last 7 days):\n"
        f"{json.dumps(catalysts, indent=2)}"
    )
