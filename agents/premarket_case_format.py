"""
agents/premarket_case_format.py

Shared evidence block for the pre-market bull/bear debate — same
rationale as agents/case_format.py (both sides must argue from
identical bytes), kept as a separate module because the evidence
shape differs: pre-market metrics + a candle read, no catalyst
report.

Evidence = scanner row + candle read + news headlines. The news
block was added after the first live runs: with only metrics and
structure to argue from, every gapping stock produced the same
templated case ("unusual volume, extended, exhaustion risk...") —
headlines are the stock-specific facts that let one stock's case
actually differ from another's. When there are no headlines, the
block says so explicitly, and the prompts treat "no news behind
this gap" as a first-class finding rather than a gap to fill with
invented narrative.

Works on plain dicts (rows from the JSON files) rather than
dataclasses: pre-market components exchange data through files, and
re-hydrating classes just to format text would couple this module to
every producer's internals.
"""

import json

from tools.datapaths import list_path


def load_session_news(session_date: str | None, label: str) -> dict[str, list]:
    """
    News for THIS session only, shared by both debate agents (same
    file, same freshness rule -> same bytes). A news file left over
    from a previous session must not silently feed today's debate —
    stale headlines presented as current would be worse than none, so
    a date mismatch degrades to "no news" with a warning.
    """
    news_path = list_path("premarket_news.json")
    if not news_path.exists():
        print(f"[{label}] {news_path} not found - arguing without news "
              f"(run tools.premarket_news first for differentiated cases)")
        return {}
    payload = json.loads(news_path.read_text())
    if session_date and payload.get("session_date") != session_date:
        print(f"[{label}] news file is for {payload.get('session_date')}, "
              f"scan is for {session_date} - ignoring stale news")
        return {}
    return payload.get("news", {})


def format_premarket_evidence(
    scan_row: dict,
    candle: dict | None,
    news: list[dict] | None = None,
) -> str:
    candle_block = (
        json.dumps(candle, indent=2) if candle
        else "(no candle read available for this stock)"
    )
    news_block = (
        json.dumps(news, indent=2) if news
        else "(no recent headlines found for this stock)"
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
        f"pre-market candle):\n{candle_block}\n\n"
        f"Recent news headlines (last 7 days, newest first):\n"
        f"{news_block}"
    )
