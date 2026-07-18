"""
tools/daily_report.py

Deterministic daily report builder — no LLM, no judgment, just an
append-only event log per session date. The orchestrator's tick mode
records what actually happened (stages run, orders placed, guards
tripped, errors raised) into data/daily_report_<date>.json, and the
GitHub Actions workflow commits that file once at the end of the day
— so what the system did is readable on GitHub without digging
through Action logs.

Append-only by design: each tick loads the existing file, appends
its events, rewrites. Nothing ever edits history — the report is an
audit artifact, and audit artifacts don't get revised.
"""

import json
from datetime import date, datetime
from pathlib import Path

from tools.market_calendar import ET

REPORT_DIR = Path("data")


def report_path(day: date) -> Path:
    return REPORT_DIR / f"daily_report_{day.isoformat()}.json"


def append_event(day: date, event_type: str, detail: dict | None = None):
    path = report_path(day)
    doc = (json.loads(path.read_text()) if path.exists()
           else {"date": day.isoformat(), "events": []})
    event = {
        "at_et": datetime.now(ET).isoformat(timespec="seconds"),
        "type": event_type,
    }
    if detail:
        event["detail"] = detail
    doc["events"].append(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))


def summarize_execution(execution_report: list[dict]) -> dict:
    """
    Collapse an execute_signals-style report into what the daily
    report cares about: what was submitted (or would have been), and
    every guard that fired with its reason. The guard reasons are the
    interesting part — they're how you see the safety net working
    (or over-firing) day over day without reading logs.
    """
    orders = [e for e in execution_report
              if e.get("action") in ("submitted", "dry_run")]
    guard_skips = [
        {"symbol": e["symbol"], "reason": e["reason"]}
        for e in execution_report
        if e.get("action") == "skipped" and e.get("reason")
    ]
    return {
        "orders": orders,
        "guard_skips": guard_skips,
        "counts": {"orders": len(orders), "skipped": len(guard_skips)},
    }
