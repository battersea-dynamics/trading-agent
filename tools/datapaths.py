"""
tools/datapaths.py

Single source of truth for where dated pipeline artifacts live.

==================== TEMPORARY ARRANGEMENT ====================
The by-date layout (data/lists/<date>/, data/reports/) exists for
an ACTIVE REVIEW PERIOD: daily lists and reports are being read
day-by-day for a few weeks while the system is calibrated, so
nothing may overwrite them. This is NOT a permanent design
decision - once the review period ends, revisit: likely revert to
plain overwritten files (the daily report already summarizes each
day) or add a retention/cleanup policy. If you're reading this
months later and the review period is long over, that revisit
never happened - do it.
===============================================================

Layout:
  data/lists/<date>/    every per-run list/case/decision file,
                        partitioned by ET session date
  data/reports/         daily_report_<date>.json (date in filename)
  data/weekly/          reserved for the future weekly summary
  data/ (loose)         genuinely runtime-only state that SHOULD be
                        overwritten: universe cache, orchestrator
                        state, portfolio snapshot

Paths are functions, not module constants, on purpose: a constant
Path captures the date at import time, so any process alive across
midnight (or a module imported at 23:59 ET) would write today's
files into yesterday's folder. Resolving the date at call time
makes that impossible.
"""

from datetime import date, datetime
from pathlib import Path

from tools.market_calendar import ET

LISTS_ROOT = Path("data/lists")
REPORTS_ROOT = Path("data/reports")
WEEKLY_ROOT = Path("data/weekly")


def _resolve_day(day: date | None) -> date:
    return day if day is not None else datetime.now(ET).date()


def list_path(filename: str, day: date | None = None) -> Path:
    """data/lists/<date>/<filename>, directory created on demand."""
    folder = LISTS_ROOT / _resolve_day(day).isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def report_path(day: date | None = None) -> Path:
    """data/reports/daily_report_<date>.json."""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    return REPORTS_ROOT / f"daily_report_{_resolve_day(day).isoformat()}.json"
