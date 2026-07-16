"""
pipeline.py

The two entry points of the system, kept deliberately separate:

  daily_scan()      universe -> catalyst pre-scan -> scanner
                    -> data/shortlist.json
  check_shortlist() data/shortlist.json -> deep catalyst/news check
                    -> signal agent -> execution agent (dry-run
                    unless told otherwise)

Why a JSON file between them instead of one function calling the
other? Because the seam is where scheduling will live. daily_scan is
cheap and LLM-free — it can run pre-market on a timer with no risk.
check_shortlist spends LLM calls and (in live mode) money — you may
want it minutes later, market-hours only, after a manual look at the
file, or triggered more than once a day against the same shortlist.
Two processes with a file handoff means adding a scheduler later is
two cron lines / GitHub Actions jobs pointing at commands that already
exist — no refactor, and the file doubles as an audit trail of what
each day's scan actually said.

Nothing in here contains logic of its own — it only sequences calls
into the tools/ and agents/ modules and owns the file format. That's
what an entry point should be: thin enough that you can read it top
to bottom and know the whole system.

Usage:
  python pipeline.py scan            # stage 1, writes the shortlist
  python pipeline.py check           # stage 2, dry-run (no orders)
  python pipeline.py check --live    # stage 2, submits paper orders
  python pipeline.py all [--live]    # both stages back to back
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from tools.catalysts import build_catalyst_report, prescan_earnings
from tools.scanner import ScanResult, scan
from tools.universe_builder import load_universe

SHORTLIST_PATH = Path("data/shortlist.json")
TOP_N = 15


def daily_scan(
    output_path: Path = SHORTLIST_PATH,
    top_n: int = TOP_N,
) -> list[ScanResult]:
    """Stage 1: cheap, deterministic, LLM-free. Safe to run on a timer."""
    universe = load_universe()          # cached daily, rebuilt when stale
    flagged = prescan_earnings(universe, days_ahead=3)
    shortlist = scan(top_n=top_n, symbols=universe, catalyst_flags=flagged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "catalyst_flagged": len(flagged),
        "shortlist": [r.to_dict() for r in shortlist],
    }, indent=2))

    print(f"scan: {len(universe)} symbols -> shortlist of {len(shortlist)} "
          f"({sum(1 for r in shortlist if r.catalyst)} catalyst-flagged) "
          f"-> {output_path}")
    return shortlist


def check_shortlist(
    input_path: Path = SHORTLIST_PATH,
    live: bool = False,
) -> list[dict]:
    """
    Stage 2: judgment and (optionally) money. Reads whatever stage 1
    last wrote — including a warning if it's stale, since a shortlist
    from three days ago describes a market that no longer exists.
    """
    # Imported here, not at module top: these pull in CrewAI (slow
    # import) and require GEMINI_API_KEY. Stage 1 shouldn't pay
    # either cost just because it shares an entry-point file.
    from agents.execution_agent import execute_signals
    from agents.signal_agent import analyze_shortlist

    if not input_path.exists():
        raise SystemExit(f"{input_path} not found - run `pipeline.py scan` first")

    payload = json.loads(input_path.read_text())
    generated = datetime.fromisoformat(payload["generated_at"])
    age_hours = (datetime.now() - generated).total_seconds() / 3600
    if age_hours > 24:
        print(f"WARNING: shortlist is {age_hours:.0f}h old - "
              f"consider re-running the scan", file=sys.stderr)

    shortlist = [ScanResult(**row) for row in payload["shortlist"]]
    symbols = [r.symbol for r in shortlist]

    catalyst_report = build_catalyst_report(symbols)
    decisions = analyze_shortlist(shortlist, catalyst_report)
    report = execute_signals(decisions, live=live)

    print(json.dumps(report, indent=2))
    if not live:
        print("\n(dry run - pass --live to submit paper orders)",
              file=sys.stderr)
    return report


if __name__ == "__main__":
    args = set(sys.argv[1:])
    live = "--live" in args
    command = next((a for a in sys.argv[1:] if not a.startswith("--")), None)

    if command == "scan":
        daily_scan()
    elif command == "check":
        check_shortlist(live=live)
    elif command == "all":
        daily_scan()
        check_shortlist(live=live)
    else:
        raise SystemExit(__doc__)
