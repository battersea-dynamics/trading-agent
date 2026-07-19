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

from tools.broker import get_account, get_positions
from tools.datapaths import list_path
from tools.catalysts import build_catalyst_report, prescan_earnings
from tools.scanner import ScanResult, scan
from tools.universe_builder import load_universe

PORTFOLIO_STATE_PATH = Path("data/portfolio_state.json")  # runtime state, overwritten on purpose
TOP_N = 15


def snapshot_portfolio(output_path: Path = PORTFOLIO_STATE_PATH) -> dict:
    """
    Stage 0 of check_shortlist: one consistent look at the account
    before any judgment or money is involved. Written to disk rather
    than just returned so each run leaves a record of what the system
    *believed* it held when it acted — when a trade looks wrong later,
    the first question is always "what did it know at the time?".

    Plain program, no LLM: this is retrieval, not judgment.
    """
    state = {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        **get_account(),
        "positions": get_positions(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state, indent=2))
    return state


def held_symbols() -> set[str]:
    """
    Just the symbols currently held — what daily_scan needs to avoid
    wasting shortlist slots on stocks we already own. Deliberately
    separate from snapshot_portfolio(): same underlying broker call,
    but this one writes nothing, so a morning scan can't overwrite the
    portfolio_state.json audit record of the last check run.
    """
    return {p["symbol"] for p in get_positions()}


def daily_scan(
    output_path: Path | None = None,
    top_n: int = TOP_N,
) -> list[ScanResult]:
    """Stage 1: cheap, deterministic, LLM-free. Safe to run on a timer."""
    if output_path is None:
        output_path = list_path("shortlist.json")
    held = held_symbols()               # no state file written here
    universe = load_universe()          # cached daily, rebuilt when stale
    flagged = prescan_earnings(universe, days_ahead=3)

    # Over-fetch by the number of holdings, then drop held names and
    # trim back to top_n: a slot spent on a stock we already own is a
    # wasted LLM call in stage 2 (we wouldn't add to the position),
    # and this way the next-ranked stock inherits the slot instead of
    # the shortlist just shrinking.
    ranked = scan(top_n=top_n + len(held), symbols=universe,
                  catalyst_flags=flagged)
    shortlist = [r for r in ranked if r.symbol not in held][:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "catalyst_flagged": len(flagged),
        "excluded_held": sorted(held & {r.symbol for r in ranked}),
        "shortlist": [r.to_dict() for r in shortlist],
    }, indent=2))

    print(f"scan: {len(universe)} symbols -> shortlist of {len(shortlist)} "
          f"({sum(1 for r in shortlist if r.catalyst)} catalyst-flagged) "
          f"-> {output_path}")
    return shortlist


def check_shortlist(
    input_path: Path | None = None,
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

    # Stage 0: fresh portfolio state before anything else runs, so
    # every later step in this run works from the same picture of
    # cash and holdings (and the file records it).
    state = snapshot_portfolio()
    print(f"portfolio: ${state['cash']:.2f} cash, "
          f"${state['buying_power']:.2f} buying power, "
          f"{len(state['positions'])} position(s) "
          f"-> {PORTFOLIO_STATE_PATH}")

    if input_path is None:
        input_path = list_path("shortlist.json")
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

    # Review record for the by-date archive: the regular pipeline's
    # debate is in-memory (no separate bull/bear case files exist -
    # each decision's reasoning embeds both cases verbatim), so this
    # decisions file IS the daily equivalent of the premarket case/
    # decision files. Timestamped because checks run many times a day.
    check_record = list_path(
        f"check_decisions_{datetime.now().strftime('%H%M')}.json")
    check_record.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "live": live,
        "decisions": [d.model_dump() for d in decisions],
        "execution_report": report,
    }, indent=2))

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
