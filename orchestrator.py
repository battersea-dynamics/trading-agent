"""
orchestrator.py

The timing layer — and ONLY the timing layer. It knows what time it
is in New York and which entry point comes next; it contains no
trading logic, no thresholds, no data handling. Every stage below is
one import + one call into a component that already exists, already
gates itself on the market calendar, and already fails safe on bad
data. If this file were deleted, nothing about WHAT the system does
would change — only WHEN.

Timezone rule: every schedule computation happens in US Eastern
(the market's clock), via tools/market_calendar.py. Nick runs this
from the UK; neither he nor this file's logic ever reasons about the
UK offset — local time appears only in log lines, for readability.

Early-close days are handled by asking Alpaca for the session's real
open/close (session_times) instead of assuming 9:30-16:00: on the
day after Thanksgiving the close is 13:00, so "last check at
close-45min" correctly becomes 12:15, and the check loop shrinks.

The default day:

  open-45min   pre-market chain (scan -> news -> candles -> bulls
               -> bears -> trader) - produces decisions, spends no
               money
  open         poll Alpaca's clock until the session is actually
               open (bounded - a few minutes, then give up on the
               execution stage; a session that never opens means
               something is wrong and no orders is the right number
               of orders)
  open+0       pre-market execution (bracket orders; --live only)
  open+30min   daily_scan()
  every 30min  check_shortlist(), last run no later than close-45min

Manual override: any stage can be run immediately with
  python -m orchestrator --force premarket
  python -m orchestrator --force premarket_exec [--live]
  python -m orchestrator --force daily_scan
  python -m orchestrator --force check [--live]
so testing never waits for the clock. (Component-level calendar
gates still apply on non-trading days.) A future GitHub Actions
workflow calls these same entry points on its own cron — this file
is structured so the schedule constants below translate directly
into cron lines.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta

from tools.market_calendar import ET, todays_session

# ----- schedule configuration (minutes are relative to session) -----
PREMARKET_LEAD_MIN = 45        # pre-market chain starts open-45min
DAILY_SCAN_DELAY_MIN = 30      # daily_scan at open+30min
CHECK_INTERVAL_MIN = 30        # check_shortlist cadence
LAST_CHECK_BEFORE_CLOSE_MIN = 45   # no check after close-45min
OPEN_POLL_INTERVAL_SEC = 20    # how often to ask "is it open yet?"
OPEN_POLL_MAX_MIN = 5          # give up on execution this long after
                               # the scheduled open if still closed
SLEEP_CHUNK_SEC = 60           # wake at least this often while waiting


# ----- stages: one import + one call each, lazily imported so the -----
# ----- orchestrator starts instantly and LLM deps load only if used -----

def stage_premarket():
    from tools.premarket_scanner import run_premarket_scan
    from tools.premarket_news import run_premarket_news
    from agents.candle_agent import run_candle_agent
    from agents.premarket_bull_agent import run_premarket_bulls
    from agents.premarket_bear_agent import run_premarket_bears
    from tools.premarket_trader import decide_premarket_trades

    run_premarket_scan()
    run_premarket_news()
    run_candle_agent()
    run_premarket_bulls()
    run_premarket_bears()
    decide_premarket_trades()


def stage_premarket_exec(live: bool = False):
    from tools.premarket_execution import execute_premarket_decisions
    execute_premarket_decisions(live=live)


def stage_daily_scan():
    from pipeline import daily_scan
    daily_scan()


def stage_check(live: bool = False):
    from pipeline import check_shortlist
    check_shortlist(live=live)


STAGES = {
    "premarket": lambda live: stage_premarket(),
    "premarket_exec": stage_premarket_exec,
    "daily_scan": lambda live: stage_daily_scan(),
    "check": stage_check,
}


# ----- timing helpers -----

def _now() -> datetime:
    return datetime.now(ET)


def _log(message: str):
    now = _now()
    print(f"[orchestrator] {now:%H:%M:%S} ET "
          f"({now.astimezone():%H:%M} local)  {message}", flush=True)


def _sleep_until(target: datetime, label: str):
    """Chunked sleep so Ctrl+C stays responsive and logs show life."""
    while (remaining := (target - _now()).total_seconds()) > 0:
        if remaining > SLEEP_CHUNK_SEC * 5:
            _log(f"waiting for {label} at {target:%H:%M} ET "
                 f"({remaining / 60:.0f} min)")
        time.sleep(min(remaining, SLEEP_CHUNK_SEC))


def _poll_until_open(scheduled_open: datetime) -> bool:
    """
    True once Alpaca's own clock says the session is open. Bounded:
    if it still isn't open OPEN_POLL_MAX_MIN after the scheduled
    time, something unusual is happening (delayed open, halt) and we
    refuse to run the execution stage rather than trade into it.
    """
    from tools.broker import trading_client

    deadline = scheduled_open + timedelta(minutes=OPEN_POLL_MAX_MIN)
    while _now() < deadline:
        if trading_client.get_clock().is_open:
            return True
        _log(f"market not open yet, polling every "
             f"{OPEN_POLL_INTERVAL_SEC}s")
        time.sleep(OPEN_POLL_INTERVAL_SEC)
    return False


# ----- the scheduled day -----

def run_day(live: bool = False):
    session = todays_session()
    if session is None:
        _log("no trading session today - exiting")
        return
    session_open, session_close = session

    premarket_at = session_open - timedelta(minutes=PREMARKET_LEAD_MIN)
    daily_scan_at = session_open + timedelta(minutes=DAILY_SCAN_DELAY_MIN)
    last_check_at = session_close - timedelta(
        minutes=LAST_CHECK_BEFORE_CLOSE_MIN)

    _log(f"session {session_open:%H:%M}-{session_close:%H:%M} ET | "
         f"premarket {premarket_at:%H:%M}, exec ~{session_open:%H:%M}, "
         f"daily_scan {daily_scan_at:%H:%M}, checks every "
         f"{CHECK_INTERVAL_MIN}min until {last_check_at:%H:%M}"
         + (" | LIVE ORDERS" if live else " | dry-run"))

    if _now() < premarket_at:
        _sleep_until(premarket_at, "pre-market chain")
    _log("stage: premarket chain")
    stage_premarket()

    _sleep_until(session_open, "market open")
    if _poll_until_open(session_open):
        _log("stage: premarket execution")
        stage_premarket_exec(live=live)
    else:
        _log(f"market still closed {OPEN_POLL_MAX_MIN}min after "
             f"scheduled open - SKIPPING premarket execution")

    _sleep_until(daily_scan_at, "daily scan")
    _log("stage: daily_scan")
    stage_daily_scan()

    check_at = daily_scan_at
    while (check_at := check_at + timedelta(
            minutes=CHECK_INTERVAL_MIN)) <= last_check_at:
        _sleep_until(check_at, "shortlist check")
        _log("stage: check_shortlist")
        stage_check(live=live)

    _log("trading day schedule complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Timing layer for the trading pipeline")
    parser.add_argument("--force", choices=sorted(STAGES),
                        help="run one stage immediately, skip all waiting")
    parser.add_argument("--live", action="store_true",
                        help="execution stages submit real paper orders")
    args = parser.parse_args()

    if args.force:
        _log(f"forced stage: {args.force}"
             + (" | LIVE ORDERS" if args.live else " | dry-run"))
        STAGES[args.force](args.live)
    else:
        try:
            run_day(live=args.live)
        except KeyboardInterrupt:
            _log("interrupted - exiting")
            sys.exit(130)
