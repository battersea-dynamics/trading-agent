"""
tools/market_calendar.py

Shared gate for every pre-market component: is there a market today?

Asking Alpaca's calendar API instead of checking weekday() means
holidays, half-days, and any future schedule quirks are Alpaca's
problem, not ours — the exchange's own calendar is the only source
that's right by definition.

Timezone note that matters here: "today" means today IN NEW YORK.
This machine runs on CET, where 1:00 AM Saturday is still Friday
afternoon in New York — using the local date would wrongly close the
gate (or worse, open it). Every date in this module is computed in
America/New_York.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

# The broker module already owns the one shared Alpaca connection -
# reuse it rather than constructing a second client.
from tools.broker import trading_client

ET = ZoneInfo("America/New_York")


def is_trading_day(day: date) -> bool:
    """True if the exchange calendar has a session on `day`."""
    calendar = trading_client.get_calendar(
        GetCalendarRequest(start=day, end=day)
    )
    return any(session.date == day for session in calendar)


def is_market_open_today() -> bool:
    """
    True if today (in New York) is a trading day. The standard first
    call of every pre-market component: if this is False, the
    component prints why and exits cleanly.
    """
    return is_trading_day(datetime.now(ET).date())


def session_times(day: date) -> tuple[datetime, datetime] | None:
    """
    (open, close) as ET-aware datetimes for `day`'s session, or None
    if there is no session. This is what a scheduler must use instead
    of hardcoding 9:30-16:00: on early-close days (day after
    Thanksgiving, Christmas Eve) the exchange closes at 13:00, and
    "last run no later than close minus 45 minutes" means something
    different on those days. Alpaca's calendar returns the real
    times; we only attach the timezone (its datetimes are naive ET).
    """
    calendar = trading_client.get_calendar(
        GetCalendarRequest(start=day, end=day)
    )
    for session in calendar:
        if session.date == day:
            return (session.open.replace(tzinfo=ET),
                    session.close.replace(tzinfo=ET))
    return None


def todays_session() -> tuple[datetime, datetime] | None:
    """session_times for today in New York."""
    return session_times(datetime.now(ET).date())


if __name__ == "__main__":
    today_et = datetime.now(ET).date()
    print(f"today in New York: {today_et}")
    print(f"market open today: {is_market_open_today()}")
