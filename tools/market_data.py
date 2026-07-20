"""
tools/market_data.py

One rule, in one place: never ask Alpaca for market data newer than
the free plan is entitled to see.

THE BUG THIS EXISTS TO PREVENT
------------------------------
Alpaca's free plan does not include *real-time* SIP (consolidated)
data — only SIP delayed by 15 minutes, plus real-time IEX. A bars
request that doesn't pin `end` reaches implicitly into "now", so as
soon as the window touched the last 15 minutes the API rejected the
whole call:

    APIError: {"message":"subscription does not permit querying
                          recent SIP data"}

It failed *by time of day*, which is why it looked intermittent: a
daily-bars request pre-market is fine (the newest daily bar is
yesterday's), but the pre-market scanner's minute-bar window
(04:00 -> 09:30 ET, requested at 08:45) always included the current
minute, so the pre-market chain failed every single scheduled run.

WHY CAP THE WINDOW RATHER THAN SWITCH TO feed=IEX
-------------------------------------------------
IEX is one exchange, not the consolidated tape: measured against SIP
on the same days, IEX carries only ~1.6-4.5% of a symbol's volume
(AAPL 3.4%, KLRS 1.6%), and for a thin pre-market window it was 0.1%
(1,294 shares vs 1,144,260). Two things would break:

  * Absolute thresholds are calibrated on consolidated volume —
    universe_builder's MIN_AVG_VOLUME (500k ADV) and
    premarket_scanner's MIN_PM_VOLUME (5k). On IEX, genuinely liquid
    names fail them (ISRG, 4.1M real ADV, reads as 184k), and the
    universe collapses to mega-caps.
  * Relative volume compares *today* against a *multi-day baseline*.
    Feeds must match on both sides or the ratio is meaningless, so
    "IEX for today, SIP for history" is not an option either.

(`feed=DELAYED_SIP` is rejected outright by this plan — "invalid
feed" — so it isn't an escape hatch.)

Capping `end` keeps the consolidated feed and every existing
threshold valid, at the cost of ~16 minutes of freshness. That is
cheap here: the universe filter is a 15-day average, the scanner's
lookback is 20 days, and the pre-market window is ~4.5 hours long.

Real-time *quotes* are unaffected and deliberately not routed through
this module: tools.broker.get_quote works on the free plan, and order
sizing must use the current ask, never a 16-minute-old one.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# 15 is the entitlement boundary; the extra minute absorbs clock skew
# between this machine and Alpaca (exactly-15 intermittently 403s).
SIP_DELAY_MINUTES = 16

_ET = ZoneInfo("America/New_York")


def sip_safe_end(requested_end: datetime | None = None) -> datetime:
    """
    The latest timestamp this plan may request, optionally bounded by
    a caller's own `end`. Pass the result as `end=` on EVERY bars
    request — including ones whose window is already historical, so
    the rule is auditable as "no bars request omits sip_safe_end".

    `requested_end=None` means "as fresh as allowed".

    Naive datetimes are read as MARKET time (ET), not the machine's
    timezone: this box runs on UK time, where interpreting a bare
    09:30 locally would silently rewind it to 04:30 ET and truncate
    most of the pre-market window.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SIP_DELAY_MINUTES)
    if requested_end is None:
        return cutoff
    if requested_end.tzinfo is None:
        requested_end = requested_end.replace(tzinfo=_ET)
    return min(requested_end, cutoff)
