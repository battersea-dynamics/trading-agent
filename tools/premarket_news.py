"""
tools/premarket_news.py

News pull for the pre-market shortlist — the missing ingredient that
was making the bull/bear debate templated. With only scanner metrics
and candle structure to argue from, every gapping stock looks alike
("unusual volume, extended, exhaustion risk...") and the agents could
only remix checklist vocabulary. Headlines are the stock-specific
facts that let one case actually differ from another.

A separate component with its own file, rather than a fetch inside
each agent runner, for two reasons:
  - one Finnhub pass instead of two (12 symbols x 2 agents would
    double the calls for identical data), and
  - the debate's symmetry rule: bull and bear must argue from
    identical bytes. Two fetches minutes apart can return different
    headlines, and then a score difference between the sides stops
    meaning judgment.

Reuses get_recent_news() from tools/catalysts.py unchanged — same
Finnhub source and shape as the regular-session pipeline, just
pointed at the pre-market shortlist.
"""

import json
from datetime import datetime
from pathlib import Path

from tools.catalysts import get_recent_news
from tools.market_calendar import ET, is_market_open_today

SCAN_PATH = Path("data/premarket_scan.json")
OUTPUT_PATH = Path("data/premarket_news.json")


def run_premarket_news(output_path: Path = OUTPUT_PATH) -> dict[str, list]:
    """
    Entry point. Headlines for every stock in the pre-market scan's
    shortlist -> data/premarket_news.json. One Finnhub call per
    symbol (12-15 calls, well inside the 60/min free tier).
    """
    if not is_market_open_today():
        print("premarket_news: market closed today - nothing to do")
        return {}
    if not SCAN_PATH.exists():
        raise SystemExit(f"{SCAN_PATH} not found - run the premarket "
                         f"scanner first")

    scan = json.loads(SCAN_PATH.read_text())
    news: dict[str, list] = {}
    for row in scan["shortlist"]:
        symbol = row["symbol"]
        try:
            news[symbol] = get_recent_news(symbol)
        except Exception as exc:
            # A failed news pull must not kill the chain: the debate
            # prompts handle "no news available" as a first-class case.
            print(f"[pm-news] {symbol}: fetch failed ({exc}) - "
                  f"recorded as no news")
            news[symbol] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "session_date": scan.get("session_date"),
        "news": news,
    }, indent=2))
    with_news = sum(1 for v in news.values() if v)
    print(f"premarket_news: {with_news}/{len(news)} symbols have "
          f"headlines -> {output_path}")
    return news


if __name__ == "__main__":
    for symbol, items in run_premarket_news().items():
        first = items[0]["headline"][:60] if items else "(none)"
        print(f"{symbol:6s} {len(items)} headline(s)  {first}")
