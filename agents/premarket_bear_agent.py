"""
agents/premarket_bear_agent.py

Pre-market twin of agents/bear_agent.py. Same skeleton as every
debate agent (committed one-sided prose, honest anchored score);
what's rewritten is the failure-mode checklist, because pre-market
trades die differently than regular-session trades:

  gap fade            the majority outcome for gaps without a real
                      catalyst: the open is the high of the day and
                      the gap closes back toward yesterday
  thin-tape pricing   pre-market prints are a handful of shares on
                      wide spreads; the "price" may simply be wrong
                      as a forecast of where real size trades at open
  opening auction     the 9:30 cross brings the day's real liquidity;
                      pre-market direction frequently reverses in the
                      first minutes when actual size shows up
  priced-in catalyst  if the news broke hours ago and the gap has
                      already stopped building, the market is done
                      reacting - buying the open buys the top

The checklist is the bear's edge for the same reason as before: a
bear that names the specific way THIS trade dies produces signal; a
bear that says "gaps are risky" shifts every score down by a
constant and produces nothing.

News is part of the evidence (added after the first live runs, same
motive as the bull: footprint-only argument produced templated
cases). The bear's news requirement has a deliberate asymmetry: for
a bull, "no headline explains the gap" caps the score low; for a
bear it's affirmative evidence — an unexplained gap is the classic
gap-fade setup — so the prompt says to cite the absence as support,
not write around it. Headlines that do exist must be addressed by
name: stale? priced in? weaker than the move implies?
"""

import json
from datetime import datetime
from pathlib import Path

from crewai import Agent, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agents.llm_runner import gemini_llm, run_task
from tools.datapaths import list_path
from agents.premarket_case_format import (
    format_premarket_evidence,
    load_session_news,
)
from tools.market_calendar import ET, is_market_open_today

load_dotenv()



class PremarketBearCase(BaseModel):
    symbol: str
    bear_case: str
    bear_risk: float = Field(
        ge=0.0, le=1.0,
        description="How dangerous buying at the open actually is, "
                    "not how scary the case is worded",
    )


def build_premarket_bear_agent() -> Agent:
    return Agent(
        role="Pre-Market Bear Analyst",
        goal=(
            "Construct the strongest genuine case AGAINST buying each "
            "gapping stock at today's open, and honestly rate the "
            "danger of that entry."
        ),
        backstory=(
            "You are the designated bear in a pre-market debate, and "
            "your specialty is how gap trades die. You know the base "
            "rates: most gaps without a durable catalyst fade; "
            "pre-market prices are set by a thin tape that real "
            "opening liquidity routinely overrules; and a gap that has "
            "stopped building before the open usually means the "
            "reaction is finished. Name the specific failure mode in "
            "THIS stock's evidence — generic caution is a firing "
            "offense. Do not hedge toward optimism; a bull analyst "
            "sees the same data. Only evidence in front of you (no "
            "invented news), and your risk score is your honest rating "
            "— if the setup is genuinely clean, scoring it low is "
            "doing your job. One more standard: a bear case that could "
            "be pasted onto any gapper is worthless. Tie every risk "
            "you name to something specific in THIS stock's evidence — "
            "a headline, its date, a number — and when the honest read "
            "is 'ordinary gap, ordinary risks', say that plainly at "
            "middling risk instead of inventing drama."
        ),
        llm=gemini_llm(),
        tools=[],
        allow_delegation=False,
        verbose=True,
    )


def build_premarket_bear_task(
    agent: Agent, scan_row: dict, candle: dict | None,
    news: list[dict] | None = None,
) -> Task:
    return Task(
        description=(
            f"Make the strongest genuine bear case against buying "
            f"{scan_row['symbol']} at today's open.\n\n"
            + format_premarket_evidence(scan_row, candle, news) +
            "\n\nRequirement: engage with the news block. If headlines "
            "exist, your case must address the strongest one by name — "
            "is it stale (check its date against the gap), already "
            "priced in, or weaker than the move implies? If NO headline "
            "explains the gap, that absence is itself evidence: cite it "
            "explicitly as support for the gap-fade failure mode rather "
            "than writing around it.\n\n"
            "Pre-market failure modes to check the evidence "
            "against (cite the ones that apply — do not pad):\n"
            "  - gap fade: no sign of a durable catalyst in the volume "
            "signature; the open becomes the high of the day\n"
            "  - thin-tape pricing: pre-market volume too small for "
            "the printed price to mean anything about where real size "
            "trades at 9:30\n"
            "  - opening-auction reversal: the structure suggests "
            "pre-market direction won't survive first contact with "
            "real liquidity\n"
            "  - already priced in: the gap has stopped building — "
            "whatever drove it, the market is done reacting and the "
            "open is exit liquidity\n"
            "  - exhaustion structure: the candle read itself argues "
            "the move is finished\n\n"
            "Then rate the danger of buying at the open:\n"
            "  0.9+  exceptional danger - a named, imminent, likely "
            "failure mode\n"
            "  0.7   serious - specific risks clearly outweigh the "
            "setup\n"
            "  0.5   coin flip - real risks, real setup\n"
            "  <0.3  strained - you had to stretch to find the risks"
        ),
        expected_output=(
            "JSON: symbol, bear_case (3-5 sentences naming the "
            "specific failure modes in this stock's evidence), "
            "bear_risk (0.0-1.0 anchored)."
        ),
        agent=agent,
        output_pydantic=PremarketBearCase,
    )


def analyze_premarket_bear(
    scan_row: dict, candle: dict | None, news: list[dict] | None = None,
) -> PremarketBearCase | None:
    agent = build_premarket_bear_agent()
    task = build_premarket_bear_task(agent, scan_row, candle, news)
    return run_task(agent, task, label="pm-bear", symbol=scan_row["symbol"])


def run_premarket_bears(output_path: Path | None = None) -> dict[str, dict]:
    """Entry point: bear cases for the whole pre-market shortlist."""
    if not is_market_open_today():
        print("premarket_bears: market closed today - nothing to do")
        return {}
    auto_verify = output_path is None
    if output_path is None:
        output_path = list_path("premarket_bear_cases.json")
    scan_path = list_path("premarket_scan.json")
    candles_path = list_path("premarket_candles.json")
    if not scan_path.exists():
        raise SystemExit(f"{scan_path} not found - run the premarket "
                         f"scanner first")

    scan = json.loads(scan_path.read_text())
    candles = (json.loads(candles_path.read_text()).get("reads", {})
               if candles_path.exists() else {})
    news = load_session_news(scan.get("session_date"), label="pm-bear")

    cases: dict[str, dict] = {}
    for row in scan["shortlist"]:
        row = {**row, "session_date": scan.get("session_date")}
        case = analyze_premarket_bear(
            row, candles.get(row["symbol"]), news.get(row["symbol"])
        )
        if case is not None:
            cases[row["symbol"]] = case.model_dump()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "session_date": scan.get("session_date"),
        "cases": cases,
    }, indent=2))
    print(f"premarket_bears: {len(cases)}/{len(scan['shortlist'])} argued "
          f"-> {output_path}")

    # Numeric fact-check every case just written (deterministic, no
    # LLM) - adds numbers_verified/unverified_numbers to the file.
    # Partial safeguard: numbers only, see tools/case_verifier.py.
    if auto_verify:
        from tools.case_verifier import verify_premarket_case_file
        cases = verify_premarket_case_file("bear")
    return cases


if __name__ == "__main__":
    for symbol, case in run_premarket_bears().items():
        print(f"{symbol:6s} risk {case['bear_risk']:.2f}")
