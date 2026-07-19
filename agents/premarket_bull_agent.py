"""
agents/premarket_bull_agent.py

Pre-market twin of agents/bull_agent.py — same architecture (genuine
one-sided advocacy, honesty confined to the score, anchored scale),
different domain knowledge.

What changed in the prompt vs the regular-session bull, and why:

  The advocacy checklist is pre-market-specific. A regular-session
  bull argues momentum and room to run; a pre-market bull's case
  lives or dies on three things the brief names: is the volume
  signature consistent with a REAL catalyst (heavy, sustained prints
  — not one thin spike), is the pre-market bid holding its gains
  into the open (building, not fading), and does the candle agent's
  structure read support continuation rather than exhaustion.

  News is part of the evidence (added after the first live runs).
  Footprint-only argument produced templated cases — every gapper
  got the same volume/gap language because there was nothing
  stock-specific to cite. The prompt now REQUIRES engaging with the
  news block: cite the headline that explains the gap, or state
  plainly that none does — and a no-story stock must be scored low
  (<= 0.4) rather than dressed in a narrative. "Nothing here" is an
  explicitly correct output.

  Exit levels think in gap terms. Pre-market winners are exited
  against gap-and-go continuation, not 20-day swing ranges — the
  prompt says to size the take-profit to what the gap structure
  supports.
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



class PremarketBullCase(BaseModel):
    symbol: str
    bull_case: str
    bull_confidence: float = Field(
        ge=0.0, le=1.0,
        description="How strong the bull case actually is, not how "
                    "strongly it is worded",
    )
    take_profit_pct: float = Field(ge=0.5, le=20.0)
    stop_loss_pct: float = Field(ge=0.5, le=10.0)


def build_premarket_bull_agent() -> Agent:
    return Agent(
        role="Pre-Market Bull Analyst",
        goal=(
            "Construct the strongest genuine case FOR buying each "
            "gapping stock at today's open, and honestly rate that "
            "case's strength."
        ),
        backstory=(
            "You are the designated bull in a pre-market debate. Your "
            "specialty is telling real gaps from fake ones using only "
            "their footprint: a genuine catalyst leaves heavy, "
            "SUSTAINED pre-market volume and a bid that holds or builds "
            "into the open; a fake gap prints thin volume that fades. "
            "Argue the strongest case the evidence supports — momentum, "
            "volume signature, candle structure — without hedging; a "
            "bear analyst sees the same data. Hard limits: only "
            "evidence in front of you (you have NOT been shown any "
            "news, so never invent a specific catalyst — argue at most "
            "that the footprint implies one), and your confidence "
            "score is your honest professional rating, not part of the "
            "advocacy. A thin case argued well is still thin; score "
            "it low. One more standard: a bull case that could be "
            "pasted onto any gapping stock is worthless. Your case "
            "must name what is specific to THIS stock today — a "
            "headline, a dated event, a number — or admit there is "
            "nothing specific. 'Nothing here, low confidence' is a "
            "correct, professional output; a dramatic narrative built "
            "on generic gap mechanics is a failure."
        ),
        llm=gemini_llm(),
        tools=[],
        allow_delegation=False,
        verbose=True,
    )


def build_premarket_bull_task(
    agent: Agent, scan_row: dict, candle: dict | None,
    news: list[dict] | None = None,
) -> Task:
    return Task(
        description=(
            f"Make the strongest genuine bull case for buying "
            f"{scan_row['symbol']} at today's open (bracket exit, held "
            f"hours to a few days).\n\n"
            + format_premarket_evidence(scan_row, candle, news) +
            "\n\nGround the case in pre-market-specific strength: the "
            "volume signature (is it consistent with a real catalyst — "
            "heavy and sustained, not one spike on a dead tape), the "
            "gap's behavior (holding/building vs fading), and the "
            "candle read (does the structure support continuation).\n\n"
            "Requirement: engage with the news block. Either cite the "
            "specific headline (with its date) that explains this gap "
            "and argue from it, or state plainly that no headline "
            "explains the move. If nothing in the evidence is specific "
            "to this stock — no headline, no dated event, just generic "
            "gap-and-volume mechanics — say exactly that and score the "
            "case 0.4 or below. Do not dress a no-story stock in a "
            "story.\n\n"
            "Then rate the case:\n"
            "  0.9+  exceptional - volume signature, holding gap, and "
            "supportive structure all align\n"
            "  0.7   solid - clear evidence, one leg missing\n"
            "  0.5   coin flip - real activity but direction unclear "
            "from here\n"
            "  <0.3  forced - you had to stretch\n\n"
            "Suggest take_profit_pct sized to what this gap structure "
            "plausibly supports after the open, and stop_loss_pct wide "
            "enough for opening volatility on a stock moving this much."
        ),
        expected_output=(
            "JSON: symbol, bull_case (3-5 sentences of grounded "
            "advocacy), bull_confidence (0.0-1.0 anchored), "
            "take_profit_pct, stop_loss_pct."
        ),
        agent=agent,
        output_pydantic=PremarketBullCase,
    )


def analyze_premarket_bull(
    scan_row: dict, candle: dict | None, news: list[dict] | None = None,
) -> PremarketBullCase | None:
    agent = build_premarket_bull_agent()
    task = build_premarket_bull_task(agent, scan_row, candle, news)
    return run_task(agent, task, label="pm-bull", symbol=scan_row["symbol"])


def run_premarket_bulls(output_path: Path | None = None) -> dict[str, dict]:
    """
    Entry point: bull cases for every stock in the pre-market scan,
    written to their own file. The file (not a return value in some
    shared process) is what lets tools/premarket_trader.py stay
    LLM-free: by the time it runs, all the judgment is already on
    disk.
    """
    if not is_market_open_today():
        print("premarket_bulls: market closed today - nothing to do")
        return {}
    auto_verify = output_path is None
    if output_path is None:
        output_path = list_path("premarket_bull_cases.json")
    scan_path = list_path("premarket_scan.json")
    candles_path = list_path("premarket_candles.json")
    if not scan_path.exists():
        raise SystemExit(f"{scan_path} not found - run the premarket "
                         f"scanner first")

    scan = json.loads(scan_path.read_text())
    candles = (json.loads(candles_path.read_text()).get("reads", {})
               if candles_path.exists() else {})
    news = load_session_news(scan.get("session_date"), label="pm-bull")

    cases: dict[str, dict] = {}
    for row in scan["shortlist"]:
        row = {**row, "session_date": scan.get("session_date")}
        case = analyze_premarket_bull(
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
    print(f"premarket_bulls: {len(cases)}/{len(scan['shortlist'])} argued "
          f"-> {output_path}")

    # Numeric fact-check every case just written (deterministic, no
    # LLM) - adds numbers_verified/unverified_numbers to the file.
    # Partial safeguard: numbers only, see tools/case_verifier.py.
    if auto_verify:
        from tools.case_verifier import verify_premarket_case_file
        cases = verify_premarket_case_file("bull")
    return cases


if __name__ == "__main__":
    for symbol, case in run_premarket_bulls().items():
        print(f"{symbol:6s} conf {case['bull_confidence']:.2f}  "
              f"tp {case['take_profit_pct']:.1f}% sl {case['stop_loss_pct']:.1f}%")
