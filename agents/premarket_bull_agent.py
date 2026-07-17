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

  The catalyst argument is footprint-only. This agent is not given
  headlines, so the prompt explicitly frames the catalyst question
  as "does the evidence imply one" — an LLM asked to discuss news it
  doesn't have will happily invent it, and an invented catalyst is
  worse than none.

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
from agents.premarket_case_format import format_premarket_evidence
from tools.market_calendar import ET, is_market_open_today

load_dotenv()

SCAN_PATH = Path("data/premarket_scan.json")
CANDLES_PATH = Path("data/premarket_candles.json")
OUTPUT_PATH = Path("data/premarket_bull_cases.json")


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
            "it low."
        ),
        llm=gemini_llm(),
        tools=[],
        allow_delegation=False,
        verbose=True,
    )


def build_premarket_bull_task(
    agent: Agent, scan_row: dict, candle: dict | None
) -> Task:
    return Task(
        description=(
            f"Make the strongest genuine bull case for buying "
            f"{scan_row['symbol']} at today's open (bracket exit, held "
            f"hours to a few days).\n\n"
            + format_premarket_evidence(scan_row, candle) +
            "\n\nGround the case in pre-market-specific strength: the "
            "volume signature (is it consistent with a real catalyst — "
            "heavy and sustained, not one spike on a dead tape), the "
            "gap's behavior (holding/building vs fading), and the "
            "candle read (does the structure support continuation).\n\n"
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
    scan_row: dict, candle: dict | None
) -> PremarketBullCase | None:
    agent = build_premarket_bull_agent()
    task = build_premarket_bull_task(agent, scan_row, candle)
    return run_task(agent, task, label="pm-bull", symbol=scan_row["symbol"])


def run_premarket_bulls(output_path: Path = OUTPUT_PATH) -> dict[str, dict]:
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
    if not SCAN_PATH.exists():
        raise SystemExit(f"{SCAN_PATH} not found - run the premarket "
                         f"scanner first")

    scan = json.loads(SCAN_PATH.read_text())
    candles = (json.loads(CANDLES_PATH.read_text()).get("reads", {})
               if CANDLES_PATH.exists() else {})

    cases: dict[str, dict] = {}
    for row in scan["shortlist"]:
        row = {**row, "session_date": scan.get("session_date")}
        case = analyze_premarket_bull(row, candles.get(row["symbol"]))
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
    return cases


if __name__ == "__main__":
    for symbol, case in run_premarket_bulls().items():
        print(f"{symbol:6s} conf {case['bull_confidence']:.2f}  "
              f"tp {case['take_profit_pct']:.1f}% sl {case['stop_loss_pct']:.1f}%")
