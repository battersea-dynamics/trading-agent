"""
agents/bull_agent.py

One half of the adversarial signal stage. Given the same evidence as
the bear, this agent constructs the strongest GENUINE case for buying
— and separately rates how strong that case actually is.

Prompt design notes (why it's worded the way it is):

  One-sided on purpose. Instruction-tuned models drift toward
  balanced, hedged answers — which is exactly what killed the single
  signal agent's usefulness: everything came back cautious. The
  debate structure only produces signal if each side commits fully,
  so the prompt explicitly forbids hedging ("do not argue the other
  side; the bear agent exists for that").

  The number is the escape valve. The PROSE must be maximally
  bullish; the CONFIDENCE must be honest. "Argue the best case, then
  rate how strong it really is" decouples rhetoric from calibration —
  a weak case argued brilliantly still gets a low number. Without
  this split, forcing bullish prose would also force inflated scores
  and the downstream subtraction would be meaningless.

  Anchored scale. The trader subtracts bear_risk from
  bull_confidence, which only means something if both agents grade on
  the same scale. Hence explicit anchors (0.9+ exceptional, 0.5 coin
  flip, <0.3 forced) mirrored word-for-word in the bear prompt.

  Bull owns the exit levels. "How far can this plausibly run" is
  bull-side expertise — the same judgment as the case itself. The
  trader tempers these levels by bear risk afterwards.
"""

from crewai import Agent, Task
from pydantic import BaseModel, Field

from agents.llm_runner import gemini_llm, run_task
from tools.scanner import ScanResult

from .case_format import format_evidence


class BullCase(BaseModel):
    symbol: str
    bull_case: str
    bull_confidence: float = Field(
        ge=0.0, le=1.0,
        description="How strong the bull case actually is, not how "
                    "strongly it is worded",
    )
    take_profit_pct: float = Field(
        ge=0.5, le=20.0,
        description="Suggested exit target above entry, percent",
    )
    stop_loss_pct: float = Field(
        ge=0.5, le=10.0,
        description="Suggested exit floor below entry, percent",
    )


def build_bull_agent() -> Agent:
    return Agent(
        role="Bull Case Analyst",
        goal=(
            "Construct the strongest genuine case FOR buying each "
            "candidate stock today, and honestly rate that case's "
            "strength."
        ),
        backstory=(
            "You are the designated bull in a two-analyst debate. Your "
            "job is advocacy: find every piece of evidence that supports "
            "buying now — momentum, volume confirmation, catalysts, room "
            "to run — and present it as persuasively as the facts allow. "
            "Do not hedge and do not argue the other side; a bear "
            "analyst sees the same data and will do that. Advocacy has "
            "one hard limit: you may only use evidence actually in front "
            "of you, never invented facts. And your confidence score is "
            "not part of the advocacy — it is your honest professional "
            "rating of the case you just made. A thin case argued well "
            "is still a thin case; score it low."
        ),
        llm=gemini_llm(),
        tools=[],
        allow_delegation=False,
        verbose=True,
    )


def build_bull_task(agent: Agent, scan: ScanResult, catalysts: dict) -> Task:
    return Task(
        description=(
            f"Make the strongest genuine bull case for buying "
            f"{scan.symbol} today, entered at the current price, held "
            f"hours to a few days.\n\n"
            + format_evidence(scan, catalysts) +
            "\n\nThen rate the case on this scale:\n"
            "  0.9+  exceptional - volume, catalyst, and room to run "
            "all align\n"
            "  0.7   solid - clear evidence, one leg missing\n"
            "  0.5   coin flip - the move is real but could go either "
            "way from here\n"
            "  <0.3  forced - you had to stretch to write this\n\n"
            "Also suggest take_profit_pct and stop_loss_pct for the "
            "trade: the take-profit sized to what this volatility "
            "plausibly supports, the stop wide enough to survive normal "
            "noise on a stock moving this much."
        ),
        expected_output=(
            "JSON: symbol, bull_case (3-5 sentences of grounded "
            "advocacy citing the specific evidence), bull_confidence "
            "(0.0-1.0 on the anchored scale), take_profit_pct, "
            "stop_loss_pct."
        ),
        agent=agent,
        output_pydantic=BullCase,
    )


def analyze_bull(scan: ScanResult, catalysts: dict) -> BullCase | None:
    agent = build_bull_agent()
    task = build_bull_task(agent, scan, catalysts)
    return run_task(agent, task, label="bull", symbol=scan.symbol)
