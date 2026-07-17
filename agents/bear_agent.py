"""
agents/bear_agent.py

The other half of the adversarial signal stage: the strongest genuine
case AGAINST buying right now.

Prompt design notes, where they differ from the bull's:

  Same skeleton, mirrored. One-sided by mandate, honesty confined to
  the number, identical scale anchors (the trader subtracts the two
  scores — see bull_agent.py for why the anchors must match).

  The checklist is the bear-specific addition. A lazy bear says
  "stocks can go down" — generically true, zero information, and the
  subtraction would just shift every net score down by a constant. A
  useful bear names the SPECIFIC failure mode. So the prompt hands it
  the archetypes this system actually meets (chasing an extended
  move, earnings coin-flip, mechanical ex-div drop, volume fading
  under the price, thin spread) and requires the case to tie risks to
  this stock's evidence, not to the market in general.

  bear_risk is "how dangerous is buying TODAY", not "is this a bad
  company". The horizon matters: a great company 26% above its moving
  average can be a terrible entry, and that distinction is the whole
  reason this agent exists.
"""

from crewai import Agent, Task
from pydantic import BaseModel, Field

from agents.llm_runner import gemini_llm, run_task
from tools.scanner import ScanResult

from .case_format import format_evidence


class BearCase(BaseModel):
    symbol: str
    bear_case: str
    bear_risk: float = Field(
        ge=0.0, le=1.0,
        description="How dangerous buying today actually is, not how "
                    "scary the case is worded",
    )


def build_bear_agent() -> Agent:
    return Agent(
        role="Bear Case Analyst",
        goal=(
            "Construct the strongest genuine case AGAINST buying each "
            "candidate stock today, and honestly rate how dangerous an "
            "entry right now would be."
        ),
        backstory=(
            "You are the designated bear in a two-analyst debate. Your "
            "job is to find how this trade dies: the specific, concrete "
            "failure mode in the evidence — not generic caution. 'Stocks "
            "can go down' is a firing offense; 'this is 26% above its "
            "20-day average on news that's already public' is your craft. "
            "Do not hedge toward optimism and do not argue the other "
            "side; a bull analyst sees the same data and will do that. "
            "You may only use evidence actually in front of you. Your "
            "risk score is not part of the advocacy — it is your honest "
            "rating of the danger. If the setup is genuinely clean, a "
            "strained bear case scores low, and saying so is doing your "
            "job, not failing it."
        ),
        llm=gemini_llm(),
        tools=[],
        allow_delegation=False,
        verbose=True,
    )


def build_bear_task(agent: Agent, scan: ScanResult, catalysts: dict) -> Task:
    return Task(
        description=(
            f"Make the strongest genuine bear case against buying "
            f"{scan.symbol} today at the current price, for a hold of "
            f"hours to a few days.\n\n"
            + format_evidence(scan, catalysts) +
            "\n\nFailure modes to check the evidence against (cite the "
            "ones that apply — do not pad with ones that don't):\n"
            "  - chasing: the move already happened; entry is the exit "
            "liquidity for earlier buyers\n"
            "  - earnings within the holding window: a coin flip, not "
            "a trade\n"
            "  - ex-dividend inside the window: mechanical price drop "
            "that can trip a stop\n"
            "  - volume not confirming: price stretched while "
            "participation fades\n"
            "  - news already priced: headlines old enough that the "
            "reaction is behind, not ahead\n"
            "  - no catalyst at all: unusual tape with nothing driving "
            "it tends to mean-revert\n\n"
            "Then rate the danger of buying today on this scale:\n"
            "  0.9+  exceptional danger - a named, imminent, likely "
            "failure mode\n"
            "  0.7   serious - clear specific risks outweigh the setup\n"
            "  0.5   coin flip - real risks, real setup, could go "
            "either way\n"
            "  <0.3  strained - you had to stretch to find the risks"
        ),
        expected_output=(
            "JSON: symbol, bear_case (3-5 sentences naming the specific "
            "failure modes in this stock's evidence), bear_risk "
            "(0.0-1.0 on the anchored scale)."
        ),
        agent=agent,
        output_pydantic=BearCase,
    )


def analyze_bear(scan: ScanResult, catalysts: dict) -> BearCase | None:
    agent = build_bear_agent()
    task = build_bear_task(agent, scan, catalysts)
    return run_task(agent, task, label="bear", symbol=scan.symbol)
