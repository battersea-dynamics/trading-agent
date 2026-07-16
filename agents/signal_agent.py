"""
agents/signal_agent.py

The judgment stage of the daily pipeline. Stages 1-2 (scanner,
catalysts) are deterministic code; this is where the LLM earns its
keep — weighing "5x volume spike" against "earnings in two days" and
"three negative headlines" is reading comprehension, not arithmetic.

This agent still cannot place trades. It has no access to
tools.broker's order functions, on purpose: it emits opinions, and the
execution agent (a separate file, with the opposite asymmetry — orders
but no judgment) acts on them. That separation is what makes each half
independently testable and lets a risk layer slot between them later.

CrewAI concepts in play (recap + one new one):

  Agent   - persona (role/goal/backstory become the prompt) + an LLM.
  Task    - one unit of work. NEW here: we build one Task *per stock*.
  Crew    - the runner. We run one single-task Crew per stock in a
            plain Python loop, NOT one big multi-task sequential Crew.
            Two reasons. Architecturally: a sequential Crew feeds each
            task the previous tasks' output as context, and stocks
            should not influence each other's verdicts. Practically:
            when we tried the multi-task form, CrewAI intermittently
            returned the first task's output for later tasks
            (duplicate decisions for the wrong symbol). Full isolation
            fixes both. Multi-task Crews earn their keep when tasks
            genuinely chain — e.g. a future risk-review task that
            *should* see the signal task's output via `context=[...]`.

New in the output schema: take_profit_pct / stop_loss_pct. The agent
doesn't just say "buy" — it sizes the exit brackets, because how far a
trade can plausibly run (and how much room it needs before the thesis
is dead) is part of the same judgment as whether to enter at all. The
broker enforces these mechanically after entry; nobody watches the
position.
"""

import json
from typing import Literal

from crewai import LLM, Agent, Crew, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from tools.scanner import ScanResult

# CrewAI's LLM class routes "gemini/..." models through LiteLLM, which reads
# GEMINI_API_KEY from the environment on its own - load .env explicitly here
# rather than relying on another module's load_dotenv() running first as a
# side effect of imports.
load_dotenv()


class SignalDecision(BaseModel):
    symbol: str
    signal: Literal["buy", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    take_profit_pct: float = Field(
        ge=0.5, le=20.0,
        description="Exit target above entry, in percent (e.g. 3.0 = +3%)",
    )
    stop_loss_pct: float = Field(
        ge=0.5, le=10.0,
        description="Exit floor below entry, in percent (e.g. 1.5 = -1.5%)",
    )
    reasoning: str


def build_signal_agent() -> Agent:
    return Agent(
        role="Intraday Signal Analyst",
        goal=(
            "For each candidate stock, decide buy or hold, and for buys "
            "set realistic take-profit and stop-loss levels for a trade "
            "held hours to a few days, entered at today's price."
        ),
        backstory=(
            "You are a disciplined swing/intraday analyst. You know that "
            "unusual volume without a catalyst is often noise, that "
            "holding through an earnings report is a coin flip rather "
            "than a trade, and that an ex-dividend date mechanically "
            "drops the price. You would rather say 'hold' than force a "
            "trade — most days most stocks are a 'hold'. You size "
            "take-profits to what the current volatility plausibly "
            "supports, and stops wide enough to survive normal noise "
            "but tight enough to cap real losses. You do not place "
            "trades — you only advise."
        ),
        # gemini-flash-latest is Google's rolling alias for the current
        # flash-tier model - gemini-2.5-flash itself returns 404 for newly
        # created API keys even though it's still listed in the model
        # catalog. Swap to a dated snapshot instead if you want the model
        # pinned rather than auto-updating.
        #
        # is_litellm=True: crewai has its own "native" Gemini client (a
        # separate google-genai dependency) and prefers it by default for
        # gemini/* models. Force the LiteLLM path instead, which is what
        # reads GEMINI_API_KEY the way we've set up the environment.
        llm=LLM(model="gemini/gemini-flash-latest", is_litellm=True),
        tools=[],
        allow_delegation=False,
        verbose=True,
    )


def build_stock_task(agent: Agent, scan: ScanResult, catalysts: dict) -> Task:
    return Task(
        description=(
            f"Evaluate {scan.symbol} ({scan.sector}) as an intraday/short "
            f"swing long candidate.\n\n"
            f"Scanner metrics (vs its own 20-day history):\n"
            f"  Close: ${scan.close:.2f}\n"
            f"  Relative volume: {scan.rel_volume:.2f}x normal\n"
            f"  Day change: {scan.pct_change:+.2f}%\n"
            f"  Distance from 20-day MA: {scan.ma_distance:+.2f}%\n\n"
            f"Catalyst report (earnings within 14 days, ex-dividend "
            f"dates within 14 days, headlines from the last 7 days):\n"
            f"{json.dumps(catalysts, indent=2)}\n\n"
            "Decide: buy or hold. Long-only — if the setup looks like a "
            "short, that's a hold. If earnings land within the next 2 "
            "trading days, lean strongly toward hold unless the setup is "
            "exceptional, and say so. Always fill take_profit_pct and "
            "stop_loss_pct even for holds (your hypothetical levels), "
            "keep take_profit_pct larger than stop_loss_pct, and note "
            "any ex-dividend date that falls inside the expected holding "
            "window."
        ),
        expected_output=(
            "A JSON object: symbol, signal (buy/hold), confidence "
            "(0.0-1.0), take_profit_pct, stop_loss_pct, reasoning "
            "(2-4 sentences citing the specific evidence)."
        ),
        agent=agent,
        output_pydantic=SignalDecision,
    )


def analyze_shortlist(
    shortlist: list[ScanResult],
    catalyst_report: dict[str, dict],
) -> list[SignalDecision]:
    decisions = []
    for scan in shortlist:
        agent = build_signal_agent()
        task = build_stock_task(agent, scan, catalyst_report.get(scan.symbol, {}))
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        decision = result.tasks_output[0].pydantic
        # Trust nothing that crosses a process boundary: the LLM fills
        # the symbol field itself, so pin it to the stock we actually
        # asked about before anything downstream trades on it.
        if decision.symbol != scan.symbol:
            decision.symbol = scan.symbol
        decisions.append(decision)
    return decisions


if __name__ == "__main__":
    import sys

    from tools.catalysts import build_catalyst_report
    from tools.scanner import scan

    # Default to a small shortlist when run by hand: each stock is one
    # Gemini call plus three catalyst API calls, so a full 15 takes a
    # few minutes on free-tier rate limits.
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    shortlist = scan(top_n=top_n)
    report = build_catalyst_report([s.symbol for s in shortlist])
    decisions = analyze_shortlist(shortlist, report)

    print(json.dumps([d.model_dump() for d in decisions], indent=2))
