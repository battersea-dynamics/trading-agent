"""
agents/signal_agent.py

The first agent in the system. It has one job: look at a quote and form
an opinion. It cannot place trades — it has no access to
tools.broker.place_market_order, on purpose. Keeping "decide" and
"execute" as separate agents is what lets a future risk agent sit
between them later without anyone having to rewrite this file.

Three CrewAI concepts show up here:

  Agent  - a persona with a role/goal/backstory and an LLM. The
           backstory isn't flavor text — it's part of the prompt CrewAI
           builds for the model, same as role/goal.
  Task   - one unit of work for an agent: a description (the "job"),
           an expected_output (what shape the answer should take), and
           optionally output_pydantic to force the answer into a
           validated schema instead of free-form text.
  Crew   - the runner. It takes agents + tasks and executes them
           (here: one agent, one task, so "process" barely matters —
           it becomes interesting once there's more than one task).
"""

import json

from crewai import LLM, Agent, Crew, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from tools.broker import get_quote

# CrewAI's LLM class routes "gemini/..." models through LiteLLM, which reads
# GEMINI_API_KEY from the environment on its own - load .env explicitly here
# rather than relying on tools.broker's load_dotenv() running first as a
# side effect of the import above.
load_dotenv()


class SignalDecision(BaseModel):
    symbol: str
    signal: str = Field(description="One of: buy, sell, hold")
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


def build_signal_agent() -> Agent:
    return Agent(
        role="Market Signal Analyst",
        goal=(
            "Form a clear, well-reasoned buy/sell/hold opinion on a stock "
            "given only its current bid/ask quote."
        ),
        backstory=(
            "You are a cautious equity analyst. You never have more "
            "information than what's given to you, and you say so when "
            "the signal is weak. You do not place trades — you only "
            "advise."
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


def build_signal_task(agent: Agent, quote: dict) -> Task:
    return Task(
        description=(
            f"Here is the current quote for {quote['symbol']}:\n"
            f"  Bid: {quote['bid']}\n"
            f"  Ask: {quote['ask']}\n\n"
            "Based only on this quote (the bid/ask spread and price "
            "level), form a buy/sell/hold opinion. A raw quote is thin "
            "evidence — say so in your reasoning, and keep confidence "
            "low unless something in the spread genuinely stands out."
        ),
        expected_output=(
            "A JSON object with fields: symbol, signal (buy/sell/hold), "
            "reasoning (1-3 sentences), confidence (0.0-1.0)."
        ),
        agent=agent,
        output_pydantic=SignalDecision,
    )


def analyze_symbol(symbol: str) -> SignalDecision:
    quote = get_quote(symbol)

    agent = build_signal_agent()
    task = build_signal_task(agent, quote)

    # A Crew with one agent and one task still goes through the full
    # kickoff lifecycle - useful to see now, since the risk/execution
    # agents will plug into this same Crew as more tasks later.
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    crew.kickoff()

    return task.output.pydantic


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    decision = analyze_symbol(symbol)
    print(json.dumps(decision.model_dump(), indent=2))
