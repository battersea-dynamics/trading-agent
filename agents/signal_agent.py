"""
agents/signal_agent.py

The judgment stage — restructured from one analyst into a debate:

  bull_agent   the strongest genuine case FOR buying   (LLM)
  bear_agent   the strongest genuine case AGAINST      (LLM)
  trader       deterministic referee: net score, buy/hold,
               bear-tempered exit levels               (no LLM)

Why a debate instead of one analyst? The single agent had to hold
both sides in one head, and instruction-tuned models resolve that
tension by hedging — nearly everything came back a cautious hold.
Splitting advocacy into two committed, opposite mandates forces the
evidence to actually be argued, and moves the final weighing into
deterministic code (tools/trader.py) where the threshold is a number
you can read and tune instead of a mood inside a prompt.

This module keeps its old public surface on purpose:
  SignalDecision       unchanged schema - execution_agent.py and its
                       confidence gate work untouched
  analyze_shortlist()  same signature - pipeline.py works untouched
Everything behind that surface changed; nothing in front of it did.

Skip policy: a stock only gets a decision if BOTH cases exist. If
either side was rate-limited away, deciding on one opinion would
defeat the whole design (a bull case with no bear check is exactly
the hedged-optimism failure this restructure removes), so the stock
is skipped — no decision, no trade.
"""

import json
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agents.bear_agent import analyze_bear
from agents.bull_agent import analyze_bull
from tools.scanner import ScanResult

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
    # Numeric fact-check result (tools/case_verifier.py). Defaulted so
    # every existing producer/consumer keeps working unchanged. NOTE:
    # True means "cited numbers trace to source data" - it does NOT
    # vouch for qualitative claims (see the verifier's LIMITATION).
    numbers_verified: bool = True
    unverified_numbers: list[str] = Field(default_factory=list)


def analyze_shortlist(
    shortlist: list[ScanResult],
    catalyst_report: dict[str, dict],
) -> list[SignalDecision]:
    # Imported here, not at module top: trader imports SignalDecision
    # from this module, so a top-level import would be circular. By
    # call time this module is fully initialized and the cycle is moot.
    from tools.trader import decide

    decisions = []
    for scan in shortlist:
        catalysts = catalyst_report.get(scan.symbol, {})

        # Call pacing lives in llm_runner's global throttle - these
        # two calls (and every pair after them) are automatically
        # spaced, so no sleep is needed at this level.
        bull = analyze_bull(scan, catalysts)
        bear = analyze_bear(scan, catalysts)

        if bull is None or bear is None:
            missing = "bull" if bull is None else "bear"
            print(f"[signal] {scan.symbol}: skipped - no {missing} case "
                  f"(one-sided evidence must not become a trade)")
            continue

        decision = decide(bull, bear)

        # Numeric fact-check (deterministic, no LLM): every unit-marked
        # number in both case texts must trace to the evidence the
        # agents were shown. Partial safeguard - numbers only, see
        # tools/case_verifier.py's LIMITATION note.
        from tools.case_verifier import verify_text
        sources = (scan.to_dict(), catalysts,
                   bull.model_dump(), bear.model_dump())
        bull_ok, bull_bad = verify_text(bull.bull_case, *sources)
        bear_ok, bear_bad = verify_text(bear.bear_case, *sources)
        decision.numbers_verified = bull_ok and bear_ok
        decision.unverified_numbers = (
            [f"bull: {m}" for m in bull_bad] +
            [f"bear: {m}" for m in bear_bad]
        )
        if not decision.numbers_verified:
            print(f"[verify] {scan.symbol}: "
                  f"{'; '.join(decision.unverified_numbers)}")

        decisions.append(decision)
    return decisions


if __name__ == "__main__":
    import sys

    from tools.catalysts import build_catalyst_report
    from tools.scanner import scan

    # Small default when run by hand: each stock is now TWO Gemini
    # calls at 13s spacing, so a full 15-stock run takes ~7 minutes.
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    shortlist = scan(top_n=top_n)
    report = build_catalyst_report([s.symbol for s in shortlist])
    decisions = analyze_shortlist(shortlist, report)

    print(json.dumps([d.model_dump() for d in decisions], indent=2))
