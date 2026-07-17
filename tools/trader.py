"""
tools/trader.py

The referee of the bull/bear debate. Deterministic, no LLM — same
principle as the execution agent: the judgment already happened (two
opinionated cases with calibrated scores); combining them is policy,
and policy should be arithmetic you can read, test, and tune.

The policy:

  net_score = bull_confidence - bear_risk        range [-1, +1]
  buy when net_score >= BUY_THRESHOLD (0.2) - a buy requires the bull
  case to clearly outrun the bear case, not merely edge it.

  confidence = (net_score + 1) / 2               range [0, 1]
  This mapping is deliberate, not cosmetic: execution_agent filters
  at confidence >= 0.6, and (0.2 + 1) / 2 = 0.6 exactly. The trader's
  buy threshold and the execution agent's confidence gate are the
  same line by construction — no stock can be "bought" here and then
  dropped there, and execution_agent needs no changes.

  Exit levels come from the bull (who sized how far it can run),
  tempered by the bear: both TP and SL shrink by up to half as
  bear_risk approaches 1. A risky-but-taken trade gets a nearer
  target (take the money before the failure mode arrives) and a
  tighter stop (less room = less loss when the bear was right).
  Tempered values are clamped back into SignalDecision's schema
  bounds so tempering can never produce an invalid decision.
"""

from agents.bear_agent import BearCase
from agents.bull_agent import BullCase
from agents.signal_agent import SignalDecision

BUY_THRESHOLD = 0.2       # net_score needed to buy; == confidence 0.6
MAX_TEMPERING = 0.5       # at bear_risk 1.0, exits shrink to half

# SignalDecision schema bounds (keep in sync with agents/signal_agent.py)
TP_MIN, SL_MIN = 0.5, 0.5


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def decide(bull: BullCase, bear: BearCase) -> SignalDecision:
    """Combine one stock's bull and bear cases into a SignalDecision."""
    assert bull.symbol == bear.symbol, "bull/bear case symbol mismatch"

    net_score = bull.bull_confidence - bear.bear_risk
    signal = "buy" if net_score >= BUY_THRESHOLD else "hold"
    confidence = round((net_score + 1) / 2, 3)

    temper = 1 - MAX_TEMPERING * bear.bear_risk
    take_profit = _clamp(bull.take_profit_pct * temper, TP_MIN, 20.0)
    stop_loss = _clamp(bull.stop_loss_pct * temper, SL_MIN, 10.0)

    reasoning = (
        f"net {net_score:+.2f} = bull {bull.bull_confidence:.2f} - "
        f"bear {bear.bear_risk:.2f} (buy at >= {BUY_THRESHOLD:+.2f}). "
        f"BULL: {bull.bull_case} BEAR: {bear.bear_case}"
    )

    return SignalDecision(
        symbol=bull.symbol,
        signal=signal,
        confidence=confidence,
        take_profit_pct=round(take_profit, 2),
        stop_loss_pct=round(stop_loss, 2),
        reasoning=reasoning,
    )
