"""
agents/llm_runner.py

Shared plumbing for every LLM-backed agent: the Gemini config, the
call pacing, and the retry/skip policy. Extracted from signal_agent.py
when the signal stage split into bull + bear — with multiple modules
making LLM calls, pacing has to be enforced in ONE place or the limit
gets blown by the sum of callers each individually behaving.

The numbers, and where they come from:

  Gemini free tier allows ~5 requests/min on the flash model.
  13s spacing keeps a single caller under that. The bull/bear split
  doubles the calls per stock, but a per-minute limit doesn't care
  about totals — only rate — so the same spacing still holds; a
  15-stock run just takes ~2x the wall time (15 stocks x 2 calls
  x 13s ~ 6.5 min minimum).

  The throttle is a module-global "time since last call anywhere",
  not a sleep inside any one loop: bull then bear on the same stock
  are two calls from two modules, and both must count.

  Retries handle the 429s that slip through anyway (CrewAI can make
  more than one request per task, e.g. schema-validation retries).
  Anything non-retryable surfaces immediately — a bug should crash,
  not be silently retried.

  After all retries fail: return None. The caller treats a missing
  opinion as "no trade" — degradation must always land on the safe
  side of the ledger.
"""

import sys
import time

from crewai import LLM, Agent, Crew, Task
from dotenv import load_dotenv

load_dotenv()

CALL_SPACING_SECONDS = 13
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 45

# Pinned GA model, deliberately NOT a rolling alias and NOT -preview.
# History: we used gemini-flash-latest, and overnight it started
# resolving to gemini-3.5-flash - whose free tier allows 20 requests
# per DAY, less than one debate run. Rolling aliases let Google change
# our quota out from under us; a pinned model can only break loudly.
# The lite tier carries the highest free-tier daily quotas, which is
# what a 30+-calls-per-run pipeline actually needs. (gemini-2.5-flash
# itself now 404s: "no longer available to new users".)
GEMINI_MODEL = "gemini/gemini-3.1-flash-lite"

_last_call_at = 0.0
_daily_quota_exhausted = False


def gemini_llm() -> LLM:
    # is_litellm=True forces the LiteLLM path, which is what reads
    # GEMINI_API_KEY the way our environment is set up (CrewAI's
    # native Gemini client is a separate dependency we don't use).
    return LLM(model=GEMINI_MODEL, is_litellm=True)


def _throttle():
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    if elapsed < CALL_SPACING_SECONDS:
        time.sleep(CALL_SPACING_SECONDS - elapsed)
    _last_call_at = time.monotonic()


def run_task(agent: Agent, task: Task, label: str, symbol: str):
    """
    Run one single-task Crew with pacing and rate-limit retries.
    Returns the task's pydantic output, or None if rate limits
    persisted through all retries (caller must treat None as
    "no opinion -> no trade").

    Daily-quota 429s are handled differently from per-minute ones:
    a per-minute limit passes with a pause; a per-day limit means
    every further call today is a guaranteed failure. Retrying those
    would turn an already-dead run into an hour of sleeps, so the
    first daily-quota error latches _daily_quota_exhausted and every
    subsequent call fast-fails to None.
    """
    global _daily_quota_exhausted
    if _daily_quota_exhausted:
        print(f"[{label}] {symbol}: skipped - daily Gemini quota exhausted",
              file=sys.stderr)
        return None

    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    for attempt in range(RATE_LIMIT_RETRIES):
        _throttle()
        try:
            result = crew.kickoff()
            output = result.tasks_output[0].pydantic
            # Trust nothing that crosses a process boundary: the LLM
            # fills the symbol field itself, so pin it to the stock we
            # actually asked about.
            if getattr(output, "symbol", symbol) != symbol:
                output.symbol = symbol
            return output
        except Exception as exc:
            # APIConnectionError ("Server disconnected") observed in
            # live pre-market runs: a transient network drop, not a
            # bug - retry it like a 503 rather than crashing the run.
            retryable = ("RateLimitError", "ServiceUnavailable",
                         "APIConnectionError")
            marker = type(exc).__name__ + str(exc)
            if "PerDay" in marker or "per day" in marker:
                print(f"[{label}] {symbol}: DAILY quota exhausted - "
                      f"skipping all remaining LLM calls this run",
                      file=sys.stderr)
                _daily_quota_exhausted = True
                return None
            if not any(m in marker for m in (*retryable, "429", "503")):
                raise
            print(f"[{label}] {symbol}: rate-limited, waiting "
                  f"{RATE_LIMIT_BACKOFF_SECONDS}s "
                  f"(attempt {attempt + 1}/{RATE_LIMIT_RETRIES})",
                  file=sys.stderr)
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)

    print(f"[{label}] {symbol}: skipped - rate limit persisted",
          file=sys.stderr)
    return None
