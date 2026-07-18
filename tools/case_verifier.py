"""
tools/case_verifier.py

Deterministic numeric fact-checker for bull/bear case text. No LLM.

=========================== LIMITATION ===========================
This checks NUMBERS ONLY. It answers one narrow question: "does
every unit-marked number cited in the case text appear in the
source data (within rounding tolerance)?" It CANNOT catch:
  - invented qualitative claims (a fabricated catalyst, a made-up
    analyst opinion, "management pushed back on rumors" that never
    happened) — the exact failure that motivated the news-evidence
    fix contained no numbers at all;
  - correct numbers attached to wrong conclusions;
  - direction errors (it compares magnitudes, not signs — "up 7%"
    vs a real -7% move passes).
It is a tripwire for numeric drift/hallucination, a real but
PARTIAL safeguard. Do not mistake `numbers_verified: true` for
"this case is true".
==================================================================

Design choices:

  Flat source pool. All numbers from the scan row, candle read,
  news headlines' own text, and the case's own output fields go
  into one pool; a cited number passes if it's near ANY of them.
  Typed matching (percents against percents only) would be
  stricter but false-flags legitimate derivations; a flat pool
  means a flag genuinely indicates an untraceable number. For a
  review tool, high-precision flags beat high-recall ones.

  Only unit-marked numbers are extracted: percents (7.07%),
  multiples (5.63x, "20 times"), dollars ($55.52), comma-grouped
  counts (39,948), and word-scaled values (8.3 million). Bare
  integers are skipped on purpose — "20-day average" and "the
  first 30 minutes" are structure, not claims.

  Tolerance: 6% relative or 0.12 absolute, whichever is looser —
  agents legitimately round ("nearly 10x" for 9.99, "80%" for
  82.4). Below that, a number isn't a rounding; it's wrong.
"""

import json
import re
from pathlib import Path

PM_SCAN_PATH = Path("data/premarket_scan.json")
PM_NEWS_PATH = Path("data/premarket_news.json")
PM_CANDLES_PATH = Path("data/premarket_candles.json")
PM_CASE_FILES = {
    "bull": Path("data/premarket_bull_cases.json"),
    "bear": Path("data/premarket_bear_cases.json"),
}

REL_TOLERANCE = 0.06
ABS_TOLERANCE = 0.12

_SCALES = {"million": 1e6, "billion": 1e9}

# Unit-marked claims only (see docstring). Each pattern captures the
# numeric part; scaled forms capture (number, scale-word).
_NUM = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"  # comma-aware
_CLAIM_PATTERNS = [
    re.compile(rf"[+-]?({_NUM})\s*%"),                  # 7.07% / 1,200%
    re.compile(rf"({_NUM})\s*(?:x\b|times\b)", re.I),   # 5.63x / 5,500x
    re.compile(rf"\$\s*({_NUM})"),                      # $55.52
    re.compile(r"\b(\d{1,3}(?:,\d{3})+)(?:\.\d+)?\b"),  # bare 39,948
]
_SCALED_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(million|billion)", re.I)
_NUMBER_IN_SOURCE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def extract_claims(text: str) -> list[float]:
    """Unit-marked numbers cited in a case text."""
    claims = []
    for number, scale in _SCALED_PATTERN.findall(text):
        claims.append(round(float(number) * _SCALES[scale.lower()], 6))
    for pattern in _CLAIM_PATTERNS:
        for match in pattern.findall(text):
            claims.append(float(match.replace(",", "")))
    return sorted(set(claims))


def collect_source_numbers(*sources) -> set[float]:
    """
    Every number in the given source objects (dicts/lists/strings
    walked recursively; numbers inside strings — e.g. a '$55 target'
    in a headline — count, since agents legitimately cite them).
    """
    pool: set[float] = set()

    def walk(node):
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            pool.add(abs(float(node)))
        elif isinstance(node, str):
            for match in _NUMBER_IN_SOURCE.findall(node):
                pool.add(abs(float(match.replace(",", ""))))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    for source in sources:
        walk(source)
    return pool


def _matches(claim: float, fact: float) -> bool:
    if abs(claim - fact) <= ABS_TOLERANCE:
        return True
    if fact != 0 and abs(claim - fact) / abs(fact) <= REL_TOLERANCE:
        return True
    return False


def verify_text(text: str, *sources) -> tuple[bool, list[str]]:
    """
    (numbers_verified, unverified_numbers) for one case text against
    its source data. Magnitude-only comparison (see LIMITATION).
    """
    pool = collect_source_numbers(*sources)
    unmatched = []
    for claim in extract_claims(text):
        if not any(_matches(abs(claim), fact) for fact in pool):
            unmatched.append(
                f"cited {claim:g} - no matching value in source data"
            )
    return (not unmatched, unmatched)


def verify_premarket_case_file(side: str) -> dict[str, dict]:
    """
    Verify every case in one pre-market case file against the scan /
    news / candle files, and rewrite the file with per-case
    `numbers_verified` + `unverified_numbers` fields added. Called
    automatically by the bull/bear runners after they write; also
    runnable standalone to re-check by hand.
    """
    case_path = PM_CASE_FILES[side]
    if not case_path.exists():
        raise SystemExit(f"{case_path} not found")
    payload = json.loads(case_path.read_text())

    scan_rows = {}
    if PM_SCAN_PATH.exists():
        scan = json.loads(PM_SCAN_PATH.read_text())
        scan_rows = {r["symbol"]: r for r in scan["shortlist"]}
    news = (json.loads(PM_NEWS_PATH.read_text()).get("news", {})
            if PM_NEWS_PATH.exists() else {})
    candles = (json.loads(PM_CANDLES_PATH.read_text()).get("reads", {})
               if PM_CANDLES_PATH.exists() else {})

    text_key = "bull_case" if side == "bull" else "bear_case"
    flagged = 0
    for symbol, case in payload["cases"].items():
        ok, unmatched = verify_text(
            case[text_key],
            scan_rows.get(symbol, {}),
            news.get(symbol, []),
            candles.get(symbol, {}),
            case,  # the case's own fields (its TP/SL, its score)
        )
        case["numbers_verified"] = ok
        case["unverified_numbers"] = unmatched
        if not ok:
            flagged += 1
            print(f"[verify:{side}] {symbol}: {'; '.join(unmatched)}")

    case_path.write_text(json.dumps(payload, indent=2))
    print(f"[verify:{side}] {len(payload['cases']) - flagged}/"
          f"{len(payload['cases'])} cases numerically clean")
    return payload["cases"]


if __name__ == "__main__":
    for side, path in PM_CASE_FILES.items():
        if path.exists():
            verify_premarket_case_file(side)
        else:
            print(f"[verify:{side}] {path} not found - skipped")
