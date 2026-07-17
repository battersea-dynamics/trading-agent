# trading-agent

A multi-agent intraday trading system built from the ground up as a learning project — each piece is added and understood individually rather than pulled in as a black box. Runs against Alpaca's **paper trading** account only; nothing here places live trades.

**Stack:** Python, [alpaca-py](https://github.com/alpacahq/alpaca-py) (paper only), [CrewAI](https://github.com/crewAIInc/crewAI) for agent orchestration, Google Gemini as the LLM, Finnhub for earnings/news.

## Branches

- **`main`** — the current working version. Small changes commit straight here; only large rebuilds get a temporary working branch (merged into main and deleted when done).
- **`previous`** — rolling one-step-back fallback. Before each new change lands on main, `previous` is moved to main's pre-change commit — so it always holds the last known-good version from before the most recent change.

## Pipeline

```
stage 1: daily_scan          python pipeline.py scan
  universe_builder  - all tradable US equities, filtered to price >= $3,
                      avg volume >= 500k, real stocks only (no ETPs/OTC)
  catalysts prescan - one bulk Finnhub call: who reports earnings in 1-3 days?
  scanner           - rel volume + % change + MA distance, z-scored, plus
                      an absolute-volume kicker (catches 1.5-2x builds
                      early) and a boost for catalyst-flagged names
  -> data/shortlist.json (the seam where a scheduler will plug in)

stage 2: check_shortlist     python pipeline.py check [--live]
  catalysts         - per-symbol earnings/dividends/news for the shortlist
  signal agent      - LLM judgment: buy/hold + confidence + TP/SL percents
  execution agent   - no LLM: filters, sizes, submits Alpaca bracket
                      orders (dry-run unless --live)
```

Stage 1 is cheap and LLM-free; stage 2 spends LLM calls and (with `--live`)
paper money. The JSON file between them is the scheduling seam and the audit
trail. Exits are enforced broker-side via bracket orders (attached
take-profit + stop-loss, OCO) — nothing watches positions after entry.

## Setup

Requires Python 3.12 (3.14 doesn't yet have prebuilt wheels for some dependencies — a `.venv` on 3.12 is recommended).

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys: paper-trading Alpaca keys from the [Alpaca dashboard](https://app.alpaca.markets/paper/dashboard/overview), a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey), and a free Finnhub key from [finnhub.io](https://finnhub.io/register). `.env` is gitignored; never commit real keys.

## Running

```
.venv\Scripts\python.exe pipeline.py scan          # build/refresh shortlist
.venv\Scripts\python.exe pipeline.py check         # judge + dry-run report
.venv\Scripts\python.exe pipeline.py check --live  # submit paper bracket orders
.venv\Scripts\python.exe pipeline.py all [--live]  # both stages
```
