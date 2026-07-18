# trading-agent

A multi-agent intraday trading system built from the ground up as a learning project — each piece is added and understood individually rather than pulled in as a black box. Runs against Alpaca's **paper trading** account only; nothing here places live trades.

**Stack:** Python, [alpaca-py](https://github.com/alpacahq/alpaca-py) (paper only), [CrewAI](https://github.com/crewAIInc/crewAI) for agent orchestration, Google Gemini as the LLM (pinned to `gemini-3.1-flash-lite` via LiteLLM — a rolling alias once silently moved us onto a model with a 20-requests/day quota, so the model is pinned on purpose), Finnhub for earnings/news.

## Branches

- **`main`** — the current working version. Small changes commit straight here; only large rebuilds get a temporary working branch (merged into main and deleted when done).
- **`previous`** — rolling one-step-back fallback. Before each new change lands on main, `previous` is moved to main's pre-change commit — so it always holds the last known-good version from before the most recent change.

## The trading day

`orchestrator.py` is a pure timing layer — it decides *when*, never *what*. Every stage below is independently runnable; the orchestrator just sequences them on the market's clock (all times ET, from Alpaca's own calendar, so early-close days shrink the schedule automatically):

```
open-45min   pre-market chain
open         poll Alpaca's clock until actually open (bounded), then
             pre-market execution
open+30min   daily_scan()
every 30min  check_shortlist(), last run no later than close-45min
```

### Pre-market chain (six components, file seams between them)

```
market_calendar    gate: is there a session today? (every component checks)
premarket_scanner  whole dynamic universe (~2,300 stocks) -> pre-market
                   rel volume + gap vs a PRE-MARKET baseline -> top 12
                   -> data/premarket_scan.json
premarket_news     Finnhub headlines for the shortlist
                   -> data/premarket_news.json
candle_agent       LLM reads yesterday's daily + today's PM candle
                   (pre-computed ratios; code divides, model judges)
                   -> data/premarket_candles.json
premarket bull /   adversarial debate per stock: strongest genuine case
bear agents        for and against, on identical evidence, anchored 0-1
                   scores; must cite headlines or note their absence;
                   auto fact-checked -> data/premarket_{bull,bear}_cases.json
premarket_trader   no LLM: net = bull - bear, buy at net >= 0.2,
                   bear-risk-tempered TP/SL -> data/premarket_decisions.json
premarket_execution reads decisions, applies guards (below), submits GTC
                   bracket orders - dry-run unless --live
```

### Regular-session pipeline

```
stage 0: portfolio state     (start of every check run)
  snapshot_portfolio - cash, buying power, holdings
                       -> data/portfolio_state.json (audit record)

stage 1: daily_scan          python pipeline.py scan
  held-symbol filter - shortlist slots never wasted on stocks already owned
  universe_builder   - all tradable US equities: price >= $3, avg volume
                       >= 500k, real stocks only (no ETPs/OTC/preferred)
  catalysts prescan  - one bulk Finnhub call: earnings in the next 1-3 days?
  scanner            - rel volume + % change + MA distance, z-scored, plus
                       an absolute-volume kicker and a catalyst boost
  -> data/shortlist.json

stage 2: check_shortlist     python pipeline.py check [--live]
  catalysts          - per-symbol earnings/dividends/news for the shortlist
  bull/bear debate   - same adversarial structure as pre-market (committed
                       one-sided cases, honest anchored scores, must cite
                       dated evidence or admit there is none)
  case verifier      - deterministic numeric fact-check of both case texts
  trader             - no LLM: net score -> buy/hold + tempered TP/SL
  execution agent    - no LLM: filters, sizes, submits GTC bracket orders
                       (dry-run unless --live)
```

The JSON file between every pair of stages is both the scheduling seam and the audit trail: each run leaves a record of what the system saw, argued, and decided. Exits are enforced broker-side via bracket orders (attached take-profit + stop-loss, one-cancels-other) — nothing watches positions after entry; the broker does.

## Safety guards (all in one place)

| Guard | Where | What it protects against |
|---|---|---|
| Calendar gate | every pre-market component + orchestrator | running against a closed market (weekends, holidays, half-days) |
| One-sided-evidence skip | signal orchestrators + premarket trader | a stock with only a bull case (or only a bear case) can never become a trade |
| Confidence threshold (>= 0.6) | execution agent | buys below the trader's net-score bar never execute (thresholds aligned by construction) |
| Numeric fact-checker | after every debate, both pipelines | cited numbers that don't trace to source data get flagged (`numbers_verified`) — numbers only, an invented *qualitative* claim passes; tripwire, not filter |
| Position sizing cap | execution agent | max $1,000 per position, whole shares, never the last 5% of buying power |
| Dead-quote guard | execution agent | market buys are never sized off a 0/absent ask (closed market, thin tape) |
| Price deviation guard (±2%) | premarket execution | the live open has moved >2% (either direction) from the price the debate argued about — the thesis no longer applies |
| Stale-decisions guard | premarket execution | yesterday's gap thesis can never execute today |
| GTC bracket orders | broker | exit legs never expire at the close, leaving an unprotected overnight position (Alpaca caps GTC at 90 days) |
| Dry-run by default | both execution paths | orders are only submitted with an explicit `--live` |
| Daily-quota latch | LLM runner | a burned Gemini daily quota fast-fails the run instead of retry-sleeping through guaranteed failures |

## Running

**Orchestrated (the normal way):**

```
.venv\Scripts\python.exe -m orchestrator            # full scheduled day, dry-run
.venv\Scripts\python.exe -m orchestrator --live     # ...with real paper orders
```

Start it before open−45min (13:45 UK time on normal days). All schedule math is US Eastern; log lines show both ET and your local time, so you never convert.

**Any stage manually (testing never waits for the clock):**

```
.venv\Scripts\python.exe -m orchestrator --force premarket
.venv\Scripts\python.exe -m orchestrator --force premarket_exec [--live]
.venv\Scripts\python.exe -m orchestrator --force daily_scan
.venv\Scripts\python.exe -m orchestrator --force check [--live]
```

**Or the underlying entry points directly:**

```
.venv\Scripts\python.exe pipeline.py scan | check [--live] | all [--live]
.venv\Scripts\python.exe -m tools.premarket_scanner [YYYY-MM-DD]
.venv\Scripts\python.exe -m tools.case_verifier
```

## Deliberately not built yet

So this README never implies more automation or safety than exists:

- **No hosted scheduling.** GitHub Actions is not wired up — the orchestrator only runs while your machine runs it. Its stage entry points and named schedule constants are shaped to translate directly into workflow cron lines when that day comes.
- **No calibration.** Every threshold (net-score 0.2, confidence 0.6, ±2% deviation, $1,000 cap, TP/SL tempering) is a reasoned first guess, deliberately deferred until there's live paper history to calibrate against.
- **No portfolio-level risk manager.** Nothing limits sector concentration, total simultaneous exposure, or correlated positions — each trade is judged alone. This is the planned "risk agent" slot between signal and execution, not started.
- **Numbers-only fact-checking.** The case verifier cannot catch an invented qualitative claim (a fabricated catalyst) — only numeric drift.

## Setup

Requires Python 3.12 (3.14 doesn't yet have prebuilt wheels for some dependencies — a `.venv` on 3.12 is recommended).

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys — all three are required:

| Variable | Source |
|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | paper keys from the [Alpaca dashboard](https://app.alpaca.markets/paper/dashboard/overview) |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| `FINNHUB_API_KEY` | free tier at [finnhub.io](https://finnhub.io/register) |

`.env` is gitignored; never commit real keys. Runtime artifacts live in `data/` (also gitignored).
