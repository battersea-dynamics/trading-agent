# trading-agent

A multi-agent trading system built from the ground up as a learning project — each agent is added and understood individually rather than pulled in as a black box. Runs against Alpaca's **paper trading** account only; nothing here places live trades.

**Stack:** Python, [alpaca-py](https://github.com/alpacahq/alpaca-py) (paper only), [CrewAI](https://github.com/crewAIInc/crewAI) for agent orchestration, Google Gemini as the LLM backend.

## Current state

- **`tools/broker.py`** — the harness. Plain functions (`get_account`, `get_quote`, `place_market_order`) that talk to Alpaca directly, with no LLM involved. This is the only file that touches money, and the only place that will ever call `place_market_order`.
- **`agents/signal_agent.py`** — the first (and so far only) agent. A single, isolated CrewAI `Agent` that takes a live quote and produces a structured opinion — `{symbol, signal, reasoning, confidence}` — as `buy`/`sell`/`hold`. It has no access to `place_market_order`: it can advise, not trade.

Planned next: a risk agent (sits between signal and execution, sizes/vetoes trades) and an execution agent (the only thing allowed to call `place_market_order`).

## Setup

Requires Python 3.12 (3.14 doesn't yet have prebuilt wheels for some dependencies — a `.venv` on 3.12 is recommended).

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own keys — paper-trading Alpaca keys from the [Alpaca dashboard](https://app.alpaca.markets/paper/dashboard/overview), and a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey). `.env` is gitignored; never commit real keys.

## Running

```
.venv\Scripts\python.exe -m agents.signal_agent AAPL
```

Prints the signal agent's decision as JSON for the given symbol (defaults to `AAPL`).
