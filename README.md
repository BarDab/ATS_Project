# ATS Frequent Batch Project

Market microstructure simulation based on **Budish, Cramton & Shim (2015)**.
Models a continuous limit order book where a market maker posts two-sided quotes,
snipers exploit the lag between price moves and MM quote updates, and investors
arrive randomly.

See `PROJECT_CONTEXT.md` for full parameter reference, findings, and next steps.

---

## How This Simulation Works

The simulation is **event-driven**: every action that any agent can take is represented
as an event with a timestamp, and a central loop processes them in chronological order.

### Pre-generated events

Before the simulation starts, three types of events are built upfront and sorted by
timestamp — these represent scheduled observations rather than decisions:

- **Y_JUMP** — the true asset price moves. Fires at every jump of the price process.
  No agent acts yet; this just updates the ground truth.
- **SNIPER_OBSERVE** — each sniper gets to look at the current price, 1ms after the
  corresponding jump. If quotes are stale enough to be profitable, a sniper may act.
- **MM_OBSERVE** — the market maker gets to look at the current price, 10ms after the
  jump. It then cancels and reposts its quotes. Always fires after the snipers for the
  same jump, which is the core mechanism of the arms race.

Investor arrivals are also pre-generated as a Poisson stream and interleaved into the
same sorted list.

When two snipers react to the same jump they land at the exact same timestamp. To avoid
giving either a systematic advantage, their order is randomised before the simulation
runs.

### The event loop

`SimulationRunner.run()` processes generated events one by one from the queue and dispatches to the
relevant agent:

```
Y_JUMP            → update current price, sample spread, record MM PnL snapshot
SNIPER_OBSERVE    → sniper checks for stale quotes and hits them if profitable
MM_OBSERVE        → MM cancels stale quotes and reposts at current fair value
INVESTOR_ARRIVE   → investor submits a random-direction market order
DEFERRED_LABEL    → fires 100ms after a MM fill; classifies it as informed or not
```

`DEFERRED_LABEL` events are the only ones created *during* the simulation (pushed onto
the heap when a fill occurs). Everything else is known before the first tick. The 100ms delay on `DEFERRED_LABEL` is intentional: at the moment a fill happens the MM cannot yet know whether it was informed or not — Y may still move further in the same direction, or may reverse. Waiting 100ms gives the price time to reveal its
post-fill direction, so the classification (did Y move away from the fill price by
more than `min_edge` ticks?) reflects what actually happened rather than a noisy
snapshot taken mid-jump. In a real market this corresponds to the time a dealer needs
to observe post-trade price impact before deciding whether the counterparty had
private information.

### Order submission delay

The parameter `order_submission_delay` (default `0.0`) models the latency between an
agent *deciding* to submit an order and that order *landing on the book* — exchange
network latency. When set to a non-zero value, limit and market order submissions are
not executed inline; instead two additional event types are pushed onto the heap:

```
DEFERRED_LIMIT_ORDER  → fires at decision_time + order_submission_delay; submits a limit order to the book
DEFERRED_MARKET_ORDER → fires at decision_time + order_submission_delay; submits a market order and runs fill callbacks
```

Cancellations are always immediate — they represent a risk management decision, not a
new order travelling to the exchange.

---

## Agents

### Market Maker

The MM acts as a passive liquidity provider: it continuously quotes a **bid and an ask**
around its observation of the fair value Y, earning the spread on uninformed flow while
absorbing adverse selection from snipers.

**Quoting**: at each `MM_OBSERVE`, the MM posts a bid below and an ask above its
(lagged) view of Y. It uses a single spread width that it adjusts dynamically over time.
If the current book midpoint has drifted more than a threshold from Y (stale quotes),
it reposts wider to compensate for the increased risk.

**Inventory**: the MM accumulates inventory as it fills orders. It skews its quoted
midpoint away from the direction it is long to encourage mean-reverting flow — long
positions push the mid down, short positions push it up. For simplicity there is no
hard inventory target or active hedging; the skew is the only inventory management.

**Dynamic spread**:  the MM tracks which of its recent fills came from
informed traders (snipers) versus uninformed ones (investors). It estimates **alpha**
— the fraction of informed fills over the last 50 trades — and widens its spread when
alpha is high:

```
spread = base_spread × (1 + alpha_sensitivity × alpha)
```

The spread is clamped between a floor and a ceiling. This means the MM
responds to a high-sniper environment by charging more, partially offsetting
the adverse selection loss.

**PnL decomposition**: realized PnL is split into `spread_income` (earned on
uninformed fills) and `adverse_selection_loss` (mark-to-market loss on informed fills).

### Snipers

Two snipers observe Y with a 1ms lag — faster than the MM's 10ms lag. After each
price jump, each sniper checks whether any MM quote is mispriced enough to be worth
hitting. If `Y − best_ask > min_edge`, the sniper submits a market buy; symmetrically
for the ask side.

The 9ms window between sniper observation (t+1ms) and MM observation (t+10ms) is
where sniping happens. Snipers never hold inventory — they value every fill
immediately at the current Y (we don't care much about their profit in this project, so it's just a mock).

### Investors

Investors represent uninformed, exogenous demand. They arrive via a Poisson process
(1 per second by default) and submit a random-direction market order of size 1. They
do not optimise their timing or price — they just need to trade. Same as Snipers they are submitting market orders only.

---

## Random Processes

### Price process (Y)

The true asset value follows a **jump-diffusion**: at each event, both a Brownian
motion increment and a discrete jump are applied, then the result is snapped to the
tick grid:

```
y += σ·√dt·Z  +  direction × n_ticks × tick_size
y  = round(y / tick_size) * tick_size
```

Jump times are drawn from an exponential distribution targeting 100,000 jumps per
8-hour day. Jump size is drawn from `[1, 2, 3, 4]` ticks with probabilities
`[0.70, 0.20, 0.07, 0.03]`. The BM component produces smooth drift visible at long
time horizons; the jump component produces the discrete steps visible at millisecond
resolution. We tried to model this process with sole Poisson - while it look as expected in high frequency periods it looked like white noise on lower frequencies. 

### Investor arrivals

Investor arrivals are an independent Poisson process with rate 1/sec. Inter-arrival
times are drawn from an exponential distribution, accumulated until they exceed the
trading day length. Direction (buy/sell) is drawn uniformly at random at arrival time.

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and environment manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

```bash
git clone <repo-url>
cd ATS_Project
uv sync
```

---

## Running

### Phase 1 — Price process visualisation

```bash
uv run jupyter notebook notebooks/phase1_processes.ipynb
```

Reproduces a figure similar to Figure I in the BCS paper: the Y jump-diffusion
process and lagged MM quotes shown at 4 time resolutions (full day → 250ms).

### Phase 2 / 3 — Agent simulation

```bash
uv run jupyter notebook notebooks/phase2_simulation.ipynb
```

Runs the full discrete-event simulation with MM, snipers, and investors.
Compares with-snipers vs without-snipers across PnL, inventory, spread, and
MM alpha / spread adjustment over time.

### Quick sanity check

```bash
uv run python3 -c "
from simulation.core import SimulationParams, SimulationRunner
r = SimulationRunner(SimulationParams()).run()
print(r.mm_pnl_history[-1])
"
```

---

## Project structure

```
ATS_Project/
├── simulation/
│   ├── core.py        # SimulationParams, Y process, OrderBook, SimulationRunner
│   ├── agents.py      # MarketMaker, Sniper, Investor
│   └── events.py      # Event dataclasses + EventLogger
├── notebooks/
│   ├── phase1_processes.ipynb    # Y process at 4 time resolutions
│   └── phase2_simulation.ipynb   # Agent simulation + Phase 3 analysis
├── src/
│   ├── engine.py      # Standalone continuous LOB (unused in sim)
│   ├── auction.py     # Batch auction clearing (future Phase 4)
│   └── ...
├── PROJECT_CONTEXT.md # Full parameter reference, findings, and next steps
├── pyproject.toml
└── uv.lock
```
