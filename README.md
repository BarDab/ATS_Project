# ATS Frequent Batch Project

Market microstructure simulation based on **Budish, Cramton & Shim (2015)**.
Models a continuous limit order book where a market maker posts two-sided quotes,
snipers exploit the lag between price moves and MM quote updates, and investors
arrive randomly. The central question: how much does sniper presence hurt the MM,
and how does it affect spread and liquidity?

See `PROJECT_CONTEXT.md` for full documentation, findings, and next steps.

---

## Model Logic

### Price process (Y)

The "true value" of the asset follows a **jump-diffusion** process — a combination of
continuous Brownian motion and discrete Poisson jumps, both applied at each jump event
and snapped to the tick grid:

```
y += σ·√dt·Z  +  direction × n_ticks × tick_size
y  = round(y / tick_size) * tick_size
```

The BM component produces smooth trending at long horizons (full-day chart looks like
a real price path). The jump component produces the discrete steps visible at
millisecond resolution. Jump size is drawn independently each time from
`[1, 2, 3, 4]` ticks with probabilities `[0.70, 0.20, 0.07, 0.03]`.

### Agents

**Market Maker** — observes Y with a 10ms lag, posts bid and ask around the
observed price. When the book midpoint has drifted from Y by more than a threshold
(stale quotes), the MM pulls the exposed side and reposts wider. When at inventory
limits, the corresponding side is not posted.

**Snipers (×2)** — observe Y with a 1ms lag, faster than the MM. If Y has moved
enough that an MM quote is mispriced by more than `sniper_min_edge_ticks`, the sniper
submits a market order to take it. Both snipers react to the same jump simultaneously;
their order of execution is randomised to avoid systematic advantage.

**Investors** — arrive via a Poisson process (1/sec by default), submit a random-direction
market order of size 1. They represent uninformed flow.

### Event queue

The simulation is discrete-event. For each Y jump at time `t`:

```
t + 0.000s   Y_JUMP          mark-to-market, sample spread
t + 0.001s   SNIPER_OBSERVE  snipers check for stale quotes (order randomised)
t + 0.010s   MM_OBSERVE      MM cancels/reposts quotes
```

The window **[t+1ms, t+10ms]** is where the arms race happens: snipers can hit
stale MM quotes before the MM has a chance to update them.

---

## Key Assumptions

- **Single asset, single MM.** No competition between market makers.
- **MM has no alpha.** It only reacts to Y; it cannot predict where Y is going.
- **Snipers are pure arbitrageurs.** They trade only when there is a calculable edge,
  hold no inventory, and mark profits immediately to the current Y.
- **Investors are noise traders.** Random direction, fixed size, no price sensitivity.
  They will fill against whatever is in the book, or not fill at all if the book is empty.
- **No transaction costs or fees** for any agent.
- **Refill-on-fill**: when an MM order is taken, a new order is immediately reposted
  at the same price. This keeps the book two-sided but leaves a stale quote in place
  until the next MM_OBSERVE fires 10ms later.
- **Spread sampling**: avg spread is measured at each Y_JUMP (before anyone reacts).
  One-sided book states (spread = None) are excluded. This means sniper-caused
  one-sided periods don't inflate the average — a known measurement limitation.
- **No mean reversion** in Y. Early versions used a dead-band reversion mechanism
  but it caused artificial oscillation. Real intraday prices are non-stationary;
  the BM component handles bounded drift naturally over an 8-hour window.

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

`uv sync` reads `pyproject.toml` and `uv.lock` and creates `.venv` with pinned
versions of all dependencies (`numpy`, `matplotlib`, `jupyter`, `pandas`).
Commit `uv.lock` — it guarantees identical package versions across machines.

---

## Running

### Phase 1 — Price process visualisation

```bash
uv run jupyter notebook notebooks/phase1_processes.ipynb
```

Reproduces a figure similar to Figure I in the BCS paper: the Y jump-diffusion
process and lagged MM quotes shown at 4 time resolutions (full day → 250ms).

### Phase 2 — Agent simulation

```bash
uv run jupyter notebook notebooks/phase2_simulation.ipynb
```

Runs the full discrete-event simulation with MM, snipers, and investors.
Produces PnL and inventory charts comparing with-snipers vs without-snipers,
a summary statistics table, and a spread distribution plot.

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
│   └── core.py                   # All simulation logic — both phases
├── notebooks/
│   ├── phase1_processes.ipynb    # Y process at 4 time resolutions
│   └── phase2_simulation.ipynb   # Agent simulation: with vs without snipers
├── src/
│   ├── engine.py                 # Standalone continuous LOB (unused in sim)
│   ├── auction.py                # Batch auction clearing (future Phase 3)
│   └── ...                       # bench.py, cli.py, gen.py, metrics.py
├── PROJECT_CONTEXT.md            # Full model documentation and findings
├── pyproject.toml                # Dependencies managed by uv
├── uv.lock                       # Pinned dependency versions (commit this)
└── .venv/                        # Created by uv sync (not committed)
```
