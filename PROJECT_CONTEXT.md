# ATS Project — Context & Documentation

## What This Project Is

A market microstructure simulation based on **Budish, Cramton & Shim (2015)** —
the paper arguing that continuous limit order books create a structural arms race
between market makers and high-frequency snipers, and that frequent batch auctions
would eliminate it.

The simulation models a single asset whose "true value" (Y) evolves as a jump-diffusion
process. A market maker posts two-sided quotes around Y. Snipers exploit the lag between
Y moving and the MM updating its quotes. Investors arrive randomly and trade for
exogenous reasons.

The central question: **how much does the presence of snipers hurt the market maker,
and how does that affect spread and liquidity?**

---

## Project Structure

```
simulation/
  core.py               — all simulation logic (both phases)
notebooks/
  phase1_processes.ipynb  — Y process visualisation at 4 time resolutions
  phase2_simulation.ipynb — agent simulation: with vs without snipers
src/
  engine.py             — standalone continuous LOB (price-time priority, unused in sim)
  auction.py            — batch auction clearing logic (future use)
  batch_auction.py, bench.py, cli.py, gen.py, metrics.py
logs/                   — CSV + JSON output when enable_logging=True (gitignored)
```

---

## The Price Process (Y)

Y is a **jump-diffusion** (Merton 1976 style):

```
y += BM_move + jump
y = round(y / tick_size) * tick_size
```

- **BM component** (`σ · √dt · Z`): gives smooth trending at long horizons —
  makes full-day charts look like real prices rather than pure noise.
- **Jump component**: discrete Poisson jumps of 1–4 ticks, drawn at each inter-arrival
  time. Makes the process visually step-like at millisecond resolution.
- All output is snapped to the tick grid after each step.

Key params:


| Param                  | Default             | Meaning                                             |
| ---------------------- | ------------------- | --------------------------------------------------- |
| `Y0`                   | 100.0               | Starting price                                      |
| `tick_size`            | 0.01                | Minimum price increment                             |
| `target_jumps_per_day` | 100,000             | Controls jump intensity (class var, not init field) |
| `jump_size_probs`      | [0.7,0.2,0.07,0.03] | P(1-tick), P(2-tick), P(3-tick), P(4-tick)          |
| `sigma`                | 0.05                | BM diffusion coefficient                            |
| `seed`                 | 42                  | Controls all randomness                             |

---

## Agent Design

### Market Maker

Posts a two-sided quote around Y, observed with a **10ms lag** (`mm_lag`).

**Inventory skew**: when MM is long, it skews the mid price down to encourage selling:

```
skewed_mid = Y - inventory × inventory_skew_factor × tick_size
```

**Divergence response** (`mm_pull_mode`): if the book midpoint has moved more than
`mm_divergence_threshold_ticks` from Y (i.e. quotes are stale), the MM responds:

- `"exposed"` *(default)*: only cancel the stale side (the one snipers would hit),
  repost it wider. The other side stays.
- `"both"`: cancel and repost both sides wider.
- `"skew"`: do nothing now, update skew on next normal repost.

**Refill on fill** (`mm_refill_on_fill=True`): when a limit order is filled, immediately
repost at the same price. This keeps the book two-sided but at a now-stale price until
the next MM_OBSERVE event.

**PnL tracking**: realized PnL uses volume-weighted average entry price.
Unrealized PnL = `inventory × (current_Y − avg_entry_price)`.

### Snipers (2 by default)

Observe Y with a **1ms lag** (`sniper_lag`) — faster than the MM.

Logic: if `Y − best_ask > sniper_min_edge_ticks × tick_size`, submit market buy
(snipe stale cheap ask). Symmetric for bids.

**The race**: when both snipers see the same Y jump, their SNIPER_OBSERVE events
have identical timestamps. The event queue shuffles them randomly before processing,
so neither sniper has a systematic advantage.

**Sniper PnL** = `fill_quantity × (current_Y − fill_price)` on buys, reversed on sells.
This is mark-to-market profit: they immediately value the position at current Y.

### Investors

Arrive via Poisson process (`investor_arrival_rate = 1.0/sec`).
Submit a random-direction market order of size 1. They don't optimise.

---

## Event Queue (Discrete Event Simulation)

All agents are driven by a single sorted event queue. For each Y jump at time `t`:

```
t + 0.000   Y_JUMP          — mark-to-market, log
t + 0.001   SNIPER_OBSERVE  — sniper1 or sniper2 (randomly ordered)
t + 0.001   SNIPER_OBSERVE  — the other sniper
t + 0.010   MM_OBSERVE      — MM updates quotes
```

Investor arrivals are interleaved from a separate Poisson stream.

The critical window is **[t+1ms, t+10ms]**: snipers can pick off stale MM quotes
before the MM has a chance to update. This is the mechanism the BCS paper focuses on.

---

## Key Simulation Finding (from our runs)

With 100,000 jumps/day, seed=42:


|                 | With Snipers | Without Snipers |
| --------------- | ------------ | --------------- |
| MM Realized PnL | 3,289        | 5,119           |
| Avg Spread      | 0.2745       | 0.4016          |
| Sniper Fills    | 12,783       | —              |

**Why avg spread is *smaller* with snipers** (counterintuitive):

- Without snipers: after a divergence, MM successfully cancels the exposed side and
  reposts wider → wide spread persists and gets sampled.
- With snipers: sniper hits the stale quote at t+1ms *before* the MM can widen it at
  t+10ms. The quote disappears (one-sided book → excluded from avg), then MM refills
  at the same stale price (narrow spread). Snipers pre-empt the MM's spread-widening
  response.

This illustrates the BCS adverse selection problem: snipers extract value not just by
taking stale quotes, but by preventing the MM from being compensated for the risk.

---

## How to Run

```bash
# Install dependencies (first time)
uv sync

# Phase 1 — Y process visualisation
uv run jupyter notebook notebooks/phase1_processes.ipynb

# Phase 2 — agent simulation
uv run jupyter notebook notebooks/phase2_simulation.ipynb

# Quick sanity check
uv run python3 -c "
from simulation.core import SimulationParams, SimulationRunner
r = SimulationRunner(SimulationParams()).run()
print(r.mm_pnl_history[-1])
"
```

---

## Code Entry Points


| You want to...               | Look at...                                                         |
| ---------------------------- | ------------------------------------------------------------------ |
| Change price process         | `simulate_Y()` — line 61                                          |
| Change MM behaviour          | `MarketMaker.observe_Y()` — line 265                              |
| Change sniper aggressiveness | `Sniper.observe_Y()` and `sniper_min_edge_ticks` param             |
| Add a new agent type         | Follow`Investor` pattern, pass `mm_fill_callback` for fill routing |
| Run with logging             | `SimulationParams(enable_logging=True)` → writes to `logs/`       |
| Tune the simulation          | All knobs live in`SimulationParams` (top of `core.py`)             |

---

## Open Questions / Next Steps (Phase 3)

- Implement **Frequent Batch Auctions** (`src/batch_auction.py` exists) as the
  alternative market structure and compare MM PnL / spread / sniper viability.
- Add **Hawkes process** for jump intensity (volatility clustering) — jumps that
  breed more jumps, as in real markets.
- Extend MM with **dynamic spread adjustment** based on realised adverse selection rate.
- Add **multiple MM agents** competing for flow.
