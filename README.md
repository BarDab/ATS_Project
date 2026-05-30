# ATS Frequent Batch Project

Market microstructure simulation based on Budish, Cramton & Shim (2015), modelling a compound Poisson price process and a lagged market maker.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and environment manager

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

Clone the repo and install dependencies:

```bash
git clone <repo-url>
cd ATS_Project
uv sync
```

`uv sync` reads `pyproject.toml` and creates a `.venv` with all required packages (`numpy`, `matplotlib`, `jupyter`).

## Running the simulation

### From the command line

```bash
uv run python3 -c "
from simulation.core import SimulationParams, simulate_Y, simulate_MM_quotes
params = SimulationParams()
times, prices = simulate_Y(params)
quote_times, bids, asks = simulate_MM_quotes(params, times, prices)
print(f'Total jumps: {len(times) - 1}')
"
```

### Jupyter notebook (Phase 1 figure)

```bash
uv run jupyter notebook notebooks/phase1_processes.ipynb
```

This opens the notebook in your browser. Run all cells (`Kernel > Restart & Run All`) to reproduce the multi-resolution price process figure and summary statistics.

## Project structure

```
ATS_Project/
├── simulation/
│   └── core.py              # SimulationParams, simulate_Y, MarketMaker, simulate_MM_quotes
├── notebooks/
│   └── phase1_processes.ipynb  # Phase 1 figure — Y process at 4 time resolutions
├── src/
│   ├── engine.py            # Continuous order book (price-time priority LOB)
│   ├── auction.py           # Batch auction clearing logic
│   └── ...                  # Benchmarks, CLI, metrics
├── pyproject.toml           # Dependencies managed by uv
└── .venv/                   # Created by uv sync (not committed)
```

## Key parameters

All simulation parameters live in `SimulationParams` in `simulation/core.py`. Defaults:

| Parameter | Default | Description |
|---|---|---|
| `Y0` | `100.0` | Initial asset price |
| `tick_size` | `0.25` | Minimum price increment |
| `target_jumps_per_day` | `800` | Expected jumps per trading day |
| `trading_hours` | `8` | Length of trading day in hours |
| `mm_lag` | `0.010` | Market maker observation delay (10 ms) |
| `mm_spread_ticks` | `2` | Market maker spread in ticks |
| `seed` | `42` | Random seed for reproducibility |

To change parameters, pass them when constructing `SimulationParams`:

```python
params = SimulationParams(seed=123, mm_lag=0.005, target_jumps_per_day=1200)
```
