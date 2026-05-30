"""Market microstructure simulation — Phase 1 core."""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Section 1: Parameters
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    Y0: float = 100.0
    tick_size: float = 0.01
    target_jumps_per_day = 100000
    jump_size_probs = [0.7, 0.2, 0.07, 0.03]  # prob of 1,2,3,4 tick jump
    trading_hours: float = 8.0
    seed: int = 42
    mm_lag: float = 0.010
    mm_spread_ticks: int = 4
    sigma: float = 0.05

    lambda_jump: float = field(init=False)
    T: float = field(init=False)

    def __post_init__(self):
        self.lambda_jump = self.target_jumps_per_day / (self.trading_hours * 3600)
        self.T = self.trading_hours * 3600


# ---------------------------------------------------------------------------
# Section 2: Y Process
# ---------------------------------------------------------------------------

def simulate_Y(params: SimulationParams):
    """Jump diffusion process: continuous BM + discrete Poisson jumps.

    BM produces smooth trending at long horizons; jumps produce discrete
    steps visible at short horizons. Output is rounded to the tick grid.
    Returns sparse (times, prices) arrays — only records events at jump times.
    """
    rng = np.random.default_rng(params.seed)

    times = [0.0]
    prices = [params.Y0]

    t = 0.0
    y = params.Y0

    while True:
        dt = rng.exponential(1.0 / params.lambda_jump)
        t += dt
        if t > params.T:
            break
        bm_move = rng.normal(0, params.sigma * np.sqrt(dt))
        n_ticks = rng.choice([1, 2, 3, 4], p=params.jump_size_probs)
        jump = rng.choice([-1, 1]) * n_ticks * params.tick_size
        y += bm_move + jump
        y = round(y / params.tick_size) * params.tick_size
        times.append(t)
        prices.append(y)

    return np.array(times), np.array(prices)


# ---------------------------------------------------------------------------
# Section 3: Market Maker
# ---------------------------------------------------------------------------

class MarketMaker:

    def __init__(self, params: SimulationParams):
        self.params = params
        self.bid = None
        self.ask = None
        self._last_observed_at = None

    def observe_Y(self, y_value: float, observed_at_time: float):
        half_spread = self.params.mm_spread_ticks * self.params.tick_size / 2
        self.bid = y_value - half_spread
        self.ask = y_value + half_spread
        self._last_observed_at = observed_at_time

    def get_quotes(self):
        return (self.bid, self.ask)


# ---------------------------------------------------------------------------
# Section 4: MM Quote Time Series
# ---------------------------------------------------------------------------

def simulate_MM_quotes(params: SimulationParams, times: np.ndarray, prices: np.ndarray):
    """Build MM quote time series from Y process output.

    MM observes each jump after mm_lag delay. Returns sparse arrays aligned
    with the Y event times (shifted by mm_lag).
    """
    mm = MarketMaker(params)
    quote_times = times + params.mm_lag
    bids = np.empty(len(times))
    asks = np.empty(len(times))

    for i, (t, y) in enumerate(zip(times, prices)):
        mm.observe_Y(y, t + params.mm_lag)
        b, a = mm.get_quotes()
        bids[i] = b
        asks[i] = a

    return quote_times, bids, asks
