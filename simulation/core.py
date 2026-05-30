"""Market microstructure simulation — Phase 1 & 2 core."""

from __future__ import annotations
import bisect
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


# ---------------------------------------------------------------------------
# Section 1: Parameters
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    # --- Price process ---
    Y0: float = 100.0
    tick_size: float = 0.01
    target_jumps_per_day = 100000          # class var
    jump_size_probs = [0.7, 0.2, 0.07, 0.03]   # class var
    trading_hours: float = 8.0
    seed: int = 42
    sigma: float = 0.05
    # --- Phase 1 quote tracking ---
    mm_lag: float = 0.010
    mm_spread_ticks: int = 4
    # --- Market Maker (Phase 2) ---
    mm_base_spread_ticks: int = 4
    mm_max_inventory: float = 10.0
    mm_inventory_skew_factor: float = 0.1
    mm_pull_mode: str = "exposed"
    mm_divergence_threshold_ticks: int = 2
    mm_refill_on_fill: bool = True
    # --- Snipers ---
    sniper_lag: float = 0.001
    sniper_min_edge_ticks: int = 2
    sniper_order_size: int = 1
    n_snipers: int = 2
    # --- Investors ---
    investor_arrival_rate: float = 1.0
    investor_order_size: int = 1
    # --- Logging ---
    enable_logging: bool = False
    log_dir: str = "logs"

    lambda_jump: float = field(init=False)
    T: float = field(init=False)

    def __post_init__(self):
        self.lambda_jump = self.target_jumps_per_day / (self.trading_hours * 3600)
        self.T = self.trading_hours * 3600


# ---------------------------------------------------------------------------
# Section 2: Y Process
# ---------------------------------------------------------------------------

def simulate_Y(params: SimulationParams):
    """Jump diffusion: BM (smooth long-horizon trend) + Poisson jumps (discrete short-horizon steps)."""
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
# Section 3: Simple Market Maker (Phase 1 quote tracker)
# ---------------------------------------------------------------------------

class SimpleMarketMaker:

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
# Section 4: MM Quote Time Series (Phase 1)
# ---------------------------------------------------------------------------

def simulate_MM_quotes(params: SimulationParams, times: np.ndarray, prices: np.ndarray):
    """Build MM quote time series: sparse arrays shifted by mm_lag."""
    mm = SimpleMarketMaker(params)
    quote_times = times + params.mm_lag
    bids = np.empty(len(times))
    asks = np.empty(len(times))
    for i, (t, y) in enumerate(zip(times, prices)):
        mm.observe_Y(y, t + params.mm_lag)
        bids[i], asks[i] = mm.get_quotes()
    return quote_times, bids, asks


# ---------------------------------------------------------------------------
# Section 6: Central Order Book
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    agent_id: str
    side: str
    price: float
    quantity: int
    timestamp: float


class OrderBook:

    def __init__(self):
        self.bids: list[Order] = []
        self.asks: list[Order] = []
        self._order_counter: dict[str, int] = {}
        self._order_index: dict[str, Order] = {}

    def _next_id(self, agent_id: str) -> str:
        n = self._order_counter.get(agent_id, 0) + 1
        self._order_counter[agent_id] = n
        return f"{agent_id}_{n:04d}"

    def submit_limit(self, agent_id: str, side: str, price: float,
                     quantity: int, timestamp: float) -> str:
        oid = self._next_id(agent_id)
        order = Order(oid, agent_id, side, price, quantity, timestamp)
        self._order_index[oid] = order
        if side == "bid":
            bisect.insort(self.bids, order, key=lambda o: (-o.price, o.timestamp))
        else:
            bisect.insort(self.asks, order, key=lambda o: (o.price, o.timestamp))
        return oid

    def cancel_order(self, order_id: str) -> bool:
        order = self._order_index.pop(order_id, None)
        if order is None:
            return False
        book = self.bids if order.side == "bid" else self.asks
        try:
            book.remove(order)
            return True
        except ValueError:
            return False

    def submit_market(self, agent_id: str, side: str, quantity: int,
                      timestamp: float) -> list[dict]:
        fills = []
        remaining = quantity
        book = self.asks if side == "bid" else self.bids
        while remaining > 0 and book:
            best = book[0]
            fill_qty = min(remaining, best.quantity)
            fills.append({"price": best.price, "quantity": fill_qty,
                          "matched_order_id": best.order_id,
                          "matched_agent_id": best.agent_id,
                          "taker_side": side})
            remaining -= fill_qty
            best.quantity -= fill_qty
            if best.quantity == 0:
                book.pop(0)
                self._order_index.pop(best.order_id, None)
        return fills

    def get_best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    def get_best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    def get_midpoint(self) -> float | None:
        b, a = self.get_best_bid(), self.get_best_ask()
        return (b + a) / 2 if b is not None and a is not None else None

    def get_spread(self) -> float | None:
        b, a = self.get_best_bid(), self.get_best_ask()
        return a - b if b is not None and a is not None else None


# ---------------------------------------------------------------------------
# Section 7: Market Maker Agent (Phase 2)
# ---------------------------------------------------------------------------

class MarketMaker:

    def __init__(self, params: SimulationParams, book: OrderBook,
                 logger: "EventLogger"):
        self.params = params
        self.book = book
        self.logger = logger
        self.inventory: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.avg_entry_price: float = 0.0
        self.current_bid_id: str | None = None
        self.current_ask_id: str | None = None
        self.last_observed_Y: float | None = None
        self.last_observed_time: float | None = None
        self._fill_count: int = 0
        self._last_bid_price: float | None = None
        self._last_ask_price: float | None = None

    def _snap(self, price: float) -> float:
        ts = self.params.tick_size
        return round(price / ts) * ts

    def _cancel_bid(self):
        if self.current_bid_id:
            self.book.cancel_order(self.current_bid_id)
            self.current_bid_id = None

    def _cancel_ask(self):
        if self.current_ask_id:
            self.book.cancel_order(self.current_ask_id)
            self.current_ask_id = None

    def _cancel_both(self):
        self._cancel_bid()
        self._cancel_ask()

    def _post_bid(self, mid: float, half: float, ts: float):
        if self.inventory >= self.params.mm_max_inventory:
            return
        price = self._snap(mid - half)
        self._last_bid_price = price
        self.current_bid_id = self.book.submit_limit("mm", "bid", price, 1, ts)
        self.logger.log_event(ts, "SUBMIT", "mm", order_id=self.current_bid_id,
                              side="bid", price=price, quantity=1,
                              book_bid=self.book.get_best_bid(),
                              book_ask=self.book.get_best_ask(),
                              Y_value=self.last_observed_Y,
                              mm_inventory=self.inventory,
                              mm_realized_pnl=self.realized_pnl,
                              mm_unrealized_pnl=self.unrealized_pnl)

    def _post_ask(self, mid: float, half: float, ts: float):
        if self.inventory <= -self.params.mm_max_inventory:
            return
        price = self._snap(mid + half)
        self._last_ask_price = price
        self.current_ask_id = self.book.submit_limit("mm", "ask", price, 1, ts)
        self.logger.log_event(ts, "SUBMIT", "mm", order_id=self.current_ask_id,
                              side="ask", price=price, quantity=1,
                              book_bid=self.book.get_best_bid(),
                              book_ask=self.book.get_best_ask(),
                              Y_value=self.last_observed_Y,
                              mm_inventory=self.inventory,
                              mm_realized_pnl=self.realized_pnl,
                              mm_unrealized_pnl=self.unrealized_pnl)

    def observe_Y(self, y_value: float, observed_at_time: float,
                  current_book_mid: float | None):
        self.last_observed_Y = y_value
        self.last_observed_time = observed_at_time
        ts = self.params.tick_size
        skewed_mid = y_value - self.inventory * self.params.mm_inventory_skew_factor * ts
        normal_half = self.params.mm_base_spread_ticks * ts / 2
        wide_half = self.params.mm_base_spread_ticks * 2 * ts / 2
        div = (abs(current_book_mid - y_value) / ts
               if current_book_mid is not None else 0.0)

        if div > self.params.mm_divergence_threshold_ticks:
            mode = self.params.mm_pull_mode
            if mode == "both":
                self._cancel_both()
                self._post_bid(skewed_mid, wide_half, observed_at_time)
                self._post_ask(skewed_mid, wide_half, observed_at_time)
            elif mode == "exposed":
                if current_book_mid is not None and y_value > current_book_mid:
                    self._cancel_ask()
                    self._post_ask(skewed_mid, wide_half, observed_at_time)
                else:
                    self._cancel_bid()
                    self._post_bid(skewed_mid, wide_half, observed_at_time)
            # "skew": do nothing now
        else:
            self._cancel_both()
            self._post_bid(skewed_mid, normal_half, observed_at_time)
            self._post_ask(skewed_mid, normal_half, observed_at_time)

    def on_fill(self, side: str, fill_price: float, fill_quantity: int,
                timestamp: float, current_Y: float):
        prev_inv = self.inventory
        delta = fill_quantity if side == "bid" else -fill_quantity
        new_inv = prev_inv + delta

        if prev_inv == 0.0:
            self.avg_entry_price = fill_price
        elif (abs(new_inv) > abs(prev_inv) and
              (new_inv == 0 or (prev_inv > 0) == (new_inv > 0))):
            self.avg_entry_price = (
                self.avg_entry_price * abs(prev_inv) + fill_price * fill_quantity
            ) / abs(new_inv)
        else:
            closing_qty = min(fill_quantity, abs(prev_inv))
            if side == "ask":
                self.realized_pnl += closing_qty * (fill_price - self.avg_entry_price)
            else:
                self.realized_pnl += closing_qty * (self.avg_entry_price - fill_price)
            if new_inv != 0 and ((prev_inv > 0) != (new_inv > 0)):
                self.avg_entry_price = fill_price

        self.inventory = new_inv
        self.unrealized_pnl = (self.inventory * (current_Y - self.avg_entry_price)
                               if self.inventory != 0 else 0.0)
        self._fill_count += 1

        if side == "ask":
            self.current_ask_id = None
        else:
            self.current_bid_id = None

        self.logger.log_event(timestamp, "FILL", "mm", side=side,
                              fill_price=fill_price, fill_quantity=fill_quantity,
                              Y_value=current_Y,
                              book_bid=self.book.get_best_bid(),
                              book_ask=self.book.get_best_ask(),
                              mm_inventory=self.inventory,
                              mm_realized_pnl=self.realized_pnl,
                              mm_unrealized_pnl=self.unrealized_pnl)

        if self.params.mm_refill_on_fill:
            if side == "bid" and self._last_bid_price is not None:
                if self.inventory < self.params.mm_max_inventory:
                    self.current_bid_id = self.book.submit_limit(
                        "mm", "bid", self._last_bid_price, 1, timestamp)
            elif side == "ask" and self._last_ask_price is not None:
                if self.inventory > -self.params.mm_max_inventory:
                    self.current_ask_id = self.book.submit_limit(
                        "mm", "ask", self._last_ask_price, 1, timestamp)

    def mark_to_market(self, current_Y: float):
        self.unrealized_pnl = (self.inventory * (current_Y - self.avg_entry_price)
                               if self.inventory != 0 else 0.0)

    def get_pnl_snapshot(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "inventory": self.inventory,
            "avg_entry_price": self.avg_entry_price,
        }


# ---------------------------------------------------------------------------
# Section 8: Sniper Agent
# ---------------------------------------------------------------------------

class Sniper:

    def __init__(self, agent_id: str, params: SimulationParams,
                 book: OrderBook, logger: "EventLogger",
                 mm_fill_callback=None):
        self.agent_id = agent_id
        self.params = params
        self.book = book
        self.logger = logger
        self._mm_fill_cb = mm_fill_callback
        self.realized_pnl: float = 0.0
        self.trades_executed: int = 0

    def observe_Y(self, y_value: float, observed_at_time: float):
        ts = self.params.tick_size
        edge = self.params.sniper_min_edge_ticks * ts

        best_ask = self.book.get_best_ask()
        if best_ask is not None and (y_value - best_ask) > edge:
            for f in self.book.submit_market(self.agent_id, "bid",
                                             self.params.sniper_order_size,
                                             observed_at_time):
                self.on_fill(f["price"], f["quantity"], "bid", y_value)
                if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                    self._mm_fill_cb("ask", f["price"], f["quantity"],
                                     observed_at_time, y_value)
                self.logger.log_event(observed_at_time, "FILL", self.agent_id,
                                      side="bid", fill_price=f["price"],
                                      fill_quantity=f["quantity"],
                                      matched_order_id=f["matched_order_id"],
                                      matched_agent_id=f["matched_agent_id"],
                                      Y_value=y_value)

        best_bid = self.book.get_best_bid()
        if best_bid is not None and (best_bid - y_value) > edge:
            for f in self.book.submit_market(self.agent_id, "ask",
                                             self.params.sniper_order_size,
                                             observed_at_time):
                self.on_fill(f["price"], f["quantity"], "ask", y_value)
                if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                    self._mm_fill_cb("bid", f["price"], f["quantity"],
                                     observed_at_time, y_value)
                self.logger.log_event(observed_at_time, "FILL", self.agent_id,
                                      side="ask", fill_price=f["price"],
                                      fill_quantity=f["quantity"],
                                      matched_order_id=f["matched_order_id"],
                                      matched_agent_id=f["matched_agent_id"],
                                      Y_value=y_value)

    def on_fill(self, fill_price: float, fill_quantity: int,
                side: str, current_Y: float):
        if side == "bid":
            self.realized_pnl += fill_quantity * (current_Y - fill_price)
        else:
            self.realized_pnl += fill_quantity * (fill_price - current_Y)
        self.trades_executed += 1


# ---------------------------------------------------------------------------
# Section 9: Investor Agent
# ---------------------------------------------------------------------------

class Investor:

    def __init__(self, params: SimulationParams, book: OrderBook,
                 logger: "EventLogger", rng: np.random.Generator,
                 mm_fill_callback=None):
        self.params = params
        self.book = book
        self.logger = logger
        self.rng = rng
        self._mm_fill_cb = mm_fill_callback
        self.trades_executed: int = 0

    def arrive(self, timestamp: float):
        side = self.rng.choice(["bid", "ask"])
        fills = self.book.submit_market("inv", side,
                                        self.params.investor_order_size, timestamp)
        self.logger.log_event(timestamp, "INVESTOR_ARRIVE", "inv", side=side,
                              book_bid=self.book.get_best_bid(),
                              book_ask=self.book.get_best_ask())
        if fills:
            self.trades_executed += 1
        for f in fills:
            if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                mm_side = "ask" if side == "bid" else "bid"
                self._mm_fill_cb(mm_side, f["price"], f["quantity"], timestamp, None)


# ---------------------------------------------------------------------------
# Section 11: Event Logger
# ---------------------------------------------------------------------------

_LOG_FIELDS = [
    "timestamp", "event_type", "agent_id", "order_id", "side",
    "price", "quantity", "fill_price", "fill_quantity",
    "matched_order_id", "matched_agent_id",
    "book_bid", "book_ask", "Y_value",
    "mm_inventory", "mm_realized_pnl", "mm_unrealized_pnl",
]


class EventLogger:

    def __init__(self, params: SimulationParams):
        self.enabled = params.enable_logging
        self._csv_file = None
        self._writer = None
        self._csv_path: str | None = None
        self._json_path: str | None = None
        self._n_sniper_fills = 0
        self._n_investor_fills = 0
        self._n_mm_fills = 0

        if self.enabled:
            os.makedirs(params.log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_path = os.path.join(params.log_dir, f"events_{stamp}.csv")
            self._json_path = os.path.join(params.log_dir, f"summary_{stamp}.json")
            self._csv_file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=_LOG_FIELDS,
                                          extrasaction="ignore")
            self._writer.writeheader()

    def log_event(self, timestamp, event_type, agent_id, **kwargs):
        if not self.enabled:
            return
        row = {"timestamp": timestamp, "event_type": event_type,
               "agent_id": agent_id}
        row.update(kwargs)
        self._writer.writerow(row)
        if event_type == "FILL":
            if agent_id == "mm":
                self._n_mm_fills += 1
            elif agent_id == "inv":
                self._n_investor_fills += 1
            elif agent_id.startswith("snip"):
                self._n_sniper_fills += 1

    def write_summary(self, result: "SimulationResult", params: SimulationParams):
        if not self.enabled or self._json_path is None:
            return
        import dataclasses
        params_dict = dataclasses.asdict(params)
        params_dict["target_jumps_per_day"] = params.target_jumps_per_day
        params_dict["jump_size_probs"] = list(params.jump_size_probs)
        mm_snap = result.mm_pnl_history[-1] if result.mm_pnl_history else {}
        summary = {
            "params": params_dict,
            "total_trades": (result.sniper1_trades + result.sniper2_trades
                             + result.investor_trades),
            "mm_final_realized_pnl": mm_snap.get("realized_pnl", 0.0),
            "mm_final_unrealized_pnl": mm_snap.get("unrealized_pnl", 0.0),
            "mm_final_total_pnl": mm_snap.get("total_pnl", 0.0),
            "mm_final_inventory": mm_snap.get("inventory", 0.0),
            "sniper1_pnl": result.sniper1_pnl,
            "sniper2_pnl": result.sniper2_pnl,
            "sniper1_trades": result.sniper1_trades,
            "sniper2_trades": result.sniper2_trades,
            "investor_trades": result.investor_trades,
            "avg_spread": result.avg_spread,
            "n_sniper_fills": self._n_sniper_fills,
            "n_investor_fills": self._n_investor_fills,
            "n_mm_fills": self._n_mm_fills,
            "simulation_duration_seconds": params.T,
        }
        with open(self._json_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None


# ---------------------------------------------------------------------------
# Section 10: Event, SimulationResult, SimulationRunner
# ---------------------------------------------------------------------------

@dataclass
class Event:
    time: float
    event_type: str
    agent_id: str
    payload: dict

    def __lt__(self, other: "Event") -> bool:
        return self.time < other.time


@dataclass
class SimulationResult:
    mm_pnl_history: list
    sniper1_pnl: float
    sniper2_pnl: float
    sniper1_trades: int
    sniper2_trades: int
    investor_trades: int
    avg_spread: float
    total_events: int
    params: SimulationParams


_EVENT_PRIORITY = {
    "Y_JUMP": 0, "SNIPER_OBSERVE": 1, "MM_OBSERVE": 2, "INVESTOR_ARRIVE": 3,
}


class SimulationRunner:

    def __init__(self, params: SimulationParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed + 1000)
        self.book = OrderBook()
        self.logger = EventLogger(params)
        self.mm = MarketMaker(params, self.book, self.logger)
        self.snipers: list[Sniper] = [
            Sniper(f"snip{i + 1}", params, self.book, self.logger,
                   mm_fill_callback=self._route_mm_fill)
            for i in range(params.n_snipers)
        ]
        inv_rng = np.random.default_rng(self.rng.integers(0, 2**31))
        self.investor = Investor(params, self.book, self.logger, inv_rng,
                                 mm_fill_callback=self._route_mm_fill)
        self._current_Y: float = params.Y0
        self._sniper_map: dict[str, Sniper] = {s.agent_id: s for s in self.snipers}

    def _route_mm_fill(self, side: str, fill_price: float, fill_quantity: int,
                       timestamp: float, current_Y: float | None):
        self.mm.on_fill(side, fill_price, fill_quantity, timestamp,
                        current_Y if current_Y is not None else self._current_Y)

    def build_event_queue(self, y_times: np.ndarray,
                          y_prices: np.ndarray) -> list[Event]:
        events: list[Event] = []
        for t, y in zip(y_times[1:], y_prices[1:]):
            events.append(Event(float(t), "Y_JUMP", "market", {"y_value": float(y)}))
            for sniper in self.snipers:
                events.append(Event(float(t) + self.params.sniper_lag,
                                    "SNIPER_OBSERVE", sniper.agent_id,
                                    {"y_value": float(y)}))
            events.append(Event(float(t) + self.params.mm_lag,
                                "MM_OBSERVE", "mm", {"y_value": float(y)}))

        arr_rng = np.random.default_rng(self.rng.integers(0, 2**31))
        t = 0.0
        while True:
            dt = arr_rng.exponential(1.0 / self.params.investor_arrival_rate)
            t += dt
            if t > self.params.T:
                break
            events.append(Event(t, "INVESTOR_ARRIVE", "inv", {}))

        events.sort(key=lambda e: (e.time, _EVENT_PRIORITY.get(e.event_type, 99)))

        result: list[Event] = []
        i = 0
        while i < len(events):
            j = i
            while j < len(events) and events[j].time == events[i].time:
                j += 1
            group = events[i:j]
            sn = [e for e in group if e.event_type == "SNIPER_OBSERVE"]
            if len(sn) > 1:
                self.rng.shuffle(sn)
                sn_iter = iter(sn)
                result.extend(
                    next(sn_iter) if e.event_type == "SNIPER_OBSERVE" else e
                    for e in group
                )
            else:
                result.extend(group)
            i = j

        return result

    def run(self) -> SimulationResult:
        y_times, y_prices = simulate_Y(self.params)
        events = self.build_event_queue(y_times, y_prices)
        mm_pnl_history: list[dict] = []
        spread_samples: list[float] = []

        for event in events:
            etype = event.event_type
            if etype == "Y_JUMP":
                self._current_Y = event.payload["y_value"]
                self.mm.mark_to_market(self._current_Y)
                spread = self.book.get_spread()
                if spread is not None:
                    spread_samples.append(spread)
                mm_pnl_history.append({"time": event.time,
                                       **self.mm.get_pnl_snapshot()})
                self.logger.log_event(event.time, "Y_JUMP", "market",
                                      Y_value=self._current_Y,
                                      book_bid=self.book.get_best_bid(),
                                      book_ask=self.book.get_best_ask(),
                                      mm_inventory=self.mm.inventory,
                                      mm_realized_pnl=self.mm.realized_pnl,
                                      mm_unrealized_pnl=self.mm.unrealized_pnl)
            elif etype == "SNIPER_OBSERVE":
                sniper = self._sniper_map.get(event.agent_id)
                if sniper:
                    sniper.observe_Y(event.payload["y_value"], event.time)
            elif etype == "MM_OBSERVE":
                self.mm.observe_Y(event.payload["y_value"], event.time,
                                  self.book.get_midpoint())
            elif etype == "INVESTOR_ARRIVE":
                self.investor.arrive(event.time)

        result = SimulationResult(
            mm_pnl_history=mm_pnl_history,
            sniper1_pnl=self.snipers[0].realized_pnl if self.snipers else 0.0,
            sniper2_pnl=self.snipers[1].realized_pnl if len(self.snipers) > 1 else 0.0,
            sniper1_trades=self.snipers[0].trades_executed if self.snipers else 0,
            sniper2_trades=self.snipers[1].trades_executed if len(self.snipers) > 1 else 0,
            investor_trades=self.investor.trades_executed,
            avg_spread=float(np.mean(spread_samples)) if spread_samples else 0.0,
            total_events=len(events),
            params=self.params,
        )
        self.logger.write_summary(result, self.params)
        self.logger.close()
        return result
