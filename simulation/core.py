"""Market microstructure simulation — Phase 1, 2 & 3 core."""

from __future__ import annotations
import bisect
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np

from .events import (
    EventLogger,
    BaseEvent, YJumpEvent, SniperObserveEvent, MMObserveEvent,
    InvestorArriveEvent, DeferredLabelEvent,
)


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
    # --- Market Maker (Phase 2) ---
    mm_base_spread_ticks: int = 4
    mm_max_inventory: float = 300.0
    mm_inventory_skew_factor: float = 0.1
    mm_divergence_threshold_ticks: int = 2
    mm_refill_on_fill: bool = True
    # --- Snipers ---
    sniper_lag: float = 0.001
    sniper_order_size: int = 1
    n_snipers: int = 2
    sniper_min_edge_ticks: int = 2 # this is currently used for both labeling and sniper logic, just to keep it simple, but can be separated if needed
    # --- Investors ---
    investor_arrival_rate: float = 1.0
    investor_order_size: int = 1
    # --- Informed Trade Detection (Phase 3) ---
    mm_detection_window: float = 0.1
    mm_window_size: int = 50
    mm_min_fills_for_adjustment: int = 10 # before adjusting alpha for calculating MM spread, we need minimal fills to have somewhat stable estimation
    # --- Spread Adjustment (Phase 3) ---
    mm_alpha_sensitivity: float = 2.0
    mm_spread_floor_ticks: int = 2
    mm_spread_ceiling_ticks: int = 20
    # --- Logging ---
    enable_logging: bool = False
    log_dir: str = "logs"
    run_label: str = ""

    lambda_jump: float = field(init=False)
    T: float = field(init=False)

    def __post_init__(self):
        self.lambda_jump = self.target_jumps_per_day / (self.trading_hours * 3600)
        self.T = self.trading_hours * 3600


# ---------------------------------------------------------------------------
# Section 1: Y Process
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
# Section 1b: Simple Market Maker + Quote Time Series (Phase 1)
# ---------------------------------------------------------------------------

class SimpleMarketMaker:

    def __init__(self, params: SimulationParams):
        self.params = params
        self.bid = None
        self.ask = None

    def observe_Y(self, y_value: float, observed_at_time: float):
        half_spread = self.params.mm_base_spread_ticks * self.params.tick_size / 2
        self.bid = y_value - half_spread
        self.ask = y_value + half_spread

    def get_quotes(self):
        return (self.bid, self.ask)


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
# Section 2: Central Order Book
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
# Section 7: Market Maker Agent (Phase 2 + 3)
# ---------------------------------------------------------------------------

class MarketMaker:

    def __init__(self, params: SimulationParams, book: OrderBook):
        self.params = params
        self.book = book
        self.inventory: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.avg_entry_price: float = 0.0
        self.current_bid_id: str | None = None
        self.current_ask_id: str | None = None
        self._last_bid_price: float | None = None
        self._last_ask_price: float | None = None
        # informed trade tracking (Phase 3)
        self.fill_history: deque = deque(maxlen=params.mm_window_size)
        self.pending_fills: dict = {}
        self.alpha: float = 0.0
        self.current_spread_ticks: float = params.mm_base_spread_ticks
        # PnL decomposition (Phase 3)
        self.spread_income: float = 0.0
        self.adverse_selection_loss: float = 0.0
        # history for plotting (Phase 3)
        self.alpha_history: list[dict] = []
        self.spread_history: list[dict] = []
        self.pnl_decomposition_history: list[dict] = []

    def _get_market_mid_price(self) -> float | None:
        # get current ask/bid from order book or last know bid/ask price
        bid = self.book.get_best_bid() if self.book.bids else self._last_bid_price
        ask = self.book.get_best_ask() if self.book.asks else self._last_ask_price
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None

    def _snap(self, price: float) -> float:
        tick_size = self.params.tick_size
        return round(price / tick_size) * tick_size

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

    def _post_ask(self, mid: float, half: float, ts: float):
        if self.inventory <= -self.params.mm_max_inventory:
            return
        price = self._snap(mid + half)
        self._last_ask_price = price
        self.current_ask_id = self.book.submit_limit("mm", "ask", price, 1, ts)

    def react_to_divergence(self, y_value: float, observed_at_time: float,
                  current_book_mid: float | None):
        ts = self.params.tick_size
        skewed_mid = y_value - self.inventory * self.params.mm_inventory_skew_factor * ts
        normal_half = self.current_spread_ticks * ts / 2
        wide_half = (self.current_spread_ticks * 2) * ts / 2
        div = (abs(current_book_mid - y_value) / ts
               if current_book_mid is not None else 0.0)

        if div > self.params.mm_divergence_threshold_ticks:
            self._cancel_both()
            self._post_bid(skewed_mid, wide_half, observed_at_time)
            self._post_ask(skewed_mid, wide_half, observed_at_time)

        else:
            self._cancel_both()
            self._post_bid(skewed_mid, normal_half, observed_at_time)
            self._post_ask(skewed_mid, normal_half, observed_at_time)

    def on_fill(self, fill_id: str | None, side: str, fill_price: float,
                fill_quantity: int, timestamp: float):
        
        # update inventory
        prev_inv = self.inventory
        delta = fill_quantity if side == "bid" else -fill_quantity
        new_inv = prev_inv + delta
        self.inventory = new_inv
        mid_price = self._get_market_mid_price()

        self._calculate_pnl_and_entry_price(prev_inv, new_inv, fill_price, fill_quantity, side)

        # update unrealized pnl
        
        if mid_price is not None:
            self.unrealized_pnl = (self.inventory * (mid_price - self.avg_entry_price)
                                   if self.inventory != 0 else 0.0)

        self._clear_current_order_id(side)


        if fill_id is not None:
            self.pending_fills[fill_id] = {
                "side": side,
                "fill_price": fill_price,
                "fill_time": timestamp,
                "fill_quantity": fill_quantity,
            }
        
        self._submit_new_order_after_fill(timestamp)

    def _clear_current_order_id(self, side: str):
        if side == "bid":
            self.current_bid_id = None
        else:
            self.current_ask_id = None

    def _calculate_pnl_and_entry_price(self, prev_inv: float, new_inv: float, fill_price: float, fill_quantity: int, side: str):
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

                    
    def _submit_new_order_after_fill(self, timestamp: float):
        if self.params.mm_refill_on_fill:
            if self.current_bid_id is None and self._last_bid_price is not None:
                if self.inventory < self.params.mm_max_inventory:
                    self.current_bid_id = self.book.submit_limit(
                        "mm", "bid", self._last_bid_price, 1, timestamp)
            if self.current_ask_id is None and self._last_ask_price is not None:
                if self.inventory > -self.params.mm_max_inventory:
                    self.current_ask_id = self.book.submit_limit(
                        "mm", "ask", self._last_ask_price, 1, timestamp)

    def mark_to_market(self):
        """Just update unrealized PnL based on current mid price."""
        mid_price = self._get_market_mid_price()
        if mid_price is not None:
            self.unrealized_pnl = (self.inventory * (mid_price - self.avg_entry_price)
                                   if self.inventory != 0 else 0.0)

    def _is_informed(self, fill: dict, Y_now: float) -> bool:
        threshold = self.params.sniper_min_edge_ticks * self.params.tick_size
        if fill["side"] == "ask":
            return Y_now > fill["fill_price"] + threshold
        return Y_now < fill["fill_price"] - threshold

    def _update_alpha_and_spread(self, informed: bool, timestamp: float):
        self.fill_history.append(informed)
        self.alpha = sum(self.fill_history) / len(self.fill_history)
        self.alpha_history.append({"time": timestamp, "alpha": self.alpha})
        self.current_spread_ticks = self._compute_spread()
        self.spread_history.append({"time": timestamp, "spread_ticks": self.current_spread_ticks})

    def _classify_pnl(self, fill: dict, informed: bool, Y_now: float, timestamp: float):
        half_spread = fill["fill_quantity"] * self.current_spread_ticks * self.params.tick_size / 2
        if informed:
            if fill["side"] == "ask":
                loss = fill["fill_quantity"] * (Y_now - fill["fill_price"])
            else:
                loss = fill["fill_quantity"] * (fill["fill_price"] - Y_now)
            self.adverse_selection_loss -= loss
        else:
            self.spread_income += half_spread
        self.pnl_decomposition_history.append({
            "time": timestamp,
            "spread_income": self.spread_income,
            "adverse_selection_loss": self.adverse_selection_loss,
            "total_attributed_pnl": self.spread_income + self.adverse_selection_loss,
        })

    def get_information_from_processed_trade(self, fill_id: str, Y_now: float, timestamp: float):
        """"This is updating information which are available to MM after certain delay. Returning True if trade classified as informed else False"""
        if fill_id not in self.pending_fills:
            return
        fill = self.pending_fills.pop(fill_id)
        informed = self._is_informed(fill, Y_now)
        self._update_alpha_and_spread(informed, timestamp)
        self._classify_pnl(fill, informed, Y_now, timestamp)
        return informed

    # TOOD verify if this can be be adjusted to Glosten-Milgrom
    def _compute_spread(self) -> float:
        """"Ideally this method should compute spread based on Glosten-Milgrom model"""

        # when there is not enough data to have stable alpha estimation, use base spread, otherwise adjust based on alpha and clamp to floor/ceiling
        if len(self.fill_history) < self.params.mm_min_fills_for_adjustment:
            return self.params.mm_base_spread_ticks
        adjusted = self.params.mm_base_spread_ticks * (1 + self.params.mm_alpha_sensitivity * self.alpha)
        adjusted = max(self.params.mm_spread_floor_ticks,
                       min(self.params.mm_spread_ceiling_ticks, adjusted))
        return round(adjusted)

    def get_pnl_snapshot(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "inventory": self.inventory,
            "avg_entry_price": self.avg_entry_price,
            "alpha": self.alpha,
            "current_spread_ticks": self.current_spread_ticks,
            "spread_income": self.spread_income,
            "adverse_selection_loss": self.adverse_selection_loss,
            "total_attributed_pnl": self.spread_income + self.adverse_selection_loss,
            "n_fills_labeled": len(self.fill_history),
            "n_pending_fills": len(self.pending_fills),
        }


# ---------------------------------------------------------------------------
# Section 8: Sniper Agent
# ---------------------------------------------------------------------------

class Sniper:

    def __init__(self, agent_id: str, params: SimulationParams,
                 book: OrderBook, mm_fill_callback=None):
        self.agent_id = agent_id
        self.params = params
        self.book = book
        self._mm_fill_cb = mm_fill_callback
        self.realized_pnl: float = 0.0
        self.trades_executed: int = 0

    def snipe(self, y_value: float, observed_at_time: float):
        tick_size = self.params.tick_size
        edge = self.params.sniper_min_edge_ticks * tick_size

        best_ask = self.book.get_best_ask()
        if best_ask is not None and (y_value - best_ask) > edge:
            for f in self.book.submit_market(self.agent_id, "bid",
                                             self.params.sniper_order_size,
                                             observed_at_time):
                self.on_fill(f["price"], f["quantity"], "bid", y_value)
                if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                    self._mm_fill_cb("ask", f["price"], f["quantity"],
                                     observed_at_time, y_value,
                                     f["matched_order_id"], self.agent_id)

        best_bid = self.book.get_best_bid()
        if best_bid is not None and (best_bid - y_value) > edge:
            for f in self.book.submit_market(self.agent_id, "ask",
                                             self.params.sniper_order_size,
                                             observed_at_time):
                self.on_fill(f["price"], f["quantity"], "ask", y_value)
                if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                    self._mm_fill_cb("bid", f["price"], f["quantity"],
                                     observed_at_time, y_value,
                                     f["matched_order_id"], self.agent_id)

    def on_fill(self, fill_price: float, fill_quantity: int,
                side: str, current_Y: float):
        # Not real PNL just assesment, TODO remove this part
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
                 rng: np.random.Generator, mm_fill_callback=None):
        self.params = params
        self.book = book
        self.rng = rng
        self._mm_fill_cb = mm_fill_callback
        self.trades_executed: int = 0

    def arrive(self, timestamp: float):
        side = self.rng.choice(["bid", "ask"])
        fills = self.book.submit_market("inv", side,
                                        self.params.investor_order_size, timestamp)
        if fills:
            self.trades_executed += 1
        for f in fills:
            if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                mm_side = "ask" if side == "bid" else "bid"
                self._mm_fill_cb(mm_side, f["price"], f["quantity"], timestamp,
                                 None, f["matched_order_id"], "inv")


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
    mm_alpha_history: list[dict]
    mm_spread_history: list[dict]
    mm_pnl_decomposition_history: list[dict]
    mm_final_alpha: float
    mm_final_spread_ticks: float
    spread_samples: list[float]


class SimulationRunner:

    def __init__(self, params: SimulationParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed + 1000)
        self.book = OrderBook()
        self.logger = EventLogger(params)
        self.mm = MarketMaker(params, self.book)
        self.snipers: list[Sniper] = [
            Sniper(f"snip{i + 1}", params, self.book,
                   mm_fill_callback=self._route_mm_fill)
            for i in range(params.n_snipers)
        ]
        inv_rng = np.random.default_rng(self.rng.integers(0, 2**31))
        self.investor = Investor(params, self.book, inv_rng,
                                 mm_fill_callback=self._route_mm_fill)
        self._current_Y: float = params.Y0
        self._sniper_map: dict[str, Sniper] = {s.agent_id: s for s in self.snipers}
        self._event_queue: list | None = None
        self._seq_counter: int = 0

    def _mm_state(self) -> dict:
        bid = self.book.get_best_bid()
        ask = self.book.get_best_ask()
        spread = round(ask - bid, 4) if bid is not None and ask is not None else None
        return {
            "book_bid": bid,
            "book_ask": ask,
            "book_spread": spread,
            "mm_inventory": self.mm.inventory,
            "mm_realized_pnl": self.mm.realized_pnl,
            "mm_unrealized_pnl": self.mm.unrealized_pnl,
            "alpha": self.mm.alpha,
            "current_spread_ticks": self.mm.current_spread_ticks,
            "spread_income": self.mm.spread_income,
            "adverse_selection_loss": self.mm.adverse_selection_loss,
        }

    def _route_mm_fill(self, side: str, fill_price: float, fill_quantity: int,
                       timestamp: float, current_Y: float | None,
                       fill_id: str | None = None, taker_agent_id: str | None = None):
        self.mm.on_fill(fill_id, side, fill_price, fill_quantity, timestamp)
        self.logger.log_event(timestamp, "MM_FILL", "mm",
                              order_id=fill_id, side=side,
                              fill_price=fill_price, fill_quantity=fill_quantity,
                              taker_agent_id=taker_agent_id,
                              Y_value=self._current_Y,
                              **self._mm_state())
        if fill_id is not None and self._event_queue is not None:
            label_time = timestamp + self.params.mm_detection_window
            if label_time <= self.params.T:
                seq = self._seq_counter
                self._seq_counter += 1
                heapq.heappush(self._event_queue,
                               DeferredLabelEvent(time=label_time, fill_id=fill_id, _seq=seq))

    def build_event_queue(self, y_times: np.ndarray,
                          y_prices: np.ndarray) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        for t, y in zip(y_times[1:], y_prices[1:]):
            events.append(YJumpEvent(time=float(t), y_value=float(y)))
            for sniper in self.snipers:
                events.append(SniperObserveEvent(
                    time=float(t) + self.params.sniper_lag,
                    agent_id=sniper.agent_id,
                    y_value=float(y)))
            events.append(MMObserveEvent(time=float(t) + self.params.mm_lag, y_value=float(y)))

        arr_rng = np.random.default_rng(self.rng.integers(0, 2**31))
        t = 0.0
        while True:
            dt = arr_rng.exponential(1.0 / self.params.investor_arrival_rate)
            t += dt
            if t > self.params.T:
                break
            events.append(InvestorArriveEvent(time=t))

        events.sort(key=lambda e: (e.time, e.PRIORITY))

        result: list[BaseEvent] = []
        i = 0
        while i < len(events):
            j = i
            while j < len(events) and events[j].time == events[i].time:
                j += 1
            group = events[i:j]
            sn = [e for e in group if isinstance(e, SniperObserveEvent)]
            if len(sn) > 1:
                self.rng.shuffle(sn)
                sn_iter = iter(sn)
                result.extend(
                    next(sn_iter) if isinstance(e, SniperObserveEvent) else e
                    for e in group
                )
            else:
                result.extend(group)
            i = j

        for i, e in enumerate(result):
            e._seq = i
        self._seq_counter = len(result)
        heapq.heapify(result)
        return result

    def run(self) -> SimulationResult:
        y_times, y_prices = simulate_Y(self.params)
        self._event_queue = self.build_event_queue(y_times, y_prices)
        mm_pnl_history: list[dict] = []
        spread_samples: list[float] = []
        total_event_count = 0

        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            total_event_count += 1
            
            match event:
                # jumpt event -> 
                case YJumpEvent(y_value=y):
                    self._current_Y = y
                    self.mm.mark_to_market()
                    spread = self.book.get_spread()
                    if spread is not None:
                        spread_samples.append(spread)
                    mm_pnl_history.append({"time": event.time,
                                           **self.mm.get_pnl_snapshot()})
                    self.logger.log_event(event.time, "Y_JUMP", "market",
                                          Y_value=self._current_Y,
                                          **self._mm_state())
                case SniperObserveEvent(agent_id=aid, y_value=y):
                    sniper = self._sniper_map.get(aid)
                    if sniper:
                        sniper.snipe(y, event.time)
                        self.logger.log_event(event.time, "SNIPER_OBSERVE", aid,
                                              Y_value=y, **self._mm_state())
                case MMObserveEvent(y_value=y):
                    self.mm.react_to_divergence(y, event.time, self.book.get_midpoint())
                    self.logger.log_event(event.time, "MM_OBSERVE", "mm",
                                          Y_value=y, **self._mm_state())
                case InvestorArriveEvent():
                    self.investor.arrive(event.time)
                    self.logger.log_event(event.time, "INVESTOR_ARRIVE", "inv",
                                          **self._mm_state())
                case DeferredLabelEvent(fill_id=fid):
                    informed = self.mm.get_information_from_processed_trade(
                        fill_id=fid, Y_now=self._current_Y, timestamp=event.time)
                    if informed is not None:
                        self.logger.log_event(event.time, "DEFERRED_LABEL", "mm",
                                              order_id=fid,
                                              Y_value=self._current_Y,
                                              informed=informed,
                                              **self._mm_state())

        result = SimulationResult(
            mm_pnl_history=mm_pnl_history,
            sniper1_pnl=self.snipers[0].realized_pnl if self.snipers else 0.0,
            sniper2_pnl=self.snipers[1].realized_pnl if len(self.snipers) > 1 else 0.0,
            sniper1_trades=self.snipers[0].trades_executed if self.snipers else 0,
            sniper2_trades=self.snipers[1].trades_executed if len(self.snipers) > 1 else 0,
            investor_trades=self.investor.trades_executed,
            avg_spread=float(np.mean(spread_samples)) if spread_samples else 0.0,
            total_events=total_event_count,
            params=self.params,
            mm_alpha_history=self.mm.alpha_history,
            mm_spread_history=self.mm.spread_history,
            mm_pnl_decomposition_history=self.mm.pnl_decomposition_history,
            mm_final_alpha=self.mm.alpha,
            mm_final_spread_ticks=self.mm.current_spread_ticks,
            spread_samples=spread_samples,
        )
        self.logger.write_summary(result, self.params)
        self.logger.close()
        return result
