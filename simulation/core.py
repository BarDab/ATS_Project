"""Market microstructure simulation"""

from __future__ import annotations
import bisect
import heapq
from dataclasses import dataclass, field
import numpy as np

from .events import (
    EventLogger,
    BaseEvent, YJumpEvent, SniperObserveEvent, MMObserveEvent,
    InvestorArriveEvent, DeferredLabelEvent,
    DeferredLimitOrderEvent, DeferredMarketOrderEvent,
)
from .agents import MarketMaker, Sniper, Investor


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
    mm_max_inventory: float = 100.0
    mm_inventory_skew_factor: float = 0.1
    mm_divergence_threshold_ticks: int = 2
    mm_refill_on_fill: bool = True
    # --- Snipers ---
    sniper_lag: float = 0.001
    order_submission_delay: float = 0.02
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
    # --- Event ordering ---
    random_queue_ordering: bool = True
    # --- Logging ---
    enable_logging: bool = False
    log_dir: str = "logs"
    run_label: str = ""

    lambda_jump: float = field(init=False)
    T: float = field(init=False)

    def __post_init__(self):
        self.lambda_jump = self.target_jumps_per_day / (self.trading_hours * 3600)
        self.T = self.trading_hours * 3600



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
        self.mm = MarketMaker(params, self.book, schedule_event=self._push_event)
        self.snipers: list[Sniper] = [
            Sniper(f"snip{i + 1}", params, self.book,
                   mm_fill_callback=self._route_mm_fill,
                   schedule_event=self._push_event)
            for i in range(params.n_snipers)
        ]
        inv_rng = np.random.default_rng(self.rng.integers(0, 2**31))
        self.investor = Investor(params, self.book, inv_rng,
                                 mm_fill_callback=self._route_mm_fill,
                                 schedule_event=self._push_event)
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

    def _push_event(self, event: BaseEvent) -> None:
        if self._event_queue is None:
            return
        if self.params.random_queue_ordering:
            event._seq = int(self.rng.integers(0, 2**62))
        else:
            event._seq = self._seq_counter
            self._seq_counter += 1
        heapq.heappush(self._event_queue, event)

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
        if fill_id is not None:
            label_time = timestamp + self.params.mm_detection_window
            if label_time <= self.params.T:
                self._push_event(DeferredLabelEvent(time=label_time, fill_id=fill_id))

    def build_event_queue(self, y_times: np.ndarray,
                          y_prices: np.ndarray) -> list[BaseEvent]:
        """Generate all events (Y jumps, sniper observes, MM observes, investor arrivals) and sort by time. Using heap for efficient event processing."""
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

        if self.params.random_queue_ordering:
            events.sort(key=lambda e: (e.time, e.PRIORITY))
            for e in events:
                e._seq = int(self.rng.integers(0, 2**62))
        else:
            events.sort(key=lambda e: (
                e.time,
                e.PRIORITY,
                int(self.rng.integers(0, 2**31)) if isinstance(e, SniperObserveEvent) else 0,
            ))
            for i, e in enumerate(events):
                e._seq = i
        self._seq_counter = len(events)
        heapq.heapify(events) #sorts events, it guarantees earliest that event is first
        return events

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
                case DeferredLimitOrderEvent(agent_id=aid, side=s, price=p, quantity=q):
                    if not event.cancelled:
                        oid = self.book.submit_limit(aid, s, p, q, event.time)
                        if event.register_id_callback is not None:
                            event.register_id_callback(oid)
                case DeferredMarketOrderEvent(agent_id=aid, side=s, quantity=q):
                    fills = self.book.submit_market(aid, s, q, event.time)
                    if event.post_fill_handler is not None:
                        event.post_fill_handler(fills, event.time)

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
