"""Agent implementations: MarketMaker, Sniper, Investor."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .events import DeferredLimitOrderEvent, DeferredMarketOrderEvent

if TYPE_CHECKING:
    from .core import SimulationParams, OrderBook


class MarketMaker:

    def __init__(self, params: SimulationParams, book: OrderBook, schedule_event=None):
        self.params = params
        self.book = book
        self._schedule_event = schedule_event
        self.inventory: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.avg_entry_price: float = 0.0
        self.current_bid_id: str | None = None
        self.current_ask_id: str | None = None
        self._last_bid_price: float | None = None
        self._last_ask_price: float | None = None
        self._pending_bid_event: DeferredLimitOrderEvent | None = None
        self._pending_ask_event: DeferredLimitOrderEvent | None = None
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
        if self._pending_bid_event is not None:
            self._pending_bid_event.cancelled = True
            self._pending_bid_event = None

    def _cancel_ask(self):
        if self.current_ask_id:
            self.book.cancel_order(self.current_ask_id)
            self.current_ask_id = None
        if self._pending_ask_event is not None:
            self._pending_ask_event.cancelled = True
            self._pending_ask_event = None

    def _cancel_both(self):
        self._cancel_bid()
        self._cancel_ask()

    def _post_bid(self, mid: float, half: float, ts: float):
        if self.inventory >= self.params.mm_max_inventory:
            return
        price = self._snap(mid - half)
        self._last_bid_price = price
        delay = self.params.order_submission_delay
        if delay == 0.0 or self._schedule_event is None:
            self.current_bid_id = self.book.submit_limit("mm", "bid", price, 1, ts)
        else:
            if self._pending_bid_event is not None:
                self._pending_bid_event.cancelled = True
            def _reg(oid):
                self.current_bid_id = oid
                self._pending_bid_event = None
            ev = DeferredLimitOrderEvent(
                time=ts + delay, agent_id="mm", side="bid",
                price=price, quantity=1, register_id_callback=_reg)
            self._pending_bid_event = ev
            self._schedule_event(ev)

    def _post_ask(self, mid: float, half: float, ts: float):
        if self.inventory <= -self.params.mm_max_inventory:
            return
        price = self._snap(mid + half)
        self._last_ask_price = price
        delay = self.params.order_submission_delay
        if delay == 0.0 or self._schedule_event is None:
            self.current_ask_id = self.book.submit_limit("mm", "ask", price, 1, ts)
        else:
            if self._pending_ask_event is not None:
                self._pending_ask_event.cancelled = True
            def _reg(oid):
                self.current_ask_id = oid
                self._pending_ask_event = None
            ev = DeferredLimitOrderEvent(
                time=ts + delay, agent_id="mm", side="ask",
                price=price, quantity=1, register_id_callback=_reg)
            self._pending_ask_event = ev
            self._schedule_event(ev)

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
        if not self.params.mm_refill_on_fill:
            return
        delay = self.params.order_submission_delay
        if self.current_bid_id is None and self._pending_bid_event is None \
                and self._last_bid_price is not None:
            if self.inventory < self.params.mm_max_inventory:
                if delay == 0.0 or self._schedule_event is None:
                    self.current_bid_id = self.book.submit_limit(
                        "mm", "bid", self._last_bid_price, 1, timestamp)
                else:
                    price = self._last_bid_price
                    def _reg(oid):
                        self.current_bid_id = oid
                        self._pending_bid_event = None
                    ev = DeferredLimitOrderEvent(
                        time=timestamp + delay, agent_id="mm", side="bid",
                        price=price, quantity=1, register_id_callback=_reg)
                    self._pending_bid_event = ev
                    self._schedule_event(ev)
        if self.current_ask_id is None and self._pending_ask_event is None \
                and self._last_ask_price is not None:
            if self.inventory > -self.params.mm_max_inventory:
                if delay == 0.0 or self._schedule_event is None:
                    self.current_ask_id = self.book.submit_limit(
                        "mm", "ask", self._last_ask_price, 1, timestamp)
                else:
                    price = self._last_ask_price
                    def _reg(oid):
                        self.current_ask_id = oid
                        self._pending_ask_event = None
                    ev = DeferredLimitOrderEvent(
                        time=timestamp + delay, agent_id="mm", side="ask",
                        price=price, quantity=1, register_id_callback=_reg)
                    self._pending_ask_event = ev
                    self._schedule_event(ev)

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



class Sniper:

    def __init__(self, agent_id: str, params: SimulationParams,
                 book: OrderBook, mm_fill_callback=None, schedule_event=None):
        self.agent_id = agent_id
        self.params = params
        self.book = book
        self._mm_fill_cb = mm_fill_callback
        self._schedule_event = schedule_event
        self.realized_pnl: float = 0.0
        self.trades_executed: int = 0
        self.trades_attempted: int = 0

    def snipe(self, y_value: float, observed_at_time: float):
        tick_size = self.params.tick_size
        edge = self.params.sniper_min_edge_ticks * tick_size
        delay = self.params.order_submission_delay

        best_ask = self.book.get_best_ask()
        if best_ask is not None and (y_value - best_ask) > edge:
            self.trades_attempted += 1
            if delay == 0.0 or self._schedule_event is None:
                for f in self.book.submit_market(self.agent_id, "bid",
                                                 self.params.sniper_order_size,
                                                 observed_at_time):
                    self.on_fill(f["price"], f["quantity"], "bid", y_value)
                    if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                        self._mm_fill_cb("ask", f["price"], f["quantity"],
                                         observed_at_time, y_value,
                                         f["matched_order_id"], self.agent_id)
            else:
                y_snap = y_value
                cb = self._mm_fill_cb; ref = self; a_id = self.agent_id
                def _handler_bid(fills, fire_time, _y=y_snap, _cb=cb, _ref=ref, _aid=a_id):
                    for f in fills:
                        _ref.on_fill(f["price"], f["quantity"], "bid", _y)
                        if _cb and f["matched_agent_id"] == "mm":
                            _cb("ask", f["price"], f["quantity"],
                                fire_time, _y, f["matched_order_id"], _aid)
                self._schedule_event(DeferredMarketOrderEvent(
                    time=observed_at_time + delay, agent_id=self.agent_id,
                    side="bid", quantity=self.params.sniper_order_size,
                    post_fill_handler=_handler_bid))

        best_bid = self.book.get_best_bid()
        if best_bid is not None and (best_bid - y_value) > edge:
            self.trades_attempted += 1
            if delay == 0.0 or self._schedule_event is None:
                for f in self.book.submit_market(self.agent_id, "ask",
                                                 self.params.sniper_order_size,
                                                 observed_at_time):
                    self.on_fill(f["price"], f["quantity"], "ask", y_value)
                    if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                        self._mm_fill_cb("bid", f["price"], f["quantity"],
                                         observed_at_time, y_value,
                                         f["matched_order_id"], self.agent_id)
            else:
                y_snap = y_value
                cb = self._mm_fill_cb; ref = self; a_id = self.agent_id
                def _handler_ask(fills, fire_time, _y=y_snap, _cb=cb, _ref=ref, _aid=a_id):
                    for f in fills:
                        _ref.on_fill(f["price"], f["quantity"], "ask", _y)
                        if _cb and f["matched_agent_id"] == "mm":
                            _cb("bid", f["price"], f["quantity"],
                                fire_time, _y, f["matched_order_id"], _aid)
                self._schedule_event(DeferredMarketOrderEvent(
                    time=observed_at_time + delay, agent_id=self.agent_id,
                    side="ask", quantity=self.params.sniper_order_size,
                    post_fill_handler=_handler_ask))

    def on_fill(self, fill_price: float, fill_quantity: int,
                side: str, current_Y: float):
        # Not real PNL just assesment, TODO remove this part
        if side == "bid":
            self.realized_pnl += fill_quantity * (current_Y - fill_price)
        else:
            self.realized_pnl += fill_quantity * (fill_price - current_Y)
        self.trades_executed += 1


class Investor:

    def __init__(self, params: SimulationParams, book: OrderBook,
                 rng: np.random.Generator, mm_fill_callback=None, schedule_event=None):
        self.params = params
        self.book = book
        self.rng = rng
        self._mm_fill_cb = mm_fill_callback
        self._schedule_event = schedule_event
        self.trades_executed: int = 0

    def arrive(self, timestamp: float):
        side = self.rng.choice(["bid", "ask"])
        delay = self.params.order_submission_delay
        if delay == 0.0 or self._schedule_event is None:
            fills = self.book.submit_market("inv", side,
                                            self.params.investor_order_size, timestamp)
            if fills:
                self.trades_executed += 1
            for f in fills:
                if self._mm_fill_cb and f["matched_agent_id"] == "mm":
                    mm_side = "ask" if side == "bid" else "bid"
                    self._mm_fill_cb(mm_side, f["price"], f["quantity"], timestamp,
                                     None, f["matched_order_id"], "inv")
        else:
            mm_side = "ask" if side == "bid" else "bid"
            cb = self._mm_fill_cb; ref = self
            def _handler(fills, fire_time, _mm_side=mm_side, _cb=cb, _ref=ref):
                if fills:
                    _ref.trades_executed += 1
                for f in fills:
                    if _cb and f["matched_agent_id"] == "mm":
                        _cb(_mm_side, f["price"], f["quantity"], fire_time,
                            None, f["matched_order_id"], "inv")
            self._schedule_event(DeferredMarketOrderEvent(
                time=timestamp + delay, agent_id="inv", side=side,
                quantity=self.params.investor_order_size,
                post_fill_handler=_handler))