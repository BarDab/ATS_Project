"""Continuous order book engine with price-time priority."""

import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional


class OrderBook:
    """Price-time priority LOB: dict-of-deque per level + heaps for best prices."""

    def __init__(self):
        # {price: deque of orders} for bids/asks
        self.bids: Dict[float, deque] = {}
        self.asks: Dict[float, deque] = {}
        # Min-heap for asks (low prices first), max-heap for bids (negate prices)
        self.bid_heap: List[float] = []
        self.ask_heap: List[float] = []
        # id -> (side, price, qty) for CANCEL lookups
        self.id_index: Dict[int, Tuple[str, float, int]] = {}
        # Logs
        self.trades: List[dict] = []
        self.quotes: List[dict] = []

    def add_order(self, order_id: int, side: str, price: Optional[float], qty: int, order_type: str):
        """Add LIMIT or MARKET order; return fill if any."""
        if order_type == "MARKET":
            # Execute immediately at best available prices
            fills = self._execute_market(order_id, side, qty)
            return fills
        elif order_type == "IOC":
            # Immediate-or-cancel: match what you can, cancel rest
            fills = self._execute_ioc(order_id, side, price, qty)
            return fills
        else:
            # LIMIT: try to match, rest goes to book
            fills = self._match_limit(order_id, side, price, qty)
            return fills

    def cancel_order(self, order_id: int) -> bool:
        """Cancel by id; return True if found and removed."""
        if order_id not in self.id_index:
            return False

        side, price, qty = self.id_index.pop(order_id)
        levels = self.bids if side == "BUY" else self.asks

        if price not in levels:
            return False

        # Remove from deque
        level = levels[price]
        new_level = deque()
        found = False
        for oid, q in level:
            if oid == order_id:
                found = True
            else:
                new_level.append((oid, q))

        if not new_level:
            del levels[price]
            self._rebuild_heaps()
        else:
            levels[price] = new_level

        return found

    def _match_limit(self, order_id: int, side: str, price: float, qty: int) -> List[dict]:
        """Match aggressively, rest rests on book."""
        fills = []
        remaining = qty

        if side == "BUY":
            # Match against asks
            while remaining > 0 and self.ask_heap:
                best_ask = self.ask_heap[0]
                if best_ask > price:
                    break
                level = self.asks[best_ask]
                while level and remaining > 0:
                    ask_id, ask_qty = level[0]
                    trade_qty = min(remaining, ask_qty)
                    fills.append({
                        "buyer_id": order_id,
                        "seller_id": ask_id,
                        "price": best_ask,
                        "qty": trade_qty,
                        "taker_side": "BUY",
                    })
                    remaining -= trade_qty
                    ask_qty -= trade_qty
                    if ask_qty == 0:
                        level.popleft()
                        del self.id_index[ask_id]
                    else:
                        level[0] = (ask_id, ask_qty)
                        self.id_index[ask_id] = ("SELL", best_ask, ask_qty)
                        break

                if not level:
                    del self.asks[best_ask]
                    heapq.heappop(self.ask_heap)

            # Place remainder on book
            if remaining > 0:
                if price not in self.bids:
                    self.bids[price] = deque()
                    heapq.heappush(self.bid_heap, -price)
                self.bids[price].append((order_id, remaining))
                self.id_index[order_id] = ("BUY", price, remaining)

        else:  # SELL
            # Match against bids
            while remaining > 0 and self.bid_heap:
                best_bid = -self.bid_heap[0]
                if best_bid < price:
                    break
                level = self.bids[best_bid]
                while level and remaining > 0:
                    bid_id, bid_qty = level[0]
                    trade_qty = min(remaining, bid_qty)
                    fills.append({
                        "buyer_id": bid_id,
                        "seller_id": order_id,
                        "price": best_bid,
                        "qty": trade_qty,
                        "taker_side": "SELL",
                    })
                    remaining -= trade_qty
                    bid_qty -= trade_qty
                    if bid_qty == 0:
                        level.popleft()
                        del self.id_index[bid_id]
                    else:
                        level[0] = (bid_id, bid_qty)
                        self.id_index[bid_id] = ("BUY", best_bid, bid_qty)
                        break

                if not level:
                    del self.bids[best_bid]
                    heapq.heappop(self.bid_heap)

            # Place remainder on book
            if remaining > 0:
                if price not in self.asks:
                    self.asks[price] = deque()
                    heapq.heappush(self.ask_heap, price)
                self.asks[price].append((order_id, remaining))
                self.id_index[order_id] = ("SELL", price, remaining)

        for fill in fills:
            self.trades.append(fill)

        return fills

    def _execute_market(self, order_id: int, side: str, qty: int) -> List[dict]:
        """Execute MARKET order at best available prices."""
        fills = []
        remaining = qty

        if side == "BUY":
            while remaining > 0 and self.ask_heap:
                best_ask = self.ask_heap[0]
                level = self.asks[best_ask]
                while level and remaining > 0:
                    ask_id, ask_qty = level[0]
                    trade_qty = min(remaining, ask_qty)
                    fills.append({
                        "buyer_id": order_id,
                        "seller_id": ask_id,
                        "price": best_ask,
                        "qty": trade_qty,
                        "taker_side": "BUY",
                    })
                    remaining -= trade_qty
                    ask_qty -= trade_qty
                    if ask_qty == 0:
                        level.popleft()
                        del self.id_index[ask_id]
                    else:
                        level[0] = (ask_id, ask_qty)
                        self.id_index[ask_id] = ("SELL", best_ask, ask_qty)
                        break

                if not level:
                    del self.asks[best_ask]
                    heapq.heappop(self.ask_heap)
        else:  # SELL
            while remaining > 0 and self.bid_heap:
                best_bid = -self.bid_heap[0]
                level = self.bids[best_bid]
                while level and remaining > 0:
                    bid_id, bid_qty = level[0]
                    trade_qty = min(remaining, bid_qty)
                    fills.append({
                        "buyer_id": bid_id,
                        "seller_id": order_id,
                        "price": best_bid,
                        "qty": trade_qty,
                        "taker_side": "SELL",
                    })
                    remaining -= trade_qty
                    bid_qty -= trade_qty
                    if bid_qty == 0:
                        level.popleft()
                        del self.id_index[bid_id]
                    else:
                        level[0] = (bid_id, bid_qty)
                        self.id_index[bid_id] = ("BUY", best_bid, bid_qty)
                        break

                if not level:
                    del self.bids[best_bid]
                    heapq.heappop(self.bid_heap)

        for fill in fills:
            self.trades.append(fill)

        return fills

    def _execute_ioc(self, order_id: int, side: str, price: float, qty: int) -> List[dict]:
        """IOC: match what you can at limit price, cancel rest."""
        fills = []
        remaining = qty

        if side == "BUY":
            while remaining > 0 and self.ask_heap:
                best_ask = self.ask_heap[0]
                if best_ask > price:
                    break
                level = self.asks[best_ask]
                while level and remaining > 0:
                    ask_id, ask_qty = level[0]
                    trade_qty = min(remaining, ask_qty)
                    fills.append({
                        "buyer_id": order_id,
                        "seller_id": ask_id,
                        "price": best_ask,
                        "qty": trade_qty,
                        "taker_side": "BUY",
                    })
                    remaining -= trade_qty
                    ask_qty -= trade_qty
                    if ask_qty == 0:
                        level.popleft()
                        del self.id_index[ask_id]
                    else:
                        level[0] = (ask_id, ask_qty)
                        self.id_index[ask_id] = ("SELL", best_ask, ask_qty)
                        break

                if not level:
                    del self.asks[best_ask]
                    heapq.heappop(self.ask_heap)
        else:  # SELL
            while remaining > 0 and self.bid_heap:
                best_bid = -self.bid_heap[0]
                if best_bid < price:
                    break
                level = self.bids[best_bid]
                while level and remaining > 0:
                    bid_id, bid_qty = level[0]
                    trade_qty = min(remaining, bid_qty)
                    fills.append({
                        "buyer_id": bid_id,
                        "seller_id": order_id,
                        "price": best_bid,
                        "qty": trade_qty,
                        "taker_side": "SELL",
                    })
                    remaining -= trade_qty
                    bid_qty -= trade_qty
                    if bid_qty == 0:
                        level.popleft()
                        del self.id_index[bid_id]
                    else:
                        level[0] = (bid_id, bid_qty)
                        self.id_index[bid_id] = ("BUY", best_bid, bid_qty)
                        break

                if not level:
                    del self.bids[best_bid]
                    heapq.heappop(self.bid_heap)

        for fill in fills:
            self.trades.append(fill)

        return fills

    def _rebuild_heaps(self):
        """Rebuild heaps from scratch (used after cancel)."""
        self.bid_heap = [-p for p in self.bids.keys()]
        self.ask_heap = list(self.asks.keys())
        heapq.heapify(self.bid_heap)
        heapq.heapify(self.ask_heap)

    def snapshot(self) -> dict:
        """Return current best bid/ask for quote logging."""
        best_bid = -self.bid_heap[0] if self.bid_heap else None
        best_ask = self.ask_heap[0] if self.ask_heap else None
        return {"bid": best_bid, "ask": best_ask}

