"""Batch auction: uniform clearing price that maximizes volume."""

from typing import List, Dict, Optional, Tuple
from collections import deque


def clear_batch(orders: List[dict], pre_mid: Optional[float] = None, tick: float = 0.01) -> Tuple[Optional[float], List[dict]]:
    """
    Find uniform clearing price, allocate fills in FIFO order.

    Args:
        orders: List of dicts with 'order_id', 'side', 'price', 'qty', 'type', 'timestamp'.
                MARKET orders treated as limit at extreme prices.
        pre_mid: Pre-auction mid for tie-breaking (optional).
        tick: Tick size for rounding midpoint (default 0.01).

    Returns:
        (clearing_price, fills) where fills is list of {buyer_id, seller_id, price, qty, taker_side}.
    """
    # Separate bids and asks
    bids = []
    asks = []

    for o in orders:
        if o["type"] == "CANCEL":
            continue  # Ignore cancels in batch
        if o["type"] == "IOC":
            # IOC in batch: treat as limit
            pass

        side = o["side"]
        price = o.get("price")
        qty = o["qty"]
        order_id = o["order_id"]
        timestamp = o.get("timestamp", 0)

        if o["type"] == "MARKET":
            # MARKET buy: willing to pay infinity; MARKET sell: willing to accept 0
            price = 1e9 if side == "BUY" else 0.0

        if side == "BUY":
            bids.append({"order_id": order_id, "price": price, "qty": qty, "timestamp": timestamp})
        else:
            asks.append({"order_id": order_id, "price": price, "qty": qty, "timestamp": timestamp})

    # Sort bids descending by price, then by timestamp, then by order_id (price-time priority)
    bids.sort(key=lambda x: (-x["price"], x["timestamp"], x["order_id"]))
    asks.sort(key=lambda x: (x["price"], x["timestamp"], x["order_id"]))

    # Build aggregate levels
    bid_levels = {}
    ask_levels = {}
    for b in bids:
        bid_levels[b["price"]] = bid_levels.get(b["price"], 0) + b["qty"]
    for a in asks:
        ask_levels[a["price"]] = ask_levels.get(a["price"], 0) + a["qty"]

    # Find all candidate prices
    candidates = sorted(set(bid_levels.keys()) | set(ask_levels.keys()))
    if not candidates:
        return None, []

    # For each candidate, compute cumulative demand/supply
    best_volume = 0
    winners = []

    for p in candidates:
        demand = sum(q for px, q in bid_levels.items() if px >= p)
        supply = sum(q for px, q in ask_levels.items() if px <= p)
        volume = min(demand, supply)

        if volume > best_volume:
            best_volume = volume
            winners = [p]
        elif volume == best_volume and volume > 0:
            winners.append(p)

    if best_volume == 0:
        return None, []

    # Tie-break: closest to pre_mid (if tied, pick lowest); if no pre_mid, use midpoint rounded to tick
    if len(winners) == 1:
        clearing_price = winners[0]
    else:
        if pre_mid is not None:
            # Find minimum distance to pre_mid
            min_dist = min(abs(p - pre_mid) for p in winners)
            # Among those with minimum distance, pick the lowest price
            closest = [p for p in winners if abs(p - pre_mid) == min_dist]
            clearing_price = min(closest)
        else:
            # No pre_mid: use midpoint of tie band rounded to tick
            midpoint = (winners[0] + winners[-1]) / 2.0
            clearing_price = round(midpoint / tick) * tick

    # Allocate fills at clearing_price, FIFO
    fills = _allocate_fills(bids, asks, clearing_price, best_volume)

    return clearing_price, fills


def _allocate_fills(bids: List[dict], asks: List[dict], price: float, target_vol: int) -> List[dict]:
    """Allocate fills at uniform price, FIFO within each side."""
    # Filter: only bids >= price and asks <= price
    valid_bids = [b for b in bids if b["price"] >= price]
    valid_asks = [a for a in asks if a["price"] <= price]

    fills = []
    traded = 0
    bid_idx = 0
    ask_idx = 0

    # Convert to mutable for partial fills
    bid_rem = [b["qty"] for b in valid_bids]
    ask_rem = [a["qty"] for a in valid_asks]

    while traded < target_vol and bid_idx < len(valid_bids) and ask_idx < len(valid_asks):
        if bid_rem[bid_idx] == 0:
            bid_idx += 1
            continue
        if ask_rem[ask_idx] == 0:
            ask_idx += 1
            continue

        qty = min(bid_rem[bid_idx], ask_rem[ask_idx], target_vol - traded)
        fills.append({
            "buyer_id": valid_bids[bid_idx]["order_id"],
            "seller_id": valid_asks[ask_idx]["order_id"],
            "price": price,
            "qty": qty,
            "taker_side": "BUY",  # Batch convention: consistent taker_side
        })
        bid_rem[bid_idx] -= qty
        ask_rem[ask_idx] -= qty
        traded += qty

    return fills

