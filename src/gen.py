"""Deterministic order generator with seeded RNG."""

import random
import csv
import sys
from typing import TextIO


def generate_orders(
    n: int,
    seed: int,
    auction_interval_ms: int,
    cross_rate: float,
    tick_size: float = 0.01,
    output: TextIO = sys.stdout,
):
    """
    Generate deterministic CSV of orders.

    Args:
        n: Number of orders to generate.
        seed: Random seed for reproducibility.
        auction_interval_ms: Batch auction interval (affects timestamps).
        cross_rate: Fraction of orders that should cross spread (0.0-1.0).
        tick_size: Price tick increment.
        output: File-like object to write CSV.
    """
    rng = random.Random(seed)

    # Start with a drifting mid price
    mid = 100.0
    spread_ticks = 5
    timestamp = 0

    writer = csv.writer(output)
    writer.writerow(["timestamp", "order_id", "type", "side", "price", "qty"])

    live_ids = []
    next_id = 1

    for i in range(n):
        # Drift mid slightly
        if rng.random() < 0.1:
            mid += rng.choice([-1, 1]) * tick_size * rng.randint(1, 3)
            mid = max(mid, 50.0)  # Floor at 50

        # Decide order type
        type_roll = rng.random()
        if type_roll < 0.05 and live_ids:
            # CANCEL
            cancel_id = rng.choice(live_ids)
            live_ids.remove(cancel_id)
            writer.writerow([timestamp, next_id, "CANCEL", "", cancel_id, ""])
            next_id += 1
        else:
            # Decide LIMIT/MARKET/IOC
            if type_roll < 0.80:
                order_type = "LIMIT"
            elif type_roll < 0.95:
                order_type = "IOC"
            else:
                order_type = "MARKET"

            # Decide side
            side = "BUY" if rng.random() < 0.5 else "SELL"

            # Decide price
            if order_type == "MARKET":
                price = ""
            else:
                # Decide if crossing
                if rng.random() < cross_rate:
                    # Aggressive: cross the spread
                    if side == "BUY":
                        price = mid + tick_size * rng.randint(0, spread_ticks)
                    else:
                        price = mid - tick_size * rng.randint(0, spread_ticks)
                else:
                    # Passive: inside spread
                    if side == "BUY":
                        price = mid - tick_size * rng.randint(1, spread_ticks * 2)
                    else:
                        price = mid + tick_size * rng.randint(1, spread_ticks * 2)

                price = round(price / tick_size) * tick_size
                price = max(price, tick_size)  # Floor at tick_size

            # Quantity
            qty = rng.randint(1, 100)

            if order_type != "MARKET":
                writer.writerow([timestamp, next_id, order_type, side, f"{price:.2f}", qty])
            else:
                writer.writerow([timestamp, next_id, order_type, side, "", qty])

            live_ids.append(next_id)
            next_id += 1

        # Increment timestamp (nanoseconds)
        timestamp += rng.randint(100, 10000)

    output.flush()

