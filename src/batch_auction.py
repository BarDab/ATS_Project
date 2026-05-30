"""Batch Auction Simulator: simple uniform-price auction CLI."""


def prompt_float(prompt, min_val=None):
    """Prompt for a float, enforcing minimum if provided."""
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if min_val is not None and value < min_val:
            print(f"Value must be >= {min_val}.")
            continue

        return value


def prompt_int(prompt, min_val=None):
    """Prompt for an int, enforcing minimum if provided."""
    while True:
        raw = input(f"{prompt}: ").strip()
        if not raw.isdigit() and not (raw.startswith("-") and raw[1:].isdigit()):
            print("Please enter a whole number.")
            continue
        value = int(raw)

        if min_val is not None and value < min_val:
            print(f"Value must be >= {min_val}.")
            continue

        return value


def collect_orders():
    """Collect buy/sell orders interactively."""
    orders = []
    next_id = 1

    print("\nEnter orders. Press Enter at 'side' to finish.")

    while True:
        side = input("side [buy/sell] (Enter to stop): ").strip().lower()
        if side == "":
            break
        if side not in {"buy", "sell"}:
            print("Side must be 'buy' or 'sell'.")
            continue

        price = prompt_float("price", min_val=0.0)
        quantity = prompt_int("quantity", min_val=1)

        orders.append({"id": next_id, "side": side, "px": float(price), "qty": int(quantity)})
        next_id += 1

    return orders


def aggregate_levels(orders):
    """Aggregate quantities per price level for bids and asks."""
    bids_levels = {}
    asks_levels = {}

    for order in orders:
        levels = bids_levels if order["side"] == "buy" else asks_levels
        levels[order["px"]] = levels.get(order["px"], 0) + order["qty"]

    return bids_levels, asks_levels


def cumulative_at_price(bids_levels, asks_levels, p):
    """Cumulative demand and supply measured at price p."""
    demand = sum(qty for px, qty in bids_levels.items() if px >= p)
    supply = sum(qty for px, qty in asks_levels.items() if px <= p)
    return demand, supply


def find_clearing_price(bids_levels, asks_levels):
    """Find clearing price that maximizes executed volume."""
    candidates = sorted({*bids_levels.keys(), *asks_levels.keys()})
    if not candidates:
        return None, 0, None, None

    best_volume = 0
    price_stats = []

    for price in candidates:
        demand, supply = cumulative_at_price(bids_levels, asks_levels, price)
        volume = min(demand, supply)
        price_stats.append((price, demand, supply, volume))

        if volume > best_volume:
            best_volume = volume

    if best_volume <= 0:
        return None, 0, None, None

    winning_prices = sorted(px for px, _, _, vol in price_stats if vol == best_volume)

    if len(winning_prices) == 1:
        price = winning_prices[0]
        return price, best_volume, price, price

    p_low = winning_prices[0]
    p_high = winning_prices[-1]
    p_star = (p_low + p_high) / 2

    return p_star, best_volume, p_low, p_high


def match_at_volume(orders, target_volume):
    """Match orders greedily up to target volume using price priority."""
    bids = sorted(
        [o.copy() for o in orders if o["side"] == "buy"],
        key=lambda o: (-o["px"], o["id"]),
    )
    asks = sorted(
        [o.copy() for o in orders if o["side"] == "sell"],
        key=lambda o: (o["px"], o["id"]),
    )

    trades = []
    traded = 0
    bid_i = 0
    ask_i = 0

    # Greedy match by price priority; keep it transparent.
    while traded < target_volume and bid_i < len(bids) and ask_i < len(asks):
        bid = bids[bid_i]
        ask = asks[ask_i]

        if bid["px"] < ask["px"]:
            break

        remaining = target_volume - traded
        qty = min(bid["qty"], ask["qty"], remaining)

        if qty <= 0:
            break

        trades.append((bid["id"], ask["id"], qty))
        traded += qty
        bid["qty"] -= qty
        ask["qty"] -= qty

        if bid["qty"] == 0:
            bid_i += 1
        if ask["qty"] == 0:
            ask_i += 1

    return trades, traded


def print_book(orders):
    """Display the collected order book."""
    print("\nOrder book (you entered):")
    print("  BIDS:")
    bids = sorted((o for o in orders if o["side"] == "buy"), key=lambda o: (-o["px"], o["id"]))
    asks = sorted((o for o in orders if o["side"] == "sell"), key=lambda o: (o["px"], o["id"]))

    if bids:
        for order in bids:
            print(
                f"    id={order['id']:>3}  px={order['px']:>8.2f}  qty={order['qty']}"
            )
    else:
        print("    (none)")

    print("  ASKS:")
    if asks:
        for order in asks:
            print(
                f"    id={order['id']:>3}  px={order['px']:>8.2f}  qty={order['qty']}"
            )
    else:
        print("    (none)")


def print_results(p_star, v_star, p_low, p_high, trades):
    """Print final auction results and matched trades."""
    print("\n=== Auction Result ===")
    print(f"Clearing volume: {v_star}")

    if p_star is None or v_star <= 0:
        print("No trade — book didn’t cross.")
        return

    if p_low is not None and p_high is not None and p_low != p_high:
        print(
            f"Clearing price: {p_star:.4f}  (midpoint of [{p_low:.4f}, {p_high:.4f}])"
        )
    else:
        print(f"Clearing price: {p_star:.4f}")

    if trades:
        print("\nMatched trades (uniform-price):")
        for bid_id, ask_id, qty in trades:
            print(f"  bid {bid_id:>3}  ↔  ask {ask_id:>3}  @ {p_star:.4f}  qty={qty}")
    else:
        print("No individual trades matched; this should not happen with positive volume.")


def main():
    """Run the auction CLI end-to-end."""
    print("Batch Auction Simulator")

    orders = collect_orders()
    print_book(orders)

    bids_levels, asks_levels = aggregate_levels(orders)
    p_star, v_star, p_low, p_high = find_clearing_price(bids_levels, asks_levels)

    if p_star is None or v_star == 0:
        print("\nNo trade — book didn’t cross.")
        return

    trades, traded = match_at_volume(orders, v_star)
    print_results(p_star, traded, p_low, p_high, trades)


if __name__ == "__main__":
    main()
