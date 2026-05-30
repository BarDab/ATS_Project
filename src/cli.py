"""CLI interface for batch auction simulator."""

import argparse
import csv
import sys
import os
import time
from typing import List

from src.engine import OrderBook
from src.auction import clear_batch
from src.gen import generate_orders
from src.bench import Benchmark
from src.metrics import load_trades, load_quotes, compute_vwap, compare_modes
from typing import Tuple, Optional


def _pre_auction_snapshot(batch_orders: List[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute pre-auction best bid/ask/mid from batch orders.
    
    Returns:
        (best_bid, best_ask, pre_mid) or (None, None, None) if insufficient data.
    """
    bid_prices = []
    ask_prices = []
    
    for o in batch_orders:
        if o["type"] in ["LIMIT", "IOC"] and o.get("price") is not None:
            if o["side"] == "BUY":
                bid_prices.append(o["price"])
            elif o["side"] == "SELL":
                ask_prices.append(o["price"])
    
    best_bid = max(bid_prices) if bid_prices else None
    best_ask = min(ask_prices) if ask_prices else None
    
    if best_bid is not None and best_ask is not None:
        pre_mid = (best_bid + best_ask) / 2.0
    else:
        pre_mid = None
    
    return best_bid, best_ask, pre_mid


def cmd_gen(args):
    """Generate orders CSV."""
    generate_orders(
        n=args.n,
        seed=args.seed,
        auction_interval_ms=args.auction_ms,
        cross_rate=args.cross_rate,
        output=sys.stdout,
    )


def cmd_simulate(args):
    """Run simulation in batch or continuous mode."""
    # Read orders
    orders = []
    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order = {
                "timestamp": int(row["timestamp"]),
                "order_id": int(row["order_id"]),
                "type": row["type"],
                "side": row["side"],
                "qty": int(row["qty"]) if row["qty"] else 0,
            }
            if row["price"]:
                order["price"] = float(row["price"])
            else:
                order["price"] = None

            # For CANCEL, the price field contains the target order_id
            if order["type"] == "CANCEL":
                order["cancel_id"] = int(row["price"])

            orders.append(order)

    os.makedirs(args.out, exist_ok=True)

    if args.mode == "batch":
        _simulate_batch(orders, args.interval, args.out)
    else:
        _simulate_continuous(orders, args.out)


def cmd_benchmark(args):
    """Run benchmark with timing."""
    # Read orders
    orders = []
    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order = {
                "timestamp": int(row["timestamp"]),
                "order_id": int(row["order_id"]),
                "type": row["type"],
                "side": row["side"],
                "qty": int(row["qty"]) if row["qty"] else 0,
            }
            if row["price"]:
                order["price"] = float(row["price"])
            else:
                order["price"] = None

            if order["type"] == "CANCEL":
                order["cancel_id"] = int(row["price"])

            orders.append(order)

    os.makedirs(args.out, exist_ok=True)

    bench = Benchmark()
    bench.start()

    if args.mode == "batch":
        _benchmark_batch(orders, args.interval, args.out, bench)
    else:
        _benchmark_continuous(orders, args.out, bench)

    bench.stop()
    bench.save(os.path.join(args.out, "bench.json"), args.mode, args.interval)

    print(f"Benchmark complete. Results in {args.out}/bench.json")


def cmd_compare(args):
    """Compare batch vs continuous modes."""
    # Read orders
    orders = []
    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order = {
                "timestamp": int(row["timestamp"]),
                "order_id": int(row["order_id"]),
                "type": row["type"],
                "side": row["side"],
                "qty": int(row["qty"]) if row["qty"] else 0,
            }
            if row["price"]:
                order["price"] = float(row["price"])
            else:
                order["price"] = None

            if order["type"] == "CANCEL":
                order["cancel_id"] = int(row["price"])

            orders.append(order)

    # Run batch
    os.makedirs("out/batch", exist_ok=True)
    _simulate_batch(orders, args.interval, "out/batch")

    # Run continuous
    os.makedirs("out/continuous", exist_ok=True)
    _simulate_continuous(orders, "out/continuous")

    # Load trades
    batch_trades = load_trades("out/batch/trades.csv")
    cont_trades = load_trades("out/continuous/trades.csv")

    # Compare
    table = compare_modes(batch_trades, cont_trades)
    print(table)


def cmd_metrics(args):
    """Compute metrics and generate tearsheet."""
    trades = load_trades(args.trades)
    quotes = load_quotes(args.quotes)

    vwap = compute_vwap(trades)

    lines = [
        "# Trade Metrics Tearsheet",
        "",
        f"**Total Trades:** {len(trades)}",
        f"**Total Volume:** {sum(t['qty'] for t in trades)}",
        f"**VWAP:** {vwap:.4f}" if vwap else "**VWAP:** N/A",
        "",
    ]

    tearsheet = "\n".join(lines)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "tearsheet.md")
    with open(path, "w") as f:
        f.write(tearsheet)

    print(f"Metrics written to {path}")


def _simulate_batch(orders: List[dict], interval_ms: int, out_dir: str):
    """Simulate batch auction mode."""
    # Group orders by interval
    batches = {}
    for o in orders:
        batch_id = o["timestamp"] // (interval_ms * 1_000_000)
        if batch_id not in batches:
            batches[batch_id] = []
        batches[batch_id].append(o)

    all_trades = []
    all_quotes = []

    for batch_id in sorted(batches.keys()):
        batch_orders = batches[batch_id]
        
        # Compute pre-auction snapshot
        best_bid, best_ask, pre_mid = _pre_auction_snapshot(batch_orders)
        
        # Clear batch with computed pre_mid
        clearing_price, fills = clear_batch(batch_orders, pre_mid=pre_mid, tick=0.01)

        # Log pre-auction quote (not clearing price)
        if best_bid is not None and best_ask is not None:
            all_quotes.append({"bid": best_bid, "ask": best_ask})
        
        all_trades.extend(fills)

    # Write trades
    with open(os.path.join(out_dir, "trades.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["buyer_id", "seller_id", "price", "qty", "taker_side"])
        writer.writeheader()
        writer.writerows(all_trades)

    # Write quotes
    with open(os.path.join(out_dir, "quotes.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["bid", "ask"])
        writer.writeheader()
        writer.writerows(all_quotes)

    print(f"Batch simulation complete. {len(all_trades)} trades.")


def _simulate_continuous(orders: List[dict], out_dir: str):
    """Simulate continuous matching mode."""
    book = OrderBook()

    for o in orders:
        if o["type"] == "CANCEL":
            book.cancel_order(o["cancel_id"])
        else:
            book.add_order(o["order_id"], o["side"], o["price"], o["qty"], o["type"])

        # Snapshot quote
        snap = book.snapshot()
        book.quotes.append(snap)

    # Write trades
    with open(os.path.join(out_dir, "trades.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["buyer_id", "seller_id", "price", "qty", "taker_side"])
        writer.writeheader()
        writer.writerows(book.trades)

    # Write quotes
    with open(os.path.join(out_dir, "quotes.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["bid", "ask"])
        writer.writeheader()
        writer.writerows(book.quotes)

    print(f"Continuous simulation complete. {len(book.trades)} trades.")


def _benchmark_batch(orders: List[dict], interval_ms: int, out_dir: str, bench: Benchmark):
    """Benchmark batch mode."""
    batches = {}
    for o in orders:
        batch_id = o["timestamp"] // (interval_ms * 1_000_000)
        if batch_id not in batches:
            batches[batch_id] = []
        batches[batch_id].append(o)

    all_trades = []

    for batch_id in sorted(batches.keys()):
        batch_orders = batches[batch_id]
        
        # Compute pre-auction snapshot
        best_bid, best_ask, pre_mid = _pre_auction_snapshot(batch_orders)
        
        t0 = time.perf_counter()
        clearing_price, fills = clear_batch(batch_orders, pre_mid=pre_mid, tick=0.01)
        t1 = time.perf_counter()

        for _ in batch_orders:
            bench.record((t1 - t0) / len(batch_orders))

        all_trades.extend(fills)

    # Write trades
    with open(os.path.join(out_dir, "trades.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["buyer_id", "seller_id", "price", "qty", "taker_side"])
        writer.writeheader()
        writer.writerows(all_trades)


def _benchmark_continuous(orders: List[dict], out_dir: str, bench: Benchmark):
    """Benchmark continuous mode."""
    book = OrderBook()

    for o in orders:
        t0 = time.perf_counter()
        if o["type"] == "CANCEL":
            book.cancel_order(o["cancel_id"])
        else:
            book.add_order(o["order_id"], o["side"], o["price"], o["qty"], o["type"])
        t1 = time.perf_counter()

        bench.record(t1 - t0)

    # Write trades
    with open(os.path.join(out_dir, "trades.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["buyer_id", "seller_id", "price", "qty", "taker_side"])
        writer.writeheader()
        writer.writerows(book.trades)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Batch Auction + Continuous LOB Simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # gen
    gen_parser = subparsers.add_parser("gen", help="Generate orders CSV")
    gen_parser.add_argument("--n", type=int, required=True, help="Number of orders")
    gen_parser.add_argument("--seed", type=int, required=True, help="Random seed")
    gen_parser.add_argument("--auction_ms", type=int, required=True, help="Auction interval (ms)")
    gen_parser.add_argument("--cross_rate", type=float, required=True, help="Cross rate (0.0-1.0)")
    gen_parser.set_defaults(func=cmd_gen)

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument("--in", dest="input", required=True, help="Input CSV")
    sim_parser.add_argument("--mode", choices=["batch", "continuous"], required=True)
    sim_parser.add_argument("--interval", type=int, help="Batch interval (ms)")
    sim_parser.add_argument("--out", required=True, help="Output directory")
    sim_parser.set_defaults(func=cmd_simulate)

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--in", dest="input", required=True, help="Input CSV")
    bench_parser.add_argument("--mode", choices=["batch", "continuous"], required=True)
    bench_parser.add_argument("--interval", type=int, help="Batch interval (ms)")
    bench_parser.add_argument("--out", required=True, help="Output directory")
    bench_parser.set_defaults(func=cmd_benchmark)

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare batch vs continuous")
    cmp_parser.add_argument("--in", dest="input", required=True, help="Input CSV")
    cmp_parser.add_argument("--interval", type=int, required=True, help="Batch interval (ms)")
    cmp_parser.set_defaults(func=cmd_compare)

    # metrics
    met_parser = subparsers.add_parser("metrics", help="Compute metrics")
    met_parser.add_argument("--trades", required=True, help="Trades CSV")
    met_parser.add_argument("--quotes", required=True, help="Quotes CSV")
    met_parser.add_argument("--out", required=True, help="Output directory")
    met_parser.set_defaults(func=cmd_metrics)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

