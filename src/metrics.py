"""Trade quality metrics: mid, VWAP, slippage."""

import csv
from typing import List, Dict, Optional


def compute_mid(quotes: List[dict]) -> Optional[float]:
    """Compute average mid price from quotes."""
    mids = []
    for q in quotes:
        bid = q.get("bid")
        ask = q.get("ask")
        if bid is not None and ask is not None:
            mids.append((bid + ask) / 2.0)

    return sum(mids) / len(mids) if mids else None


def compute_vwap(trades: List[dict]) -> Optional[float]:
    """Compute volume-weighted average price."""
    if not trades:
        return None

    total_val = 0.0
    total_qty = 0

    for t in trades:
        total_val += t["price"] * t["qty"]
        total_qty += t["qty"]

    return total_val / total_qty if total_qty > 0 else None


def compute_slippage(trades: List[dict], reference_price: float) -> List[float]:
    """Compute signed slippage per trade (trade_price - reference)."""
    return [t["price"] - reference_price for t in trades]


def signed_slippage_ticks(trades: List[dict], reference_price: float, tick: float = 0.01) -> List[float]:
    """
    Compute signed slippage in ticks.
    
    Args:
        trades: List of trades with 'price', 'taker_side'.
        reference_price: Reference price (e.g., VWAP or mid).
        tick: Tick size (default 0.01).
    
    Returns:
        List of signed slippage values: ((price - ref)/tick) * (+1 for BUY, -1 for SELL).
    """
    result = []
    for t in trades:
        raw_slippage = (t["price"] - reference_price) / tick
        sign = 1 if t.get("taker_side") == "BUY" else -1
        result.append(raw_slippage * sign)
    return result


def load_trades(path: str) -> List[dict]:
    """Load trades from CSV."""
    trades = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                "buyer_id": int(row["buyer_id"]),
                "seller_id": int(row["seller_id"]),
                "price": float(row["price"]),
                "qty": int(row["qty"]),
                "taker_side": row.get("taker_side", "BUY"),
            })
    return trades


def load_quotes(path: str) -> List[dict]:
    """Load quotes from CSV."""
    quotes = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            quotes.append({
                "bid": float(row["bid"]) if row["bid"] else None,
                "ask": float(row["ask"]) if row["ask"] else None,
            })
    return quotes


def compare_modes(batch_trades: List[dict], cont_trades: List[dict]) -> str:
    """Generate markdown comparison table."""
    batch_vwap = compute_vwap(batch_trades)
    cont_vwap = compute_vwap(cont_trades)

    batch_vol = sum(t["qty"] for t in batch_trades)
    cont_vol = sum(t["qty"] for t in cont_trades)

    # Compute signed slippage (using VWAP as reference)
    batch_slippage_ticks = signed_slippage_ticks(batch_trades, batch_vwap) if batch_vwap and batch_trades else []
    cont_slippage_ticks = signed_slippage_ticks(cont_trades, cont_vwap) if cont_vwap and cont_trades else []
    
    avg_batch_slippage = sum(batch_slippage_ticks) / len(batch_slippage_ticks) if batch_slippage_ticks else 0.0
    avg_cont_slippage = sum(cont_slippage_ticks) / len(cont_slippage_ticks) if cont_slippage_ticks else 0.0

    lines = [
        "# Batch vs Continuous Comparison",
        "",
        "| Metric | Batch | Continuous |",
        "| --- | --- | --- |",
        f"| Trades | {len(batch_trades)} | {len(cont_trades)} |",
        f"| Volume | {batch_vol} | {cont_vol} |",
        f"| VWAP | {batch_vwap:.4f} | {cont_vwap:.4f} |",
        f"| Avg signed slippage (ticks) | {avg_batch_slippage:.2f} | {avg_cont_slippage:.2f} |",
        "",
    ]

    return "\n".join(lines)

