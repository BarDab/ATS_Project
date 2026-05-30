"""Benchmarking: throughput and latency tracking."""

import time
import json
import platform
import sys
from typing import List


class Benchmark:
    """Track per-order latency and compute percentiles."""

    def __init__(self):
        self.latencies_us: List[float] = []
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start overall benchmark."""
        self.start_time = time.perf_counter()

    def record(self, latency_sec: float):
        """Record single order latency in microseconds."""
        self.latencies_us.append(latency_sec * 1e6)

    def stop(self):
        """Stop overall benchmark."""
        self.end_time = time.perf_counter()

    def results(self, mode: str, interval_ms: int = None) -> dict:
        """Return benchmark results as dict."""
        if not self.latencies_us:
            return {
                "orders_processed": 0,
                "orders_per_sec": 0,
                "latency_us_p50": 0,
                "latency_us_p95": 0,
                "latency_us_p99": 0,
                "mode": mode,
                "interval_ms": interval_ms,
                "cpu": platform.processor() or platform.machine(),
                "python": sys.version.split()[0],
            }

        elapsed = self.end_time - self.start_time
        orders_processed = len(self.latencies_us)
        orders_per_sec = orders_processed / elapsed if elapsed > 0 else 0

        sorted_lat = sorted(self.latencies_us)
        p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        return {
            "orders_processed": orders_processed,
            "orders_per_sec": round(orders_per_sec, 1),
            "latency_us_p50": round(p50, 2),
            "latency_us_p95": round(p95, 2),
            "latency_us_p99": round(p99, 2),
            "mode": mode,
            "interval_ms": interval_ms,
            "cpu": platform.processor() or platform.machine(),
            "python": sys.version.split()[0],
        }

    def save(self, path: str, mode: str, interval_ms: int = None):
        """Write results to JSON file."""
        res = self.results(mode, interval_ms)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)

