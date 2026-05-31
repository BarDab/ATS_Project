"""Event dataclasses and EventLogger for the market microstructure simulation."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .core import SimulationParams, SimulationResult


@dataclass
class BaseEvent:
    time: float
    _seq: int = field(default=0)
    PRIORITY: ClassVar[int] = 99

    def __lt__(self, other: "BaseEvent") -> bool:
        if self.time != other.time:
            return self.time < other.time
        if self.PRIORITY != other.PRIORITY:
            return self.PRIORITY < other.PRIORITY
        return self._seq < other._seq


@dataclass
class YJumpEvent(BaseEvent):
    PRIORITY: ClassVar[int] = 0
    y_value: float = 0.0


@dataclass
class SniperObserveEvent(BaseEvent):
    PRIORITY: ClassVar[int] = 1
    agent_id: str = ""
    y_value: float = 0.0


@dataclass
class MMObserveEvent(BaseEvent):
    PRIORITY: ClassVar[int] = 2
    y_value: float = 0.0


@dataclass
class InvestorArriveEvent(BaseEvent):
    PRIORITY: ClassVar[int] = 3


@dataclass
class DeferredLabelEvent(BaseEvent):
    PRIORITY: ClassVar[int] = 4
    fill_id: str = ""


_LOG_FIELDS = [
    "timestamp", "event_type", "agent_id",
    "book_bid", "book_ask", "book_spread", "Y_value",
    "mm_inventory", "mm_realized_pnl", "mm_unrealized_pnl",
    "alpha", "current_spread_ticks", "spread_income", "adverse_selection_loss",
    "order_id", "side", "fill_price", "fill_quantity", "taker_agent_id",
    "informed",
]


class EventLogger:

    def __init__(self, params: SimulationParams):
        self.enabled = params.enable_logging
        self._csv_file = None
        self._writer = None
        self._csv_path: str | None = None
        self._json_path: str | None = None

        if self.enabled:
            os.makedirs(params.log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            print(f"Logging to {stamp}")
            label = f"{params.run_label}_" if params.run_label else ""
            self._csv_path = os.path.join(params.log_dir, f"events_{label}{stamp}.csv")
            self._json_path = os.path.join(params.log_dir, f"summary_{label}{stamp}.json")
            self._csv_file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=_LOG_FIELDS,
                                          extrasaction="ignore")
            self._writer.writeheader()

    def log_event(self, timestamp, event_type, agent_id, **kwargs):
        if not self.enabled:
            return
        row = {"timestamp": int(round(timestamp * 1000)), "event_type": event_type,
               "agent_id": agent_id}
        row.update(kwargs)
        self._writer.writerow(row)

    def write_summary(self, result: SimulationResult, params: SimulationParams):
        if not self.enabled or self._json_path is None:
            return
        import dataclasses
        params_dict = dataclasses.asdict(params)
        params_dict["target_jumps_per_day"] = params.target_jumps_per_day
        params_dict["jump_size_probs"] = list(params.jump_size_probs)
        mm_snap = result.mm_pnl_history[-1] if result.mm_pnl_history else {}
        summary = {
            "params": params_dict,
            "total_trades": (result.sniper1_trades + result.sniper2_trades
                             + result.investor_trades),
            "mm_final_realized_pnl": mm_snap.get("realized_pnl", 0.0),
            "mm_final_unrealized_pnl": mm_snap.get("unrealized_pnl", 0.0),
            "mm_final_total_pnl": mm_snap.get("total_pnl", 0.0),
            "mm_final_inventory": mm_snap.get("inventory", 0.0),
            "sniper1_pnl": result.sniper1_pnl,
            "sniper2_pnl": result.sniper2_pnl,
            "sniper1_trades": result.sniper1_trades,
            "sniper2_trades": result.sniper2_trades,
            "investor_trades": result.investor_trades,
            "avg_spread": result.avg_spread,
            "simulation_duration_seconds": params.T,
            "mm_final_alpha": result.mm_final_alpha,
            "mm_final_spread_ticks": result.mm_final_spread_ticks,
            "mm_spread_income": mm_snap.get("spread_income", 0.0),
            "mm_adverse_selection_loss": mm_snap.get("adverse_selection_loss", 0.0),
            "mm_total_attributed_pnl": mm_snap.get("total_attributed_pnl", 0.0),
            "mm_n_fills_labeled": mm_snap.get("n_fills_labeled", 0),
            "mm_n_pending_fills_at_end": mm_snap.get("n_pending_fills", 0),
        }
        with open(self._json_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
