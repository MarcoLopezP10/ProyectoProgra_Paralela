"""utils.edl_io

Persistence helpers for Economic Load Dispatch runs.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from utils.io import ExecutionMetadata, TimingBreakdown


@dataclass
class EDLMethodResult:
    """Persistable result for one PSO execution strategy."""

    method_key: str = ""
    strategy: str = ""
    status: str = "ok"
    best_fitness: Optional[float] = None
    fuel_cost: Optional[float] = None
    transmission_loss: Optional[float] = None
    power_balance_error: Optional[float] = None
    power_balance_residual: Optional[float] = None
    penalty: Optional[float] = None
    iterations: Optional[int] = None
    auc: Optional[float] = None
    convergence_iteration: Optional[int] = None
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    best_position: List[float] = field(default_factory=list)
    max_workers: Optional[int] = None
    batch_size: Optional[int] = None
    error: Optional[str] = None


@dataclass
class EDLRunSummary:
    """Serialisable summary for one EDL variant and seed."""

    case_name: str
    variant: str
    variant_label: str
    seed: int
    demand: float
    generator_names: List[str]
    bounds_lower: List[float]
    bounds_upper: List[float]
    use_valve_point: bool
    use_losses: bool
    penalty_power_balance: float
    w: float
    c1: float
    c2: float
    n_particles: int
    max_iters: int
    tol: float
    patience: int
    vmax_ratio: float
    v0: EDLMethodResult = field(default_factory=EDLMethodResult)
    v1: Optional[EDLMethodResult] = None
    v2: Optional[EDLMethodResult] = None
    winner: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    execution: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    notes: Optional[str] = None


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def edl_case_dir(base: str, case_name: str) -> str:
    """Directory containing all results for one dispatch case."""
    return os.path.join(base, case_name)


def edl_result_dir(base: str, case_name: str, variant: str, seed: int) -> str:
    """Directory for one case/variant/seed run."""
    return os.path.join(edl_case_dir(base, case_name), f"{variant}_s{seed}")


def save_edl_summary_json(summary: EDLRunSummary, out_path: str) -> None:
    """Write one run summary as JSON."""
    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)


def save_edl_results_csv(rows: Iterable[Dict[str, Any]], out_path: str) -> None:
    """Write an aggregate CSV for all run/method rows."""
    rows_list = list(rows)
    _ensure_parent_dir(out_path)

    max_generators = max(
        (len(row.get("best_powers", []) or []) for row in rows_list),
        default=0,
    )
    power_fields = [f"p{i}" for i in range(1, max_generators + 1)]
    fields = [
        "case_name",
        "variant",
        "variant_label",
        "seed",
        "version",
        "strategy",
        "status",
        "fitness",
        "fuel_cost",
        "losses",
        "balance_error",
        "balance_residual",
        "penalty",
        "time_s",
        "iters",
        "auc",
        "convergence_iteration",
        "max_workers",
        "batch_size",
        "error",
    ] + power_fields

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows_list:
            values = {field: row.get(field, "") for field in fields}
            powers = list(row.get("best_powers", []) or [])
            for idx, power in enumerate(powers, start=1):
                values[f"p{idx}"] = power
            writer.writerow(values)
