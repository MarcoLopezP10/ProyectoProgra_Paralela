"""utils.io

Minimal on-disk persistence for experiments (V0 for now):
- Save one JSON summary per experiment
- Optionally save the convergence curve as CSV (best fitness per iteration)

Why JSON + CSV:
- JSON is convenient for configs/metadata (human-readable, structured)
- CSV is convenient for time series (easy to plot, easy to load in pandas)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class ExperimentSummary:
    """Serializable summary for a single experiment run."""

    objective: str
    dim: int
    bounds_lower: List[float]
    bounds_upper: List[float]
    seed: int

    # Custom PSO (this project)
    w: float
    c1: float
    c2: float
    n_particles: int
    max_iters: int
    tol: float
    patience: int
    custom_best_fitness: float
    custom_iterations: int
    custom_time_s: float

    # Baseline (PySwarm)
    baseline_best_fitness: float
    baseline_iterations: int
    baseline_time_s: float

    winner: str

    # Optional metadata
    timestamp_utc: Optional[str] = None
    notes: Optional[str] = None


def save_summary_json(summary: ExperimentSummary, out_path: str) -> None:
    """Save the experiment summary as JSON."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)


def save_history_csv(history: List[float], out_path: str) -> None:
    """Save the convergence curve (best fitness per iteration) as CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("iter,best_fitness\n")
        for i, v in enumerate(history):
            f.write(f"{i},{v}\n")
