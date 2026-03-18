"""utils.io

On-disk persistence for PSO experiments.

Format choices
--------------
JSON  — configuration + scalar metrics per run (human-readable, structured,
        easy to load with json.load or pandas.read_json).
CSV   — per-iteration time series (best fitness per step). Compact, trivially
        loadable with pandas.read_csv and plottable without parsing.

Each experiment is saved under:
    results/<objective>_d<dim>_s<seed>/
        summary.json     — full config + final metrics + timing breakdown
        history.csv      — V0 convergence curve
        history_v1.csv   — V1 (threading) convergence curve (optional)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class TimingBreakdown:
    """Wall-clock time breakdown for a single PSO run."""
    total_s: float = 0.0
    eval_s: float = 0.0
    update_s: float = 0.0
    overhead_s: float = 0.0
    pct_eval: float = 0.0
    pct_update: float = 0.0


@dataclass
class MethodResult:
    """Results for one evaluation strategy (V0, V1, …)."""
    best_fitness: float = float("inf")
    iterations: int = 0
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    max_workers: Optional[int] = None   # only relevant for thread/process variants


@dataclass
class ExperimentSummary:
    """Serialisable summary for a single experiment run."""

    # ── Problem definition ────────────────────────────────────────────
    objective: str
    dim: int
    bounds_lower: List[float]
    bounds_upper: List[float]
    seed: int

    # ── Hyperparameters ───────────────────────────────────────────────
    hyperparam_source: str      # "grid_search" | "fixed_config"
    w: float
    c1: float
    c2: float
    n_particles: int
    max_iters: int
    tol: float
    patience: int

    # ── Results per method ────────────────────────────────────────────
    v0: MethodResult = field(default_factory=MethodResult)
    v1: Optional[MethodResult] = None
    baseline: Optional[MethodResult] = None

    winner: str = ""

    # ── Metadata ──────────────────────────────────────────────────────
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: Optional[str] = None


def result_dir(base: str, objective: str, dim: int, seed: int) -> str:
    """
    Return a structured output directory path.

    Example: results/sphere_d2_s42/
    """
    return os.path.join(base, f"{objective}_d{dim}_s{seed}")


def save_summary_json(summary: ExperimentSummary, out_path: str) -> None:
    """Save the experiment summary as an indented JSON file."""
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