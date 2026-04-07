"""utils.io

On-disk persistence for PSO experiments.

Format choices
--------------
JSON  — configuration + scalar metrics per run (human-readable, structured,
        easy to load with json.load or pandas.read_json).
CSV   — per-iteration time series (best fitness per step). Compact, trivially
        loadable with pandas.read_csv and plottable without parsing.

Each experiment is saved under:
    results/runs/<objective>_d<dim>_s<seed>/
        summary.json         — full config + metadata + final metrics
        history_v0.csv       — V0 iteration metrics
        history_v1.csv       — V1 iteration metrics
        history_v2.csv       — V2 iteration metrics
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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
class ExecutionMetadata:
    """Environment metadata used to reproduce experiments later."""
    repo_root: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    python_version: Optional[str] = None
    python_implementation: Optional[str] = None
    platform: Optional[str] = None
    system: Optional[str] = None
    release: Optional[str] = None
    machine: Optional[str] = None
    processor: Optional[str] = None
    hostname: Optional[str] = None
    cpu_count: Optional[int] = None
    executable: Optional[str] = None


@dataclass
class MethodResult:
    """Results for one evaluation strategy (V0, V1, …)."""
    strategy: str = ""
    best_fitness: float = float("inf")
    iterations: int = 0
    # Some external baselines do not expose iteration-by-iteration history, so
    # derived convergence metrics are optional rather than forced to fake values.
    auc: Optional[float] = None
    convergence_iteration: Optional[int] = None
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    max_workers: Optional[int] = None   # only relevant for thread/process variants
    batch_size: Optional[int] = None


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
    v2: Optional[MethodResult] = None
    baseline: Optional[MethodResult] = None

    winner: str = ""
    selected_grid_metric: Optional[str] = None

    # ── Metadata ──────────────────────────────────────────────────────
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    execution: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    notes: Optional[str] = None


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path, even for bare filenames."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def result_dir(base: str, objective: str, dim: int, seed: int) -> str:
    """
    Return a structured output directory path.

    Example: results/runs/sphere_d2_s42/
    """
    return os.path.join(base, f"{objective}_d{dim}_s{seed}")


def save_summary_json(summary: ExperimentSummary, out_path: str) -> None:
    """Save the experiment summary as an indented JSON file."""
    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)


def save_history_csv(history: List[float], out_path: str) -> None:
    """Save the convergence curve (best fitness per iteration) as CSV."""
    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("iter,best_fitness\n")
        for i, v in enumerate(history):
            f.write(f"{i},{v}\n")


def save_iteration_metrics_csv(
    iteration_records: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """Save rich per-iteration metrics as CSV."""
    _ensure_parent_dir(out_path)
    headers = [
        "iter",
        "best_fitness",
        "eval_s",
        "update_s",
        "overhead_s",
        "iter_s",
        "stall_count",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in iteration_records:
            values = [str(row.get(header, "")) for header in headers]
            f.write(",".join(values) + "\n")
