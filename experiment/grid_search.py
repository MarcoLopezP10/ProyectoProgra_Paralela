"""experiment.grid_search

Configurable grid search for PSO hyperparameters up to V2.

The search can evaluate V0, V1, or V2 using a selectable optimisation metric:
- final_fitness
- auc
- convergence_iter
- time_s
"""

from __future__ import annotations

import csv
import itertools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.pso import PSO
from core.swarm import Swarm
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from parallel.evaluator import build_evaluator

SUPPORTED_GRID_METRICS = {
    "final_fitness",
    "auc",
    "convergence_iter",
    "time_s",
}


def recommended_profile(objective_name: str, dim: int) -> Dict[str, Any]:
    """Return conservative, dimension-aware defaults for stronger fitness."""
    profile: Dict[str, Any] = {
        "w": 0.6,
        "c1": 1.4,
        "c2": 1.6,
        "n_particles": 60 if dim <= 2 else 100 if dim <= 10 else 140,
        "max_iters": 500 if dim <= 2 else 900 if dim <= 10 else 1400,
        "patience": 60 if dim <= 2 else 100 if dim <= 10 else 160,
        "tol": 1e-10,
        "vmax_ratio": 0.2,
        "quick_seeds": [0, 7, 42],
    }

    if objective_name == "ackley":
        profile.update({
            "w": 0.5,
            "c1": 1.3,
            "c2": 1.7,
            "n_particles": 70 if dim <= 2 else 110 if dim <= 10 else 160,
            "max_iters": 700 if dim <= 2 else 1100 if dim <= 10 else 1600,
            "patience": 90 if dim <= 2 else 130 if dim <= 10 else 200,
            "vmax_ratio": 0.15,
        })
    elif objective_name == "rastrigin":
        profile.update({
            "w": 0.5,
            "c1": 1.2,
            "c2": 1.8,
            "n_particles": 80 if dim <= 2 else 120 if dim <= 10 else 160,
            "max_iters": 800 if dim <= 2 else 1200 if dim <= 10 else 1800,
            "patience": 110 if dim <= 2 else 160 if dim <= 10 else 240,
            "vmax_ratio": 0.12,
        })
    elif objective_name == "rosenbrock":
        profile.update({
            "w": 0.7,
            "c1": 1.3,
            "c2": 1.6,
            "n_particles": 70 if dim <= 2 else 110 if dim <= 10 else 150,
            "max_iters": 900 if dim <= 2 else 1500 if dim <= 10 else 2200,
            "patience": 120 if dim <= 2 else 180 if dim <= 10 else 260,
            "vmax_ratio": 0.08,
        })

    if objective_name == "sphere":
        n_values = [50, 80] if dim <= 2 else [80, 120] if dim <= 10 else [120, 160]
        w_values = [0.4, 0.5, 0.6, 0.7]
        c1_values = [1.2, 1.5, 1.8]
        c2_values = [1.2, 1.5, 1.8]
    elif objective_name == "ackley":
        n_values = [60, 90] if dim <= 2 else [90, 120] if dim <= 10 else [120, 160]
        w_values = [0.4, 0.5, 0.6, 0.7]
        c1_values = [1.1, 1.3, 1.5]
        c2_values = [1.5, 1.7, 1.9]
    elif objective_name == "rastrigin":
        n_values = [80, 100] if dim <= 2 else [100, 140] if dim <= 10 else [140, 180]
        w_values = [0.4, 0.5, 0.6]
        c1_values = [1.0, 1.2, 1.4]
        c2_values = [1.6, 1.8, 2.0]
    else:
        n_values = [60, 90] if dim <= 2 else [90, 120] if dim <= 10 else [120, 160]
        w_values = [0.5, 0.6, 0.7]
        c1_values = [1.1, 1.3, 1.5]
        c2_values = [1.4, 1.6, 1.8]

    profile.update({
        "w_values": w_values,
        "c1_values": c1_values,
        "c2_values": c2_values,
        "n_particles_values": n_values,
    })
    return profile


def _metric_from_run(pso: PSO, best_fit: float, time_s: float, metric: str) -> float:
    """Compute the metric used to rank one grid-search run."""
    if metric == "final_fitness":
        return float(best_fit)
    if metric == "auc":
        return float(pso.area_under_curve())
    if metric == "convergence_iter":
        return float(pso.convergence_iteration())
    if metric == "time_s":
        return float(time_s)
    raise ValueError(f"Unsupported grid-search metric: {metric}")


def _aggregate_seed_rows(
    strategy: str,
    metric: str,
    objective_name: str,
    dim: int,
    w: float,
    c1: float,
    c2: float,
    n_particles: int,
    seeds: List[int],
    rows: List[Dict[str, float]],
    max_workers: Optional[int],
    batch_size: Optional[int],
) -> Dict[str, Any]:
    """Aggregate one hyperparameter combination across all seeds."""
    metric_values = [row["metric_value"] for row in rows]
    fitness_values = [row["best_fitness"] for row in rows]
    auc_values = [row["auc"] for row in rows]
    time_values = [row["time_s"] for row in rows]
    convergence_values = [row["convergence_iter"] for row in rows]

    return {
        "objective": objective_name,
        "dim": dim,
        "strategy": strategy,
        "selected_metric": metric,
        "w": float(w),
        "c1": float(c1),
        "c2": float(c2),
        "n_particles": int(n_particles),
        "max_workers": max_workers,
        "batch_size": batch_size,
        "mean_metric": float(np.mean(metric_values)),
        "std_metric": float(np.std(metric_values)),
        "mean_fitness": float(np.mean(fitness_values)),
        "std_fitness": float(np.std(fitness_values)),
        "mean_auc": float(np.mean(auc_values)),
        "std_auc": float(np.std(auc_values)),
        "mean_time_s": float(np.mean(time_values)),
        "std_time_s": float(np.std(time_values)),
        "mean_convergence_iter": float(np.mean(convergence_values)),
        "std_convergence_iter": float(np.std(convergence_values)),
        "min_fitness": float(np.min(fitness_values)),
        "max_fitness": float(np.max(fitness_values)),
        "seeds": list(seeds),
        "per_seed_rows": rows,
    }


def grid_search(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[List[float], List[float]],
    w_values: Optional[List[float]] = None,
    c1_values: Optional[List[float]] = None,
    c2_values: Optional[List[float]] = None,
    n_particles_values: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    max_iters: int = 200,
    tol: float = 1e-8,
    patience: int = 40,
    vmax_ratio: float = 0.2,
    strategy: str = "v0",
    metric: str = "final_fitness",
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Grid search over PSO hyperparameters for a selected execution strategy.
    """
    metric = metric.lower()
    if metric not in SUPPORTED_GRID_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(SUPPORTED_GRID_METRICS)}, got {metric!r}"
        )

    if w_values is None:
        w_values = [0.4, 0.6, 0.8]
    if c1_values is None:
        c1_values = [1.2, 1.5, 1.8]
    if c2_values is None:
        c2_values = [1.2, 1.5, 1.8]
    if n_particles_values is None:
        n_particles_values = [50]
    if seeds is None:
        seeds = [0, 1, 7, 42, 123]

    objective_name = getattr(objective_fn, "__name__", "objective")
    combos = list(itertools.product(w_values, c1_values, c2_values, n_particles_values))
    total = len(combos) * len(seeds)

    if verbose:
        print(
            f"  Grid search ({strategy.upper()} / {metric}): {len(combos)} combinations "
            f"x {len(seeds)} seeds = {total} runs (max_iters={max_iters})"
        )

    all_results: List[Dict[str, Any]] = []
    best_result: Dict[str, Any] = {}
    best_metric = float("inf")
    run_n = 0

    for w, c1, c2, n_part in combos:
        per_seed_rows: List[Dict[str, float]] = []

        for seed in seeds:
            run_n += 1
            rng = np.random.default_rng(seed)
            swarm = Swarm(
                n_particles=int(n_part),
                dim=dim,
                bounds=bounds,
                rng=rng,
                vmax_ratio=vmax_ratio,
            )
            pso = PSO(
                swarm=swarm,
                evaluator=build_evaluator(
                    strategy,
                    objective_fn,
                    max_workers=max_workers,
                    batch_size=batch_size,
                ),
                bounds_handler=ClampBounds(bounds[0], bounds[1]),
                topology=GlobalBestTopology(),
                w=float(w),
                c1=float(c1),
                c2=float(c2),
                max_iters=max_iters,
                seed=seed,
                tol=tol,
                patience=patience,
                log_every=0,
            )
            _, best_fit, time_s, _ = pso.run()
            per_seed_row = {
                "seed": float(seed),
                "best_fitness": float(best_fit),
                "auc": float(pso.area_under_curve()),
                "convergence_iter": float(pso.convergence_iteration()),
                "time_s": float(time_s),
                "metric_value": _metric_from_run(pso, best_fit, time_s, metric),
            }
            per_seed_rows.append(per_seed_row)

            if verbose:
                print(
                    f"    [{run_n}/{total}] {strategy.upper()} w={w:.2f} c1={c1:.2f} "
                    f"c2={c2:.2f} n={n_part} seed={seed} "
                    f"-> fit={best_fit:.4e} auc={per_seed_row['auc']:.4e} "
                    f"conv={int(per_seed_row['convergence_iter'])} t={time_s:.4f}s",
                    flush=True,
                )

        row = _aggregate_seed_rows(
            strategy=strategy,
            metric=metric,
            objective_name=objective_name,
            dim=dim,
            w=float(w),
            c1=float(c1),
            c2=float(c2),
            n_particles=int(n_part),
            seeds=list(seeds),
            rows=per_seed_rows,
            max_workers=max_workers,
            batch_size=batch_size,
        )
        all_results.append(row)

        if row["mean_metric"] < best_metric:
            best_metric = row["mean_metric"]
            best_result = {
                "w": row["w"],
                "c1": row["c1"],
                "c2": row["c2"],
                "n_particles": row["n_particles"],
                "strategy": strategy,
                "selected_metric": metric,
                "mean_metric": row["mean_metric"],
                "std_metric": row["std_metric"],
                "mean_fitness": row["mean_fitness"],
                "mean_auc": row["mean_auc"],
                "mean_time_s": row["mean_time_s"],
                "mean_convergence_iter": row["mean_convergence_iter"],
                "max_workers": max_workers,
                "batch_size": batch_size,
            }

    all_results.sort(key=lambda row: row["mean_metric"])
    return best_result, all_results


def simple_grid_search(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[List[float], List[float]],
    max_iters: int = 200,
    seed: int = 42,
    vmax_ratio: Optional[float] = None,
    strategy: str = "v0",
    metric: str = "final_fitness",
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lightweight multi-seed grid search used by run_single.py.
    """
    name = getattr(objective_fn, "__name__", "objective")
    profile = recommended_profile(name, dim)
    quick_seeds = list(profile["quick_seeds"])
    if seed not in quick_seeds:
        quick_seeds[-1] = seed

    grid_iters = max(60, int(max_iters * 0.5))
    best_params, _ = grid_search(
        objective_fn=objective_fn,
        dim=dim,
        bounds=bounds,
        w_values=profile["w_values"],
        c1_values=profile["c1_values"],
        c2_values=profile["c2_values"],
        n_particles_values=profile["n_particles_values"],
        seeds=quick_seeds,
        max_iters=grid_iters,
        tol=profile["tol"],
        patience=max(40, int(profile["patience"] * 0.6)),
        vmax_ratio=profile["vmax_ratio"] if vmax_ratio is None else vmax_ratio,
        strategy=strategy,
        metric=metric,
        max_workers=max_workers,
        batch_size=batch_size,
        verbose=False,
    )
    return best_params


def save_grid_search_csv(
    results: List[Dict[str, Any]],
    out_path: str,
    objective: str,
    dim: int,
) -> None:
    """Save the full grid-search results table as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fields = [
        "objective",
        "dim",
        "strategy",
        "selected_metric",
        "w",
        "c1",
        "c2",
        "n_particles",
        "max_workers",
        "batch_size",
        "mean_metric",
        "std_metric",
        "mean_fitness",
        "std_fitness",
        "mean_auc",
        "std_auc",
        "mean_time_s",
        "std_time_s",
        "mean_convergence_iter",
        "std_convergence_iter",
        "min_fitness",
        "max_fitness",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({"objective": objective, "dim": dim, **row})
