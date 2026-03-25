"""experiment.grid_search

Configurable grid search for PSO hyperparameters.

The search evaluates each (w, c1, c2, n_particles) combination over
multiple seeds and reports the average best fitness, as required by
the project specification (3×3×3 grid, 5 seeds per combination).
"""

from __future__ import annotations

import csv
import itertools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.swarm import Swarm
from core.pso import PSO
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from options.evaluator import SequentialEvaluator


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


# ──────────────────────────────────────────────────────────────────────────────
# Core grid search
# ──────────────────────────────────────────────────────────────────────────────

def grid_search(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[List[float], List[float]],
    w_values: List[float]           = None,
    c1_values: List[float]          = None,
    c2_values: List[float]          = None,
    n_particles_values: List[int]   = None,
    seeds: List[int]                = None,
    max_iters: int                  = 200,
    tol: float                      = 1e-8,
    patience: int                   = 40,
    vmax_ratio: float               = 0.2,
    verbose: bool                   = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Grid search over PSO hyperparameters with multiple seeds per combination.

    Each combination is evaluated over all seeds and ranked by mean best
    fitness. The full results table is returned so it can be saved to CSV.

    Parameters
    ----------
    objective_fn : Callable
    dim : int
    bounds : (lower, upper)
    w_values : list of inertia weights to try          (default 3×3×3 grid)
    c1_values : list of cognitive coefficients
    c2_values : list of social coefficients
    n_particles_values : list of swarm sizes
    seeds : list of random seeds                       (default 5 seeds)
    max_iters : int    Budget per combination (kept low for speed).
    tol, patience : early-stop parameters
    verbose : bool     Print progress.

    Returns
    -------
    best_params : dict   Best hyperparameter combination found.
    all_results : list   Full results table (one row per combination).
    """
    # Defaults: 3×3×3 grid with 5 seeds as specified in the project brief
    if w_values          is None: w_values          = [0.4, 0.6, 0.8]
    if c1_values         is None: c1_values         = [1.2, 1.5, 1.8]
    if c2_values         is None: c2_values         = [1.2, 1.5, 1.8]
    if n_particles_values is None: n_particles_values = [50]
    if seeds             is None: seeds             = [0, 1, 7, 42, 123]

    combos = list(itertools.product(w_values, c1_values, c2_values, n_particles_values))
    total  = len(combos) * len(seeds)

    if verbose:
        print(
            f"  Grid search: {len(combos)} combinations × {len(seeds)} seeds "
            f"= {total} runs  (max_iters={max_iters})"
        )

    all_results: List[Dict[str, Any]] = []
    best_mean   = float("inf")
    best_params: Dict[str, Any] = {}
    run_n = 0

    for w, c1, c2, n_part in combos:
        seed_fits: List[float] = []

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
                evaluator=SequentialEvaluator(objective_fn),
                bounds_handler=ClampBounds(bounds[0], bounds[1]),
                topology=GlobalBestTopology(),
                w=float(w), c1=float(c1), c2=float(c2),
                max_iters=max_iters,
                seed=seed,
                tol=tol,
                patience=patience,
                log_every=0,
            )
            _, best_fit, _, _ = pso.run()
            seed_fits.append(float(best_fit))

            if verbose:
                print(
                    f"    [{run_n}/{total}] w={w:.2f} c1={c1:.2f} c2={c2:.2f} "
                    f"n={n_part} seed={seed} → fit={best_fit:.4e}",
                    flush=True,
                )

        mean_fit = float(np.mean(seed_fits))
        std_fit  = float(np.std(seed_fits))

        row = {
            "w": float(w), "c1": float(c1), "c2": float(c2),
            "n_particles": int(n_part),
            "mean_fitness": mean_fit,
            "std_fitness": std_fit,
            "min_fitness": float(np.min(seed_fits)),
            "max_fitness": float(np.max(seed_fits)),
            "seeds": seeds,
            "per_seed_fitness": seed_fits,
        }
        all_results.append(row)

        if mean_fit < best_mean:
            best_mean = mean_fit
            best_params = {
                "w": float(w), "c1": float(c1), "c2": float(c2),
                "n_particles": int(n_part),
                "mean_fitness": mean_fit,
                "std_fitness": std_fit,
            }

    # Sort by mean fitness
    all_results.sort(key=lambda r: r["mean_fitness"])

    if verbose:
        print(
            f"\n  Best: w={best_params['w']} c1={best_params['c1']} "
            f"c2={best_params['c2']} n={best_params['n_particles']} "
            f"mean_fit={best_mean:.4e}"
        )

    return best_params, all_results


def simple_grid_search(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[List[float], List[float]],
    max_iters: int = 200,
    seed: int = 42,
    vmax_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Lightweight multi-seed grid search used by run_single.py.

    Uses a small, dimension-aware search space and three seeds so parameter
    selection is more stable without becoming too slow.
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
        verbose=False,
    )
    return best_params


# ──────────────────────────────────────────────────────────────────────────────
# CSV persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_grid_search_csv(
    results: List[Dict[str, Any]],
    out_path: str,
    objective: str,
    dim: int,
) -> None:
    """Save the full grid search results table as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fields = ["objective", "dim", "w", "c1", "c2", "n_particles",
              "mean_fitness", "std_fitness", "min_fitness", "max_fitness"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({"objective": objective, "dim": dim, **row})
