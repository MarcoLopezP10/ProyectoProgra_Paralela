"""scripts.run_benchmarks

Full benchmark suite: 4 objectives * 3 dimensions * N seeds.

Produces structured results under results/runs/ and a summary CSV
that can be loaded directly into the analysis notebook.

Usage examples
--------------
# Default: all functions, d=2/10/30, seeds 42 and 7
python -m scripts.run_benchmarks

# Single function, all dimensions, 5 seeds
python -m scripts.run_benchmarks --objective sphere --seeds 0 1 7 42 123

# Quick run (d=2 only, 1 seed, no grid search)
python -m scripts.run_benchmarks --dims 2 --seeds 42 --no-grid-search

# Large run for final report
python -m scripts.run_benchmarks --n-particles 200 --max-iters 1000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List

# Allow direct execution as `python scripts/run_benchmarks.py` from editors like
# VS Code while keeping `python -m scripts.run_benchmarks` working unchanged.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.run_single import RunConfig, run_one_objective
from objectives.sphere import sphere
from objectives.ackley import ackley
from objectives.latency_mix import latency_mix
from objectives.rosenbrock import rosenbrock
from objectives.rastrigin import rastrigin

OBJECTIVES = {
    "sphere":     sphere,
    "ackley":     ackley,
    "rosenbrock": rosenbrock,
    "rastrigin":  rastrigin,
    "latency_mix": latency_mix,
}

DEFAULT_DIMS  = [2, 10, 30]
DEFAULT_SEEDS = [42, 7]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full PSO benchmark suite (baseline + V0 + V1 + V2 + V3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--objective", nargs="+", choices=list(OBJECTIVES),
                   default=list(OBJECTIVES), metavar="FN",
                   help="Objective function(s) to benchmark.")
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS,
                   metavar="D", help="Dimensions to evaluate.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   metavar="S", help="Random seeds (one run per seed).")
    p.add_argument("--bounds-lo", type=float, default=-5.0)
    p.add_argument("--bounds-hi", type=float, default=5.0)
    p.add_argument("--n-particles", type=int, default=None)
    p.add_argument("--max-iters",   type=int, default=None)
    p.add_argument("--tol",         type=float, default=None)
    p.add_argument("--patience",    type=int, default=None)
    p.add_argument("--vmax-ratio",  type=float, default=None,
                   help="Initial velocity cap as a fraction of the search range.")
    p.add_argument("--log-every",   type=int, default=0,
                   help="Per-iteration log frequency (0 = off, keeps output clean).")
    p.add_argument("--workers",     type=int, default=None,
                   help="Max workers for ThreadPoolExecutor.")
    p.add_argument("--process-workers", type=int, default=None,
                   help="Max workers for ProcessPoolExecutor.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Particles per process task in V2.")
    p.add_argument("--grid-search", action="store_true",
                   help="Use grid search to pick hyperparameters (slower).")
    p.add_argument("--no-grid-search", dest="grid_search", action="store_false")
    p.set_defaults(grid_search=False)
    p.add_argument("--grid-strategy", choices=["v0", "v1", "v2", "v3"], default="v0",
                   help="Strategy used during the optional grid search.")
    p.add_argument("--grid-metric", choices=["final_fitness", "auc", "convergence_iter", "time_s"],
                   default="final_fitness",
                   help="Metric minimized during the optional grid search.")
    p.add_argument("--out-dir",   default="results/runs")
    p.add_argument("--plots-dir", default="logs/convergence")
    p.add_argument("--log-dir",   default="logs")
    p.add_argument("--summary-csv", default="results/benchmark_summary.csv",
                   help="Path for the aggregated summary CSV.")
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# CSV summary writer
# ──────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "objective", "dim", "seed",
    "w", "c1", "c2", "n_particles",
    "v0_fitness", "v0_iters", "v0_time_s",
    "v0_auc", "v0_convergence_iter",
    "v0_pct_eval", "v0_pct_update",
    "v1_fitness", "v1_iters", "v1_time_s",
    "v1_auc", "v1_convergence_iter",
    "v1_pct_eval", "v1_pct_update",
    "v1_speedup",
    "v2_fitness", "v2_iters", "v2_time_s",
    "v2_auc", "v2_convergence_iter",
    "v2_pct_eval", "v2_pct_update",
    "v2_speedup",
    "v3_fitness", "v3_iters", "v3_time_s",
    "v3_auc", "v3_convergence_iter",
    "v3_pct_eval", "v3_pct_update",
    "v3_speedup",
    "process_workers", "batch_size",
    "selected_grid_metric",
    "baseline_fitness", "baseline_time_s",
    "winner",
]


def _result_to_row(result: dict) -> dict:
    v0 = result["v0"]
    v1 = result["v1"]
    bl = result["baseline"]
    hp = result["hyperparams"]
    v1_speedup = v0["time_s"] / v1["time_s"] if v1["time_s"] > 0 else float("inf")
    v2 = result["v2"]
    v2_speedup = v0["time_s"] / v2["time_s"] if v2["time_s"] > 0 else float("inf")
    v3 = result["v3"]
    v3_speedup = v0["time_s"] / v3["time_s"] if v3["time_s"] > 0 else float("inf")
    return {
        "objective":    result["objective"],
        "dim":          result["dim"],
        "seed":         result["seed"],
        "w":            hp["w"],
        "c1":           hp["c1"],
        "c2":           hp["c2"],
        "n_particles":  hp["n_particles"],
        "v0_fitness":   v0["best_fit"],
        "v0_iters":     v0["iters"],
        "v0_time_s":    round(v0["time_s"], 5),
        "v0_auc":       round(v0["auc"], 6),
        "v0_convergence_iter": v0["convergence_iter"],
        "v0_pct_eval":  round(v0["timing"]["pct_eval"], 2),
        "v0_pct_update":round(v0["timing"]["pct_update"], 2),
        "v1_fitness":   v1["best_fit"],
        "v1_iters":     v1["iters"],
        "v1_time_s":    round(v1["time_s"], 5),
        "v1_auc":       round(v1["auc"], 6),
        "v1_convergence_iter": v1["convergence_iter"],
        "v1_pct_eval":  round(v1["timing"]["pct_eval"], 2),
        "v1_pct_update":round(v1["timing"]["pct_update"], 2),
        "v1_speedup":   round(v1_speedup, 4),
        "v2_fitness":   v2["best_fit"],
        "v2_iters":     v2["iters"],
        "v2_time_s":    round(v2["time_s"], 5),
        "v2_auc":       round(v2["auc"], 6),
        "v2_convergence_iter": v2["convergence_iter"],
        "v2_pct_eval":  round(v2["timing"]["pct_eval"], 2),
        "v2_pct_update":round(v2["timing"]["pct_update"], 2),
        "v2_speedup":   round(v2_speedup, 4),
        "v3_fitness":   v3["best_fit"],
        "v3_iters":     v3["iters"],
        "v3_time_s":    round(v3["time_s"], 5),
        "v3_auc":       round(v3["auc"], 6),
        "v3_convergence_iter": v3["convergence_iter"],
        "v3_pct_eval":  round(v3["timing"]["pct_eval"], 2),
        "v3_pct_update":round(v3["timing"]["pct_update"], 2),
        "v3_speedup":   round(v3_speedup, 4),
        "process_workers": v2["max_workers"],
        "batch_size":   v2["batch_size"],
        "selected_grid_metric": result.get("selected_grid_metric"),
        "baseline_fitness": bl["best_fit"],
        "baseline_time_s":  round(bl["time_s"], 5),
        "winner":       result["winner"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)

    total = len(args.objective) * len(args.dims) * len(args.seeds)
    print(
        f"\nBenchmark suite: {len(args.objective)} functions × "
        f"{len(args.dims)} dims × {len(args.seeds)} seeds = {total} runs\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
    rows: List[dict] = []
    suite_start = time.perf_counter()

    run_n = 0
    for obj_name in args.objective:
        obj_fn = OBJECTIVES[obj_name]
        for dim in args.dims:
            for seed in args.seeds:
                run_n += 1
                bounds = ([args.bounds_lo] * dim, [args.bounds_hi] * dim)
                cfg = RunConfig(
                    seed=seed,
                    dim=dim,
                    bounds=bounds,
                    n_particles=args.n_particles,
                    use_grid_search=args.grid_search,
                    grid_metric=args.grid_metric,
                    grid_strategy=args.grid_strategy,
                    thread_max_workers=args.workers,
                    process_max_workers=args.process_workers,
                    batch_size=args.batch_size,
                    max_iters=args.max_iters,
                    tol=args.tol,
                    patience=args.patience,
                    vmax_ratio=args.vmax_ratio,
                    log_every=args.log_every,
                    out_dir=args.out_dir,
                    plots_dir=args.plots_dir,
                    log_dir=args.log_dir,
                    repo_root=".",
                    save_files=True,
                )

                print(f"\n[{run_n}/{total}] {obj_name.upper()} | d={dim} | seed={seed}")
                print("-" * 50)

                result = run_one_objective(obj_fn, cfg)
                rows.append(_result_to_row(result))

    # ── Write aggregated CSV ──────────────────────────────────────────
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    suite_elapsed = time.perf_counter() - suite_start
    print(f"\n{'='*60}")
    print(f"  Benchmark suite complete in {suite_elapsed:.1f}s")
    print(f"  Summary CSV → {args.summary_csv}")
    print(f"  Results     → {args.out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
