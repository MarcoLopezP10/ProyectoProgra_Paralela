"""scripts.run_benchmarks

Full benchmark suite: 4 objectives * 3 dimensions * N seeds.

Produces structured results under results/ and a summary CSV
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
import time
from typing import List

from experiment.run_single import RunConfig, run_one_objective
from objectives.sphere import sphere
from objectives.ackley import ackley
from objectives.rosenbrock import rosenbrock
from objectives.rastrigin import rastrigin

OBJECTIVES = {
    "sphere":     sphere,
    "ackley":     ackley,
    "rosenbrock": rosenbrock,
    "rastrigin":  rastrigin,
}

DEFAULT_DIMS  = [2, 10, 30]
DEFAULT_SEEDS = [42, 7]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the full PSO benchmark suite (V0 + V1 + PySwarm baseline).",
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
    p.add_argument("--n-particles", type=int, default=80)
    p.add_argument("--max-iters",   type=int, default=500)
    p.add_argument("--tol",         type=float, default=1e-10)
    p.add_argument("--patience",    type=int, default=50)
    p.add_argument("--log-every",   type=int, default=0,
                   help="Per-iteration log frequency (0 = off, keeps output clean).")
    p.add_argument("--workers",     type=int, default=None,
                   help="Max workers for ThreadPoolExecutor.")
    p.add_argument("--grid-search", action="store_true",
                   help="Use grid search to pick hyperparameters (slower).")
    p.add_argument("--no-grid-search", dest="grid_search", action="store_false")
    p.set_defaults(grid_search=False)
    p.add_argument("--out-dir",   default="results")
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
    "v0_pct_eval", "v0_pct_update",
    "v1_fitness", "v1_iters", "v1_time_s",
    "v1_pct_eval", "v1_pct_update",
    "v1_speedup",
    "baseline_fitness", "baseline_time_s",
    "winner",
]


def _result_to_row(result: dict) -> dict:
    v0 = result["v0"]
    v1 = result["v1"]
    bl = result["baseline"]
    hp = result["hyperparams"]
    speedup = v0["time_s"] / v1["time_s"] if v1["time_s"] > 0 else float("inf")
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
        "v0_pct_eval":  round(v0["timing"]["pct_eval"], 2),
        "v0_pct_update":round(v0["timing"]["pct_update"], 2),
        "v1_fitness":   v1["best_fit"],
        "v1_iters":     v1["iters"],
        "v1_time_s":    round(v1["time_s"], 5),
        "v1_pct_eval":  round(v1["timing"]["pct_eval"], 2),
        "v1_pct_update":round(v1["timing"]["pct_update"], 2),
        "v1_speedup":   round(speedup, 4),
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

    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
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
                    thread_max_workers=args.workers,
                    max_iters=args.max_iters,
                    tol=args.tol,
                    patience=args.patience,
                    log_every=args.log_every,
                    out_dir=args.out_dir,
                    plots_dir=args.plots_dir,
                    log_dir=args.log_dir,
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