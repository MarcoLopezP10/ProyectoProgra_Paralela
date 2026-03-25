"""scripts.run_grid_search

Dedicated script for the hyperparameter grid search.

Runs a 3×3×3 grid over (w, c1, c2) with 5 seeds per combination
for each objective function and dimension, as required by the
project specification.

Usage examples
--------------
# Full grid search: all functions, d=2/10/30, 5 seeds
python -m scripts.run_grid_search

# Single function and dimension (faster)
python -m scripts.run_grid_search --objective sphere --dims 2

# Custom grid
python -m scripts.run_grid_search --w 0.4 0.6 0.8 --c1 1.0 1.5 2.0 --c2 1.0 1.5 2.0

# Custom seeds
python -m scripts.run_grid_search --seeds 0 1 7 42 123

# Quick test run
python -m scripts.run_grid_search --dims 2 --seeds 42 7 --max-iters 100
"""

from __future__ import annotations

import argparse
import os
import time

from experiment.grid_search import (
    grid_search,
    recommended_profile,
    save_grid_search_csv,
)
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
DEFAULT_SEEDS = [0, 1, 7, 42, 123]   # 5 seeds as specified


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Grid search for PSO hyperparameters (3×3×3, 5 seeds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--objective", nargs="+", choices=list(OBJECTIVES),
                   default=list(OBJECTIVES), metavar="FN")
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS,
                   metavar="D")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   metavar="S", help="Seeds to average over per combination.")

    # Grid definition (3×3×3 default)
    p.add_argument("--w",  nargs="+", type=float, default=None,
                   help="Inertia weight values to try.")
    p.add_argument("--c1", nargs="+", type=float, default=None,
                   help="Cognitive coefficient values to try.")
    p.add_argument("--c2", nargs="+", type=float, default=None,
                   help="Social coefficient values to try.")
    p.add_argument("--n-particles", nargs="+", type=int, default=None,
                   help="Swarm size(s) to include in the grid.")

    p.add_argument("--bounds-lo", type=float, default=-5.0)
    p.add_argument("--bounds-hi", type=float, default=5.0)
    p.add_argument("--max-iters", type=int,   default=150,
                   help="Iterations per combination (kept low for speed).")
    p.add_argument("--tol",       type=float, default=1e-8)
    p.add_argument("--patience",  type=int,   default=40)
    p.add_argument("--vmax-ratio", type=float, default=None,
                   help="Initial velocity cap as a fraction of the search range.")

    p.add_argument("--out-dir",   default="results/grid_search",
                   help="Directory for CSV output files.")
    p.add_argument("--verbose",   action="store_true",
                   help="Print progress for every combination.")
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    combo_counts = []
    combo_labels = []
    for obj_name in args.objective:
        for dim in args.dims:
            profile = recommended_profile(obj_name, dim)
            w_values = args.w if args.w is not None else profile["w_values"]
            c1_values = args.c1 if args.c1 is not None else profile["c1_values"]
            c2_values = args.c2 if args.c2 is not None else profile["c2_values"]
            n_values = args.n_particles if args.n_particles is not None else profile["n_particles_values"]
            combo_counts.append(len(w_values) * len(c1_values) * len(c2_values) * len(n_values))
            combo_labels.append(
                f"{len(w_values)}x{len(c1_values)}x{len(c2_values)}x{len(n_values)}"
            )

    n_combos = max(combo_counts) if combo_counts else 0
    total_runs = sum(combo_counts) * len(args.seeds)
    combo_label = combo_labels[0] if combo_labels and len(set(combo_labels)) == 1 else "adaptive"

    print(f"\nGrid search")
    print(f"  Grid    : {combo_label} combinations/profile (max {n_combos})")
    print(f"  Seeds   : {args.seeds}  ({len(args.seeds)} per combination)")
    print(f"  Runs    : {total_runs} total")
    print(f"  Budget  : {args.max_iters} iters/run\n")

    suite_start = time.perf_counter()

    for obj_name in args.objective:
        obj_fn = OBJECTIVES[obj_name]
        for dim in args.dims:
            bounds = ([args.bounds_lo] * dim, [args.bounds_hi] * dim)
            profile = recommended_profile(obj_name, dim)
            w_values = args.w if args.w is not None else profile["w_values"]
            c1_values = args.c1 if args.c1 is not None else profile["c1_values"]
            c2_values = args.c2 if args.c2 is not None else profile["c2_values"]
            n_values = args.n_particles if args.n_particles is not None else profile["n_particles_values"]

            print(f"{'─'*55}")
            print(f"  {obj_name.upper()}  d={dim}")
            print(f"{'─'*55}")

            t0 = time.perf_counter()
            best_params, all_results = grid_search(
                objective_fn=obj_fn,
                dim=dim,
                bounds=bounds,
                w_values=w_values,
                c1_values=c1_values,
                c2_values=c2_values,
                n_particles_values=n_values,
                seeds=args.seeds,
                max_iters=args.max_iters,
                tol=args.tol,
                patience=args.patience,
                vmax_ratio=profile["vmax_ratio"] if args.vmax_ratio is None else args.vmax_ratio,
                verbose=args.verbose,
            )
            elapsed = time.perf_counter() - t0

            # ── Print top-5 results ───────────────────────────────────
            print(f"\n  Top 5 combinations ({obj_name} d={dim}):")
            print(f"  {'w':>5} {'c1':>5} {'c2':>5} {'n':>5} "
                  f"{'mean_fit':>14} {'std_fit':>12}")
            print(f"  {'-'*55}")
            for row in all_results[:5]:
                print(
                    f"  {row['w']:>5.2f} {row['c1']:>5.2f} {row['c2']:>5.2f} "
                    f"{row['n_particles']:>5d} "
                    f"{row['mean_fitness']:>14.4e} {row['std_fitness']:>12.4e}"
                )
            print(f"\n  Best → w={best_params['w']} c1={best_params['c1']} "
                  f"c2={best_params['c2']} n={best_params['n_particles']} "
                  f"mean={best_params['mean_fitness']:.4e} "
                  f"std={best_params['std_fitness']:.4e}")
            print(f"  Elapsed: {elapsed:.1f}s\n")

            # ── Save CSV ──────────────────────────────────────────────
            csv_path = os.path.join(
                args.out_dir,
                f"grid_{obj_name}_d{dim}.csv",
            )
            save_grid_search_csv(all_results, csv_path, obj_name, dim)
            print(f"  Saved → {csv_path}")

    suite_elapsed = time.perf_counter() - suite_start
    print(f"\n{'='*55}")
    print(f"  Grid search complete in {suite_elapsed:.1f}s")
    print(f"  Results → {args.out_dir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
