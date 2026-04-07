"""scripts.run_pso

Entry point for a single reproducible PSO run
(V0 + V1 + V2 + PySwarm baseline).

Usage examples
--------------
# Default: all 4 objectives, d=2, seed=42
python -m scripts.run_pso

# Specific function, dimension and seed
python -m scripts.run_pso --objective sphere --dim 10 --seed 123

# All objectives, three dimensions
python -m scripts.run_pso --dim 2 10 30

# With grid search
python -m scripts.run_pso --dim 2 --grid-search

# Skip saving files (dry run)
python -m scripts.run_pso --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution as `python scripts/run_pso.py` from editors like
# VS Code while keeping `python -m scripts.run_pso` working unchanged.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.run_single import RunConfig, run_one_objective
from objectives.sphere import sphere
from objectives.ackley import ackley
from objectives.rosenbrock import rosenbrock
from objectives.rastrigin import rastrigin

OBJECTIVES = {
    "sphere": sphere,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PSO (V0 sequential + V1 threading + V2 multiprocessing) on benchmark functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem
    p.add_argument(
        "--objective", nargs="+",
        choices=list(OBJECTIVES), default=list(OBJECTIVES),
        metavar="FN",
        help="Objective function(s) to optimise.",
    )
    p.add_argument(
        "--dim", nargs="+", type=int, default=[2],
        metavar="D",
        help="Problem dimension(s). Multiple values run all combinations.",
    )
    p.add_argument(
        "--bounds-lo", type=float, default=-5.0,
        help="Lower bound (same for all dimensions).",
    )
    p.add_argument(
        "--bounds-hi", type=float, default=5.0,
        help="Upper bound (same for all dimensions).",
    )

    # Reproducibility
    p.add_argument(
        "--seed", nargs="+", type=int, default=[42],
        metavar="S",
        help="Random seed(s). Multiple values run all combinations.",
    )

    # Hyperparameters
    p.add_argument("--w",  type=float, default=None,  help="Inertia weight.")
    p.add_argument("--c1", type=float, default=None,  help="Cognitive coefficient.")
    p.add_argument("--c2", type=float, default=None,  help="Social coefficient.")
    p.add_argument("--n-particles", type=int, default=None, help="Swarm size.")
    p.add_argument("--max-iters",   type=int, default=None, help="Max iterations.")
    p.add_argument("--tol",         type=float, default=None, help="Convergence tolerance.")
    p.add_argument("--patience",    type=int, default=None, help="Early-stop patience.")
    p.add_argument("--vmax-ratio",  type=float, default=None,
                   help="Initial velocity cap as a fraction of the search range.")
    p.add_argument("--log-every",   type=int, default=10,
                   help="Log a per-iteration line every N iters (0 = off).")

    # Threading
    p.add_argument("--workers", type=int, default=None,
                   help="Max workers for ThreadPoolExecutor (None = auto).")
    p.add_argument("--process-workers", type=int, default=None,
                   help="Max workers for ProcessPoolExecutor (None = auto).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Particles per process task in V2 (None = auto).")

    # Grid search
    p.add_argument("--grid-search", action="store_true",
                   help="Run a lightweight grid search before the main run.")
    p.add_argument("--grid-strategy", choices=["v0", "v1", "v2"], default="v0",
                   help="Strategy used during the optional grid search.")
    p.add_argument("--grid-metric", choices=["final_fitness", "auc", "convergence_iter", "time_s"],
                   default="final_fitness",
                   help="Metric minimized during the optional grid search.")

    # Output
    p.add_argument("--out-dir",   default="results/runs",      help="Results directory.")
    p.add_argument("--plots-dir", default="logs/convergence",  help="Plots directory.")
    p.add_argument("--log-dir",   default="logs",              help="Log directory.")
    p.add_argument("--no-save",   action="store_true",         help="Skip saving files.")

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    total_runs = len(args.objective) * len(args.dim) * len(args.seed)
    print(
        f"\nRunning {total_runs} experiment(s): "
        f"{args.objective} × dim{args.dim} × seed{args.seed}\n"
    )

    for obj_name in args.objective:
        obj_fn = OBJECTIVES[obj_name]
        for dim in args.dim:
            for seed in args.seed:
                bounds = ([args.bounds_lo] * dim, [args.bounds_hi] * dim)
                cfg = RunConfig(
                    seed=seed,
                    dim=dim,
                    bounds=bounds,
                    w=args.w,
                    c1=args.c1,
                    c2=args.c2,
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
                    save_files=not args.no_save,
                )

                print(f"\n{'='*60}")
                print(f"  {obj_name.upper()}  |  d={dim}  |  seed={seed}")
                print(f"{'='*60}")
                run_one_objective(obj_fn, cfg)


if __name__ == "__main__":
    main()
