"""scripts.run_edl

CLI entry point for Economic Load Dispatch runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.run_edl import EDLRunConfig, EDL_VARIANTS, run_edl_suite


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PSO-based Economic Load Dispatch variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        default="cases/edl_case_3u.json",
        help="Path to the EDL case JSON file.",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=list(EDL_VARIANTS),
        default=list(EDL_VARIANTS),
        metavar="EDL",
        help="EDL variant(s) to execute.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[42],
        metavar="S",
        help="Random seed(s).",
    )
    parser.add_argument("--w", type=float, default=0.6, help="Inertia weight.")
    parser.add_argument("--c1", type=float, default=1.4, help="Cognitive coefficient.")
    parser.add_argument("--c2", type=float, default=1.6, help="Social coefficient.")
    parser.add_argument("--n-particles", type=int, default=80, help="Swarm size.")
    parser.add_argument("--max-iters", type=int, default=500, help="Maximum iterations.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance.")
    parser.add_argument("--patience", type=int, default=80, help="Early-stop patience.")
    parser.add_argument(
        "--vmax-ratio",
        type=float,
        default=0.2,
        help="Initial velocity cap as a fraction of the search range.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max workers for V1 threading.",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=None,
        help="Max workers for V2 multiprocessing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Particles per process task in V2.",
    )
    parser.add_argument(
        "--penalty-power-balance",
        type=float,
        default=1e6,
        help="Quadratic penalty coefficient for the power-balance constraint.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/edl",
        help="Directory for EDL output files.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for EDL logs.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing JSON/CSV outputs to disk.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = EDLRunConfig(
        case_path=args.case,
        seeds=list(args.seed),
        variants=list(args.variant),
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        n_particles=args.n_particles,
        max_iters=args.max_iters,
        tol=args.tol,
        patience=args.patience,
        vmax_ratio=args.vmax_ratio,
        thread_max_workers=args.workers,
        process_max_workers=args.process_workers,
        batch_size=args.batch_size,
        penalty_power_balance=args.penalty_power_balance,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        save_files=not args.no_save,
    )
    run_edl_suite(cfg)


if __name__ == "__main__":
    main()
