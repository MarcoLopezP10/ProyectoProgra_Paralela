"""scripts.make_viz

Generate swarm evolution animations for d=2 and d=3 objectives.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional

# Allow direct execution as `python scripts/make_viz.py` from editors like
# VS Code while keeping `python -m scripts.make_viz` working unchanged.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.pso import PSO
from core.swarm import Swarm
from experiment.run_single import RunConfig, _resolve_run_config
from objectives.ackley import ackley
from objectives.rastrigin import rastrigin
from objectives.rosenbrock import rosenbrock
from objectives.sphere import sphere
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from parallel.evaluator import build_evaluator
from utils.io import load_swarm_trajectory_npz, result_dir
from viz.swarm_animation import SwarmRecorder, recorder_from_trajectory, save_swarm_animation

OBJECTIVES = {
    "sphere": sphere,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PSO swarm animations for 2-D and 3-D benchmark functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--objective", nargs="+", choices=list(OBJECTIVES), default=list(OBJECTIVES), metavar="FN")
    p.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Animation dimension.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-particles", type=int, default=40, help="Swarm size for the animation run.")
    p.add_argument("--max-iters", type=int, default=150)
    p.add_argument("--bounds-lo", type=float, default=-5.0)
    p.add_argument("--bounds-hi", type=float, default=5.0)
    p.add_argument("--w", type=float, default=0.7)
    p.add_argument("--c1", type=float, default=1.5)
    p.add_argument("--c2", type=float, default=1.5)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--vmax-ratio", type=float, default=0.2)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--format", choices=["gif", "mp4"], default="gif")
    p.add_argument("--out-dir", default="logs/animations")
    p.add_argument("--resolution", type=int, default=120, help="Contour grid resolution for d=2.")
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames (subsample if needed).")
    p.add_argument("--strategy", choices=["v0", "v1", "v2", "v3", "v4"], default="v0",
                   help="Execution strategy whose real trajectory should be visualised.")
    p.add_argument("--results-dir", default="results/runs",
                   help="Directory containing persisted run results and trajectories.")
    p.add_argument("--rerun-if-missing", action="store_true",
                   help="Re-run the selected strategy if the persisted trajectory is missing.")
    p.add_argument("--trajectory-stride", type=int, default=1,
                   help="Sampling stride used when re-running to generate a fresh trajectory.")
    return p.parse_args(argv)


def run_with_recorder(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: tuple[list[float], list[float]],
    seed: int,
    n_particles: int,
    max_iters: int,
    w: float,
    c1: float,
    c2: float,
    tol: float,
    patience: int,
    vmax_ratio: float,
    strategy: str,
    trajectory_stride: int,
) -> tuple[SwarmRecorder, PSO]:
    """Run PSO and capture the real execution with a recorder callback."""
    rng = np.random.default_rng(seed)
    swarm = Swarm(
        n_particles=n_particles,
        dim=dim,
        bounds=bounds,
        rng=rng,
        vmax_ratio=vmax_ratio,
    )
    recorder = SwarmRecorder(stride=trajectory_stride)
    pso = PSO(
        swarm=swarm,
        evaluator=build_evaluator(strategy, objective_fn),
        bounds_handler=ClampBounds(bounds[0], bounds[1]),
        topology=GlobalBestTopology(),
        w=w,
        c1=c1,
        c2=c2,
        max_iters=max_iters,
        seed=seed,
        tol=tol,
        patience=patience,
        log_every=0,
        on_iteration=recorder.callback,
    )
    pso.run()
    if pso.iteration_records:
        recorder.ensure_final_state(
            pso.swarm,
            len(pso.history) - 1,
            pso.iteration_records[-1],
        )
    return recorder, pso


def _load_persisted_recorder(
    *,
    results_dir: str,
    objective_name: str,
    dim: int,
    seed: int,
    strategy: str,
) -> Optional[SwarmRecorder]:
    run_dir = result_dir(results_dir, objective_name, dim, seed)
    trajectory_path = os.path.join(run_dir, f"trajectory_{strategy}.npz")
    if not os.path.exists(trajectory_path):
        return None
    return recorder_from_trajectory(load_swarm_trajectory_npz(trajectory_path))


def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    bounds = ([args.bounds_lo] * args.dim, [args.bounds_hi] * args.dim)

    for obj_name in args.objective:
        obj_fn = OBJECTIVES[obj_name]
        print(
            f"\nAnimating {obj_name.upper()} (d={args.dim}, seed={args.seed}, strategy={args.strategy.upper()})..."
        )
        recorder = _load_persisted_recorder(
            results_dir=args.results_dir,
            objective_name=obj_name,
            dim=args.dim,
            seed=args.seed,
            strategy=args.strategy,
        )
        pso = None

        if recorder is None:
            if not args.rerun_if_missing:
                raise FileNotFoundError(
                    "Persisted trajectory not found. Re-run the experiment with "
                    "trajectory saving enabled or use --rerun-if-missing."
                )

            resolved = _resolve_run_config(
                RunConfig(
                    seed=args.seed,
                    dim=args.dim,
                    bounds=bounds,
                    w=args.w,
                    c1=args.c1,
                    c2=args.c2,
                    n_particles=args.n_particles,
                    max_iters=args.max_iters,
                    tol=args.tol,
                    patience=args.patience,
                    vmax_ratio=args.vmax_ratio,
                ),
                obj_name,
            )
            recorder, pso = run_with_recorder(
                objective_fn=obj_fn,
                dim=args.dim,
                bounds=bounds,
                seed=args.seed,
                n_particles=int(resolved["n_particles"]),
                max_iters=int(resolved["max_iters"]),
                w=float(resolved["w"]),
                c1=float(resolved["c1"]),
                c2=float(resolved["c2"]),
                tol=float(resolved["tol"]),
                patience=int(resolved["patience"]),
                vmax_ratio=float(resolved["vmax_ratio"]),
                strategy=args.strategy,
                trajectory_stride=args.trajectory_stride,
            )
            print(
                f"  Re-ran {args.strategy.upper()} to reconstruct the trajectory. "
                f"Best fitness={pso.history[-1]:.6e}"
            )
        else:
            best_fitness = recorder.fitness_history[-1] if recorder.fitness_history else float("nan")
            print(f"  Loaded persisted trajectory with {len(recorder)} frames. Best fitness={best_fitness:.6e}")

        out_path = os.path.join(
            args.out_dir,
            f"{obj_name}_{args.strategy}_d{args.dim}_s{args.seed}.{args.format}",
        )
        save_swarm_animation(
            recorder=recorder,
            objective_fn=obj_fn,
            bounds=bounds,
            out_path=out_path,
            title=f"{obj_name} - PSO swarm (d={args.dim}, seed={args.seed})",
            fps=args.fps,
            resolution=args.resolution,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
