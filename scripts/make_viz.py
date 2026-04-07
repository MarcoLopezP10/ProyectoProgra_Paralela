"""scripts.make_viz

Generate swarm evolution animations for d=2 and d=3 objectives.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

# Allow direct execution as `python scripts/make_viz.py` from editors like
# VS Code while keeping `python -m scripts.make_viz` working unchanged.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.pso import PSO
from core.swarm import Swarm
from objectives.ackley import ackley
from objectives.rastrigin import rastrigin
from objectives.rosenbrock import rosenbrock
from objectives.sphere import sphere
from options.bounds import ClampBounds
from options.evaluator import SequentialEvaluator
from options.topology import GlobalBestTopology
from viz.swarm_animation import SwarmRecorder, save_swarm_animation

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
    recorder = SwarmRecorder()
    pso = PSO(
        swarm=swarm,
        evaluator=SequentialEvaluator(objective_fn),
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
    return recorder, pso


def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    bounds = ([args.bounds_lo] * args.dim, [args.bounds_hi] * args.dim)

    for obj_name in args.objective:
        obj_fn = OBJECTIVES[obj_name]
        print(
            f"\nAnimating {obj_name.upper()} (d={args.dim}, seed={args.seed}, n={args.n_particles})..."
        )

        recorder, pso = run_with_recorder(
            objective_fn=obj_fn,
            dim=args.dim,
            bounds=bounds,
            seed=args.seed,
            n_particles=args.n_particles,
            max_iters=args.max_iters,
            w=args.w,
            c1=args.c1,
            c2=args.c2,
            tol=args.tol,
            patience=args.patience,
            vmax_ratio=args.vmax_ratio,
        )
        print(f"  Recorded {len(recorder)} frames. Best fitness={pso.history[-1]:.6e}")

        out_path = os.path.join(
            args.out_dir,
            f"{obj_name}_d{args.dim}_s{args.seed}.{args.format}",
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
