"""scripts.make_viz

Generate swarm evolution animations for d=2 objectives.

Runs a PSO with a SwarmRecorder attached to capture every iteration,
then renders a GIF animation with:
  - Particle positions on the objective function contour
  - Global best position highlighted
  - Convergence curve inset

Usage examples
--------------
# All 4 objectives, default settings
python -m scripts.make_viz

# Single objective
python -m scripts.make_viz --objective sphere

# Custom seed and iteration count
python -m scripts.make_viz --objective ackley --seed 7 --max-iters 300

# Save as MP4 instead of GIF (requires ffmpeg)
python -m scripts.make_viz --format mp4
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from core.swarm import Swarm
from core.pso import PSO
from options.bounds import ClampBounds
from options.evaluator import SequentialEvaluator
from options.topology import GlobalBestTopology
from viz.swarm_animation import SwarmRecorder, save_swarm_animation
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate PSO swarm animations for 2-D benchmark functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--objective", nargs="+", choices=list(OBJECTIVES),
                   default=list(OBJECTIVES), metavar="FN",
                   help="Objective function(s) to animate.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n-particles",type=int,   default=40,
                   help="Swarm size for the animation run.")
    p.add_argument("--max-iters",  type=int,   default=150)
    p.add_argument("--bounds-lo",  type=float, default=-5.0)
    p.add_argument("--bounds-hi",  type=float, default=5.0)
    p.add_argument("--w",          type=float, default=0.7)
    p.add_argument("--c1",         type=float, default=1.5)
    p.add_argument("--c2",         type=float, default=1.5)
    p.add_argument("--vmax-ratio", type=float, default=0.2)
    p.add_argument("--fps",        type=int,   default=6)
    p.add_argument("--format",     choices=["gif", "mp4"], default="gif")
    p.add_argument("--out-dir",    default="logs/animations")
    p.add_argument("--resolution", type=int, default=120,
                   help="Contour grid resolution (higher = slower).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames (subsample if run has more iterations).")
    return p.parse_args(argv)


def run_with_recorder(
    objective_fn,
    bounds,
    seed: int,
    n_particles: int,
    max_iters: int,
    w: float,
    c1: float,
    c2: float,
    vmax_ratio: float,
) -> SwarmRecorder:
    """
    Run a PSO and record every iteration into a SwarmRecorder.

    Because PSO.run() is a closed loop, we implement a lightweight
    manual loop here that mirrors the core logic exactly and records
    snapshots at each step.
    """
    lo, hi = bounds
    rng = np.random.default_rng(seed)
    swarm = Swarm(
        n_particles=n_particles,
        dim=2,
        bounds=bounds,
        rng=rng,
        vmax_ratio=vmax_ratio,
    )
    evaluator  = SequentialEvaluator(objective_fn)
    bounds_h   = ClampBounds(lo, hi)
    topology   = GlobalBestTopology()
    recorder   = SwarmRecorder()

    prev_best   = np.inf
    no_improve  = 0
    tol         = 1e-10
    patience    = 30

    for _ in range(max_iters):
        positions = swarm.get_positions()
        fitness   = evaluator.evaluate(positions)
        swarm.update_global_best(positions, fitness)

        # Record snapshot AFTER updating bests
        recorder.record(
            positions=swarm.get_positions(),
            global_best_position=swarm.global_best_position.copy(),
            global_best_fitness=swarm.global_best_fitness,
        )

        improvement = abs(prev_best - swarm.global_best_fitness)
        if improvement < tol:
            no_improve += 1
        else:
            no_improve = 0
        if no_improve >= patience:
            break
        prev_best = swarm.global_best_fitness

        for p in swarm.particles:
            best_pos = topology.get_best_position(p, swarm)
            p.update_velocity(best_pos, w, c1, c2)
            p.update_position()
            p.position, p.velocity = bounds_h.apply(p.position, p.velocity)

    return recorder


def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    bounds = ([args.bounds_lo] * 2, [args.bounds_hi] * 2)

    for obj_name in args.objective:
        obj_fn = OBJECTIVES[obj_name]
        print(f"\nAnimating {obj_name.upper()} (seed={args.seed}, n={args.n_particles})…")

        recorder = run_with_recorder(
            objective_fn=obj_fn,
            bounds=bounds,
            seed=args.seed,
            n_particles=args.n_particles,
            max_iters=args.max_iters,
            w=args.w,
            c1=args.c1,
            c2=args.c2,
            vmax_ratio=args.vmax_ratio,
        )
        print(f"  Recorded {len(recorder)} frames.")

        out_path = os.path.join(
            args.out_dir,
            f"{obj_name}_d2_s{args.seed}.{args.format}",
        )
        save_swarm_animation(
            recorder=recorder,
            objective_fn=obj_fn,
            bounds=bounds,
            out_path=out_path,
            title=f"{obj_name} — PSO swarm (seed={args.seed})",
            fps=args.fps,
            resolution=args.resolution,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
