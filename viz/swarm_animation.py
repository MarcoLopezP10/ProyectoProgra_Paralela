"""viz.swarm_animation

Swarm evolution visualisation for d=2.

Produces:
  - An animated GIF/MP4 showing particle positions at each iteration.
  - The objective function contour as background.
  - The global best position highlighted at every frame.
  - A small inset with the convergence curve (best fitness vs iteration).

Usage (standalone)
------------------
from viz.swarm_animation import SwarmRecorder, save_swarm_animation
from objectives.sphere import sphere

recorder = SwarmRecorder()          # attach to a PSO run
...                                 # run PSO calling recorder.record() each iter
save_swarm_animation(recorder, sphere, bounds=([-5]*2,[5]*2), out_path="swarm.gif")

Integration with PSO
--------------------
Attach a SwarmRecorder before calling pso.run() and pass record() as a
per-iteration callback — or record snapshots manually in a custom loop.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np

# Matplotlib is imported lazily so the module is importable even without a
# display (e.g. on a headless server with Agg backend).


class SwarmRecorder:
    """
    Collects swarm snapshots during a PSO run.

    Call record() once per iteration to store particle positions,
    the current global best, and the best fitness value.
    """

    def __init__(self) -> None:
        self.positions: List[np.ndarray] = []      # shape (n_particles, dim) per frame
        self.global_bests: List[np.ndarray] = []   # shape (dim,) per frame
        self.fitness_history: List[float] = []     # scalar per frame

    def record(
        self,
        positions: List[np.ndarray],
        global_best_position: np.ndarray,
        global_best_fitness: float,
    ) -> None:
        """Store one snapshot."""
        self.positions.append(np.array(positions))          # copy
        self.global_bests.append(np.array(global_best_position))
        self.fitness_history.append(float(global_best_fitness))

    def __len__(self) -> int:
        return len(self.positions)


def _make_contour_grid(
    fn: Callable[[np.ndarray], float],
    bounds: Tuple[List[float], List[float]],
    resolution: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) arrays for a 2-D contour plot of fn."""
    lo, hi = bounds
    x = np.linspace(lo[0], hi[0], resolution)
    y = np.linspace(lo[1], hi[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[fn(np.array([xi, yi])) for xi in x] for yi in y])
    return X, Y, Z


def save_swarm_animation(
    recorder: SwarmRecorder,
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[List[float], List[float]],
    out_path: str = "swarm_animation.gif",
    title: str = "PSO swarm evolution",
    fps: int = 6,
    resolution: int = 150,
    figsize: Tuple[float, float] = (10, 4.5),
    max_frames: Optional[int] = None,
    contour_levels: int = 25,
) -> str:
    """
    Generate and save an animated visualisation of swarm evolution.

    Parameters
    ----------
    recorder : SwarmRecorder
        Filled recorder from a PSO run.
    objective_fn : Callable
        The objective function (used for the contour background).
    bounds : tuple
        (lower, upper) as lists of floats, one per dimension.
    out_path : str
        Output file path. Use .gif for GIF, .mp4 for video.
    title : str
        Animation title.
    fps : int
        Frames per second.
    resolution : int
        Grid resolution for the contour plot (higher = slower).
    figsize : tuple
        Matplotlib figure size.
    max_frames : int | None
        Cap the number of frames (useful for large swarms).
    contour_levels : int
        Number of contour levels.

    Returns
    -------
    str  The output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    n_frames = len(recorder)
    if max_frames is not None:
        step = max(1, n_frames // max_frames)
        frame_indices = list(range(0, n_frames, step))
    else:
        frame_indices = list(range(n_frames))

    lo, hi = bounds

    # ── Pre-compute contour grid ───────────────────────────────────────
    print(f"  Computing contour grid ({resolution}×{resolution})…", flush=True)
    X, Y, Z = _make_contour_grid(objective_fn, bounds, resolution)
    Z_log = np.log1p(np.abs(Z))   # log scale makes contours easier to read

    # ── Figure layout: swarm on left, convergence curve on right ──────
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    ax_swarm = fig.add_subplot(gs[0])
    ax_conv  = fig.add_subplot(gs[1])

    fig.suptitle(title, fontsize=12)

    # Static contour (drawn once)
    ax_swarm.contourf(X, Y, Z_log, levels=contour_levels, cmap="viridis", alpha=0.6)
    ax_swarm.contour(X, Y, Z_log, levels=contour_levels, colors="white",
                     linewidths=0.3, alpha=0.3)
    ax_swarm.set_xlim(lo[0], hi[0])
    ax_swarm.set_ylim(lo[1], hi[1])
    ax_swarm.set_xlabel("x₁", fontsize=10)
    ax_swarm.set_ylabel("x₂", fontsize=10)
    ax_swarm.set_aspect("equal")

    # Dynamic artists
    scat_particles = ax_swarm.scatter([], [], s=18, c="white", alpha=0.75,
                                      edgecolors="none", zorder=3, label="Particles")
    scat_best      = ax_swarm.scatter([], [], s=120, c="red", marker="*",
                                      zorder=5, label="Global best")
    iter_text      = ax_swarm.text(
        0.02, 0.97, "", transform=ax_swarm.transAxes,
        fontsize=9, va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5),
    )
    ax_swarm.legend(loc="lower right", fontsize=8, framealpha=0.6)

    # Convergence inset
    ax_conv.set_title("Best fitness", fontsize=9)
    ax_conv.set_xlabel("Iteration", fontsize=8)
    ax_conv.set_ylabel("Fitness", fontsize=8)
    ax_conv.tick_params(labelsize=7)
    ax_conv.grid(True, linestyle="--", alpha=0.4)
    all_fitness = recorder.fitness_history
    # Use log scale if all values positive
    if min(all_fitness) > 0:
        ax_conv.set_yscale("log")
    conv_line, = ax_conv.plot([], [], lw=1.5, color="tab:orange")
    conv_dot,  = ax_conv.plot([], [], "o", color="red", ms=5, zorder=5)
    ax_conv.set_xlim(0, len(frame_indices) - 1)
    finite_vals = [v for v in all_fitness if np.isfinite(v) and v > 0]
    if finite_vals:
        ax_conv.set_ylim(min(finite_vals) * 0.1, max(finite_vals) * 10)

    plt.tight_layout()

    def _update(fi: int):
        real_it = frame_indices[fi]
        pos   = recorder.positions[real_it]       # (n_particles, 2)
        gbest = recorder.global_bests[real_it]    # (2,)
        fval  = recorder.fitness_history[real_it]

        scat_particles.set_offsets(pos[:, :2])
        scat_best.set_offsets(gbest[:2].reshape(1, 2))
        iter_text.set_text(f"iter {real_it}  f={fval:.3e}")

        # Convergence curve up to current frame
        xs = list(range(fi + 1))
        ys = [recorder.fitness_history[frame_indices[k]] for k in range(fi + 1)]
        conv_line.set_data(xs, ys)
        conv_dot.set_data([fi], [fval])

        return scat_particles, scat_best, iter_text, conv_line, conv_dot

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=True,
    )

    # ── Save ──────────────────────────────────────────────────────────
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=800)
        anim.save(out_path, writer=writer, dpi=120)
    else:
        # Default: GIF via Pillow
        anim.save(out_path, writer="pillow", fps=fps, dpi=100)

    plt.close(fig)
    print(f"  Saved animation → {out_path}")
    return out_path


def save_swarm_frame(
    positions: np.ndarray,
    global_best: np.ndarray,
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[List[float], List[float]],
    iteration: int,
    best_fitness: float,
    out_path: str,
    title: str = "",
    resolution: int = 150,
    contour_levels: int = 25,
) -> None:
    """
    Save a single static frame of the swarm at a given iteration.
    Useful for quick inspection without generating a full animation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lo, hi = bounds

    X, Y, Z = _make_contour_grid(objective_fn, bounds, resolution)
    Z_log = np.log1p(np.abs(Z))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(X, Y, Z_log, levels=contour_levels, cmap="viridis", alpha=0.65)
    ax.contour(X, Y, Z_log, levels=contour_levels, colors="white",
               linewidths=0.3, alpha=0.3)
    ax.scatter(positions[:, 0], positions[:, 1], s=20, c="white",
               alpha=0.8, edgecolors="none", label="Particles")
    ax.scatter(*global_best[:2], s=140, c="red", marker="*",
               zorder=5, label=f"Best (f={best_fitness:.3e})")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"{title}  iter={iteration}" if title else f"iter={iteration}", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.6)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)