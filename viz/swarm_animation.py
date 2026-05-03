"""viz.swarm_animation

Swarm evolution visualisation for d=2 and d=3.

- d=2: contour background + particles + global best + convergence curve
- d=3: 3D swarm scatter + global best + convergence curve
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


class SwarmRecorder:
    """Collect swarm snapshots during a PSO run."""

    def __init__(self, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = int(stride)
        self.iteration_numbers: List[int] = []
        self.positions: List[np.ndarray] = []
        self.global_bests: List[np.ndarray] = []
        self.fitness_history: List[float] = []
        self.iteration_metrics: List[dict[str, float]] = []

    def record(
        self,
        iteration: int,
        positions: List[np.ndarray],
        global_best_position: np.ndarray,
        global_best_fitness: float,
        iteration_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Store one snapshot."""
        self.iteration_numbers.append(int(iteration))
        self.positions.append(np.array(positions, dtype=float))
        self.global_bests.append(np.array(global_best_position, dtype=float))
        self.fitness_history.append(float(global_best_fitness))
        self.iteration_metrics.append(dict(iteration_metrics or {}))

    def callback(self, iteration: int, swarm, iteration_metrics: dict[str, float]) -> None:
        """PSO callback used to record the real run without duplicating the loop."""
        if iteration % self.stride != 0:
            return
        self.record(
            iteration=iteration,
            positions=swarm.get_positions(),
            global_best_position=swarm.global_best_position,
            global_best_fitness=swarm.global_best_fitness,
            iteration_metrics=iteration_metrics,
        )

    def ensure_final_state(self, swarm, iteration: int, iteration_metrics: dict[str, float]) -> None:
        """Append the final state when sampling skipped the last iteration."""
        if self.iteration_numbers and self.iteration_numbers[-1] == int(iteration):
            return
        self.record(
            iteration=iteration,
            positions=swarm.get_positions(),
            global_best_position=swarm.global_best_position,
            global_best_fitness=swarm.global_best_fitness,
            iteration_metrics=iteration_metrics,
        )

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def dimension(self) -> int:
        return int(self.positions[0].shape[1]) if self.positions else 0


def recorder_from_trajectory(payload: dict[str, np.ndarray]) -> SwarmRecorder:
    """Rebuild a recorder view from a persisted trajectory archive."""
    recorder = SwarmRecorder()
    iteration_numbers = payload.get("iteration_numbers", np.empty((0,), dtype=int))
    positions = payload.get("positions", np.empty((0, 0, 0), dtype=float))
    global_bests = payload.get("global_bests", np.empty((0, 0), dtype=float))
    fitness_history = payload.get("fitness_history", np.empty((0,), dtype=float))

    eval_s = payload.get("eval_s", np.empty((0,), dtype=float))
    update_s = payload.get("update_s", np.empty((0,), dtype=float))
    overhead_s = payload.get("overhead_s", np.empty((0,), dtype=float))
    iter_s = payload.get("iter_s", np.empty((0,), dtype=float))
    stall_count = payload.get("stall_count", np.empty((0,), dtype=float))

    n_frames = min(
        len(iteration_numbers),
        len(positions),
        len(global_bests),
        len(fitness_history),
    )
    for idx in range(n_frames):
        recorder.record(
            iteration=int(iteration_numbers[idx]),
            positions=np.asarray(positions[idx], dtype=float),
            global_best_position=np.asarray(global_bests[idx], dtype=float),
            global_best_fitness=float(fitness_history[idx]),
            iteration_metrics={
                "eval_s": float(eval_s[idx]) if idx < len(eval_s) else 0.0,
                "update_s": float(update_s[idx]) if idx < len(update_s) else 0.0,
                "overhead_s": float(overhead_s[idx]) if idx < len(overhead_s) else 0.0,
                "iter_s": float(iter_s[idx]) if idx < len(iter_s) else 0.0,
                "stall_count": float(stall_count[idx]) if idx < len(stall_count) else 0.0,
            },
        )
    return recorder


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
    Z = np.array([[fn(np.array([xi, yi], dtype=float)) for xi in x] for yi in y])
    return X, Y, Z


def _resolve_frame_indices(recorder: SwarmRecorder, max_frames: Optional[int]) -> List[int]:
    n_frames = len(recorder)
    if max_frames is None or n_frames <= max_frames:
        return list(range(n_frames))
    step = max(1, n_frames // max_frames)
    return list(range(0, n_frames, step))


def _configure_convergence_axis(ax_conv, recorder: SwarmRecorder, frame_indices: List[int]) -> None:
    ax_conv.set_title("Best fitness", fontsize=9)
    ax_conv.set_xlabel("Iteration", fontsize=8)
    ax_conv.set_ylabel("Fitness", fontsize=8)
    ax_conv.tick_params(labelsize=7)
    ax_conv.grid(True, linestyle="--", alpha=0.4)

    all_fitness = recorder.fitness_history
    if all_fitness and min(all_fitness) > 0:
        ax_conv.set_yscale("log")

    if recorder.iteration_numbers:
        ax_conv.set_xlim(0, recorder.iteration_numbers[frame_indices[-1]])
    finite_vals = [v for v in all_fitness if np.isfinite(v) and v > 0]
    if finite_vals:
        ax_conv.set_ylim(min(finite_vals) * 0.1, max(finite_vals) * 10)


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
    Generate and save an animated visualisation of swarm evolution for d=2 or d=3.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if len(recorder) == 0:
        raise ValueError("Recorder is empty. Run PSO with a recorder callback first.")
    if recorder.dimension not in {2, 3}:
        raise ValueError("save_swarm_animation only supports dimensions 2 and 3.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    frame_indices = _resolve_frame_indices(recorder, max_frames)
    lo, hi = bounds

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    if recorder.dimension == 3:
        ax_swarm = fig.add_subplot(gs[0], projection="3d")
    else:
        ax_swarm = fig.add_subplot(gs[0])
    ax_conv = fig.add_subplot(gs[1])
    fig.suptitle(title, fontsize=12)

    if recorder.dimension == 2:
        print(f"  Computing contour grid ({resolution}x{resolution})...", flush=True)
        X, Y, Z = _make_contour_grid(objective_fn, bounds, resolution)
        Z_log = np.log1p(np.abs(Z))
        ax_swarm.contourf(X, Y, Z_log, levels=contour_levels, cmap="viridis", alpha=0.6)
        ax_swarm.contour(
            X,
            Y,
            Z_log,
            levels=contour_levels,
            colors="white",
            linewidths=0.3,
            alpha=0.3,
        )
        ax_swarm.set_xlim(lo[0], hi[0])
        ax_swarm.set_ylim(lo[1], hi[1])
        ax_swarm.set_xlabel("x1", fontsize=10)
        ax_swarm.set_ylabel("x2", fontsize=10)
        ax_swarm.set_aspect("equal")
    else:
        ax_swarm.set_xlim(lo[0], hi[0])
        ax_swarm.set_ylim(lo[1], hi[1])
        ax_swarm.set_zlim(lo[2], hi[2])
        ax_swarm.set_xlabel("x1", fontsize=9)
        ax_swarm.set_ylabel("x2", fontsize=9)
        ax_swarm.set_zlabel("x3", fontsize=9)
        ax_swarm.view_init(elev=24, azim=38)

    _configure_convergence_axis(ax_conv, recorder, frame_indices)
    conv_line, = ax_conv.plot([], [], lw=1.5, color="tab:orange")
    conv_dot, = ax_conv.plot([], [], "o", color="red", ms=5, zorder=5)

    if recorder.dimension == 2:
        scat_particles = ax_swarm.scatter(
            [],
            [],
            s=18,
            c="white",
            alpha=0.75,
            edgecolors="none",
            zorder=3,
            label="Particles",
        )
        scat_best = ax_swarm.scatter(
            [],
            [],
            s=120,
            c="red",
            marker="*",
            zorder=5,
            label="Global best",
        )
        iter_text = ax_swarm.text(
            0.02,
            0.97,
            "",
            transform=ax_swarm.transAxes,
            fontsize=9,
            va="top",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5),
        )
    else:
        first_pos = recorder.positions[frame_indices[0]]
        first_best = recorder.global_bests[frame_indices[0]]
        scat_particles = ax_swarm.scatter(
            first_pos[:, 0],
            first_pos[:, 1],
            first_pos[:, 2],
            s=20,
            c="tab:blue",
            alpha=0.75,
            label="Particles",
        )
        scat_best = ax_swarm.scatter(
            [first_best[0]],
            [first_best[1]],
            [first_best[2]],
            s=140,
            c="red",
            marker="*",
            label="Global best",
        )
        iter_text = ax_swarm.text2D(
            0.02,
            0.97,
            "",
            transform=ax_swarm.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )

    ax_swarm.legend(loc="lower right", fontsize=8, framealpha=0.6)
    plt.tight_layout()

    def _update(fi: int) -> tuple[Any, ...]:
        real_idx = frame_indices[fi]
        iteration = recorder.iteration_numbers[real_idx]
        pos = recorder.positions[real_idx]
        gbest = recorder.global_bests[real_idx]
        fval = recorder.fitness_history[real_idx]

        if recorder.dimension == 2:
            scat_particles.set_offsets(pos[:, :2])
            scat_best.set_offsets(gbest[:2].reshape(1, 2))
        else:
            scat_particles._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            scat_best._offsets3d = ([gbest[0]], [gbest[1]], [gbest[2]])

        iter_text.set_text(f"iter {iteration}  f={fval:.3e}")

        xs = [recorder.iteration_numbers[frame_indices[k]] for k in range(fi + 1)]
        ys = [recorder.fitness_history[frame_indices[k]] for k in range(fi + 1)]
        conv_line.set_data(xs, ys)
        conv_dot.set_data([iteration], [fval])

        return scat_particles, scat_best, iter_text, conv_line, conv_dot

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=False,
    )

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=800)
        anim.save(out_path, writer=writer, dpi=120)
    else:
        anim.save(out_path, writer="pillow", fps=fps, dpi=100)

    plt.close(fig)
    print(f"  Saved animation -> {out_path}")
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
    """Save a static 2-D frame of the swarm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if positions.shape[1] != 2:
        raise ValueError("save_swarm_frame only supports 2-D data.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lo, hi = bounds

    X, Y, Z = _make_contour_grid(objective_fn, bounds, resolution)
    Z_log = np.log1p(np.abs(Z))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(X, Y, Z_log, levels=contour_levels, cmap="viridis", alpha=0.65)
    ax.contour(X, Y, Z_log, levels=contour_levels, colors="white", linewidths=0.3, alpha=0.3)
    ax.scatter(positions[:, 0], positions[:, 1], s=20, c="white", alpha=0.8, edgecolors="none")
    ax.scatter(
        *global_best[:2],
        s=140,
        c="red",
        marker="*",
        zorder=5,
        label=f"Best (f={best_fitness:.3e})",
    )
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{title}  iter={iteration}" if title else f"iter={iteration}", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.6)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
