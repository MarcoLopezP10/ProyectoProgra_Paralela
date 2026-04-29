"""viz.convergence

Minimal visualization for V0/V1/V2/V3:
- Convergence curve (best fitness vs iteration)
- Horizontal reference line for the baseline (PySwarm)

(Swarm animation for d=2/d=3 can be added in a later version.)
"""

from __future__ import annotations

import os
from typing import List, Optional


def save_convergence_plot(
    history: List[float],
    baseline_final_fitness: float,
    title: str,
    out_path: str,
    threaded_history: Optional[List[float]] = None,
    process_history: Optional[List[float]] = None,
    asyncio_history: Optional[List[float]] = None,
    y_log_if_possible: bool = True
) -> None:
    """Save a polished convergence plot for V0/V1/V2/V3 comparisons."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.set_facecolor("#fbfbfd")

    # Use distinct styles so overlapping histories can still be identified.
    ax.plot(
        history,
        linewidth=2.8,
        color="#1f77b4",
        linestyle="-",
        label="V0: Sequential",
        zorder=2,
    )
    if threaded_history:
        ax.plot(
            threaded_history,
            linewidth=2.3,
            color="#ff7f0e",
            linestyle="--",
            label="V1: Threading",
            zorder=3,
        )
    if process_history:
        ax.plot(
            process_history,
            linewidth=2.3,
            color="#2ca02c",
            linestyle=":",
            marker="o",
            markersize=3.5,
            markevery=max(1, len(process_history) // 12),
            label="V2: Multiprocessing",
            zorder=4,
        )
    if asyncio_history:
        ax.plot(
            asyncio_history,
            linewidth=2.3,
            color="#e45756",
            linestyle="-.",
            marker="D",
            markersize=3.2,
            markevery=max(1, len(asyncio_history) // 12),
            label="V3: Asyncio",
            zorder=5,
        )
    ax.axhline(
        y=baseline_final_fitness,
        linestyle="--",
        linewidth=2.0,
        color="#7f7f7f",
        alpha=0.85,
        label="PySwarm: final fitness reference",
    )

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Use log scale only when every visible value is strictly positive.
    positive_values = list(history)
    if threaded_history:
        positive_values.extend(threaded_history)
    if process_history:
        positive_values.extend(process_history)
    if asyncio_history:
        positive_values.extend(asyncio_history)

    if (
        y_log_if_possible
        and positive_values
        and min(positive_values) > 0
        and baseline_final_fitness > 0
    ):
        ax.set_yscale("log")

    ax.legend(frameon=True, facecolor="white", edgecolor="#d9d9e3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
