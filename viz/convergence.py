"""viz.convergence

Minimal visualization for V0:
- Convergence curve (best fitness vs iteration)
- Horizontal reference line for the baseline (PySwarm)

(Swarm animation for d=2/d=3 can be added in a later version.)
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt


def save_convergence_plot(
    history: List[float],
    baseline_final_fitness: float,
    title: str,
    out_path: str,
    y_log_if_possible: bool = True
) -> None:
    """Guarda un plot simple de convergencia."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(history, linewidth=2.5, label="Custom PSO (Sequential)")
    plt.axhline(
        y=baseline_final_fitness,
        linestyle="--",
        linewidth=2.5,
        label="PySwarm Final Fitness",
    )

    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Best Fitness", fontsize=11)
    plt.title(title, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    #Only uso log if there at not any 0's
    if y_log_if_possible and history and min(history) > 0 and baseline_final_fitness > 0:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
