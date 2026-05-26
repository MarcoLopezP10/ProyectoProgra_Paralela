from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiment.grid_search import SUPPORTED_GRID_METRICS, grid_search
from objectives.sphere import sphere


def test_supported_grid_metrics_are_exposed():
    assert {"final_fitness", "auc", "convergence_iter", "time_s"} == SUPPORTED_GRID_METRICS


def test_grid_search_returns_metric_and_strategy_fields():
    best_params, rows = grid_search(
        objective_fn=sphere,
        dim=2,
        bounds=([-5.0, -5.0], [5.0, 5.0]),
        w_values=[0.6],
        c1_values=[1.5],
        c2_values=[1.5],
        n_particles_values=[10],
        max_iters_values=[15],
        seeds=[42],
        max_iters=15,
        strategy="v0",
        metric="time_s",
        verbose=False,
    )

    assert rows
    assert rows[0]["strategy"] == "v0"
    assert rows[0]["selected_metric"] == "time_s"
    assert "mean_time_s" in rows[0]
    assert "mean_auc" in rows[0]
    assert rows[0]["max_iters"] == 15
    assert best_params["max_iters"] == 15
    assert best_params["selected_metric"] == "time_s"


def test_grid_search_accepts_v3_strategy():
    best_params, rows = grid_search(
        objective_fn=sphere,
        dim=2,
        bounds=([-5.0, -5.0], [5.0, 5.0]),
        w_values=[0.6],
        c1_values=[1.5],
        c2_values=[1.5],
        n_particles_values=[10],
        max_iters_values=[10],
        seeds=[42],
        max_iters=10,
        strategy="v3",
        metric="final_fitness",
        verbose=False,
    )

    assert rows
    assert rows[0]["strategy"] == "v3"
    assert best_params["strategy"] == "v3"


def test_grid_search_accepts_v4_strategy():
    best_params, rows = grid_search(
        objective_fn=sphere,
        dim=2,
        bounds=([-5.0, -5.0], [5.0, 5.0]),
        w_values=[0.6],
        c1_values=[1.5],
        c2_values=[1.5],
        n_particles_values=[10],
        max_iters_values=[10],
        seeds=[42],
        max_iters=10,
        strategy="v4",
        metric="final_fitness",
        verbose=False,
    )

    assert rows
    assert rows[0]["strategy"] == "v4"
    assert best_params["strategy"] == "v4"


def test_grid_search_can_rank_multiple_iteration_budgets():
    best_params, rows = grid_search(
        objective_fn=sphere,
        dim=2,
        bounds=([-5.0, -5.0], [5.0, 5.0]),
        w_values=[0.6],
        c1_values=[1.5],
        c2_values=[1.5],
        n_particles_values=[10],
        max_iters_values=[8, 12],
        seeds=[42],
        max_iters=12,
        strategy="v0",
        metric="final_fitness",
        verbose=False,
    )

    assert {row["max_iters"] for row in rows} == {8, 12}
    assert best_params["max_iters"] == rows[0]["max_iters"]
