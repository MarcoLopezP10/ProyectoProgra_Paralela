from __future__ import annotations

import json
import os
import sys

import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.io import (
    ExecutionMetadata,
    ExperimentSummary,
    load_swarm_trajectory_npz,
    MethodResult,
    TimingBreakdown,
    save_history_csv,
    save_iteration_metrics_csv,
    save_summary_json,
    save_swarm_trajectory_npz,
)
from utils.logger import setup_logger


def test_save_iteration_metrics_csv(tmp_path):
    out_path = tmp_path / "history_v0.csv"
    save_iteration_metrics_csv(
        [
            {
                "iter": 0,
                "best_fitness": 1.0,
                "eval_s": 0.1,
                "update_s": 0.2,
                "overhead_s": 0.0,
                "iter_s": 0.3,
                "stall_count": 0,
            }
        ],
        str(out_path),
    )
    content = out_path.read_text(encoding="utf-8")
    assert "best_fitness" in content
    assert "eval_s" in content


def test_save_summary_json_includes_execution_metadata(tmp_path):
    out_path = tmp_path / "summary.json"
    summary = ExperimentSummary(
        objective="sphere",
        dim=2,
        bounds_lower=[-5.0, -5.0],
        bounds_upper=[5.0, 5.0],
        seed=42,
        hyperparam_source="fixed_config",
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_particles=30,
        max_iters=100,
        tol=1e-8,
        patience=20,
        v0=MethodResult(
            strategy="V0 Sequential",
            best_fitness=0.0,
            iterations=10,
            auc=1.23,
            convergence_iteration=8,
            timing=TimingBreakdown(total_s=1.0),
        ),
        execution=ExecutionMetadata(python_version="3.11"),
        winner="V0 Sequential",
        winner_internal="V0 Sequential",
        winner_overall="PySwarm",
        baseline_reference="PySwarm baseline improves on the best internal fitness by 1.0e-03.",
    )
    save_summary_json(summary, str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["execution"]["python_version"] == "3.11"
    assert data["v0"]["auc"] == 1.23
    assert data["winner_internal"] == "V0 Sequential"
    assert data["winner_overall"] == "PySwarm"


def test_save_summary_json_preserves_unavailable_method_fields(tmp_path):
    out_path = tmp_path / "summary.json"
    summary = ExperimentSummary(
        objective="sphere",
        dim=2,
        bounds_lower=[-5.0, -5.0],
        bounds_upper=[5.0, 5.0],
        seed=42,
        hyperparam_source="fixed_config",
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_particles=30,
        max_iters=100,
        tol=1e-8,
        patience=20,
        v0=MethodResult(strategy="V0 Sequential", best_fitness=0.0, iterations=10),
        v2=MethodResult(
            status="unavailable",
            strategy="V2 Multiprocessing",
            error="PermissionError: sandbox blocked multiprocessing",
        ),
    )

    save_summary_json(summary, str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["v2"]["status"] == "unavailable"
    assert data["v2"]["best_fitness"] is None
    assert "sandbox blocked multiprocessing" in data["v2"]["error"]


def test_save_summary_json_includes_v4_results(tmp_path):
    out_path = tmp_path / "summary.json"
    summary = ExperimentSummary(
        objective="sphere",
        dim=2,
        bounds_lower=[-5.0, -5.0],
        bounds_upper=[5.0, 5.0],
        seed=42,
        hyperparam_source="fixed_config",
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_particles=30,
        max_iters=100,
        tol=1e-8,
        patience=20,
        v0=MethodResult(strategy="V0 Sequential", best_fitness=0.0, iterations=10),
        v4=MethodResult(
            strategy="V4 Vectorized",
            best_fitness=0.0,
            iterations=10,
            auc=1.11,
            convergence_iteration=7,
            timing=TimingBreakdown(total_s=0.5, eval_s=0.2, update_s=0.2, overhead_s=0.1),
        ),
    )

    save_summary_json(summary, str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["v4"]["strategy"] == "V4 Vectorized"
    assert data["v4"]["auc"] == 1.11


def test_save_summary_json_supports_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary = ExperimentSummary(
        objective="sphere",
        dim=2,
        bounds_lower=[-5.0, -5.0],
        bounds_upper=[5.0, 5.0],
        seed=42,
        hyperparam_source="fixed_config",
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_particles=30,
        max_iters=100,
        tol=1e-8,
        patience=20,
    )

    save_summary_json(summary, "summary.json")

    assert (tmp_path / "summary.json").exists()


def test_save_and_load_swarm_trajectory_npz(tmp_path):
    out_path = tmp_path / "trajectory_v0.npz"
    save_swarm_trajectory_npz(
        iteration_numbers=[0, 2],
        positions=[
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            np.array([[0.5, 1.5], [2.5, 3.5]], dtype=float),
        ],
        global_bests=[
            np.array([1.0, 2.0], dtype=float),
            np.array([0.5, 1.5], dtype=float),
        ],
        fitness_history=[5.0, 2.5],
        iteration_records=[
            {"eval_s": 0.1, "update_s": 0.2, "overhead_s": 0.01, "iter_s": 0.31, "stall_count": 0},
            {"eval_s": 0.08, "update_s": 0.18, "overhead_s": 0.02, "iter_s": 0.28, "stall_count": 1},
        ],
        out_path=str(out_path),
    )

    payload = load_swarm_trajectory_npz(str(out_path))

    np.testing.assert_array_equal(payload["iteration_numbers"], np.array([0, 2]))
    assert payload["positions"].shape == (2, 2, 2)
    np.testing.assert_allclose(payload["global_bests"][1], np.array([0.5, 1.5]))
    np.testing.assert_allclose(payload["fitness_history"], np.array([5.0, 2.5]))
    np.testing.assert_allclose(payload["stall_count"], np.array([0.0, 1.0]))


def test_save_csv_helpers_support_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    save_history_csv([1.0, 0.5], "history.csv")
    save_iteration_metrics_csv(
        [
            {
                "iter": 0,
                "best_fitness": 1.0,
                "eval_s": 0.1,
                "update_s": 0.2,
                "overhead_s": 0.0,
                "iter_s": 0.3,
                "stall_count": 0,
            }
        ],
        "metrics.csv",
    )

    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "metrics.csv").exists()


def test_setup_logger_reconfigures_file_handler(tmp_path):
    name = "pso_logger_reconfigure_test"
    first_dir = tmp_path / "logs_a"
    second_dir = tmp_path / "logs_b"

    logger = setup_logger(name=name, log_dir=str(first_dir), log_file="a.log")
    logger.info("first message")
    for handler in logger.handlers:
        handler.flush()

    logger = setup_logger(name=name, log_dir=str(second_dir), log_file="b.log")
    logger.info("second message")
    for handler in logger.handlers:
        handler.flush()

    first_log = (first_dir / "a.log").read_text(encoding="utf-8")
    second_log = (second_dir / "b.log").read_text(encoding="utf-8")

    assert "first message" in first_log
    assert "second message" not in first_log
    assert "second message" in second_log

    logging.getLogger(name).handlers.clear()
