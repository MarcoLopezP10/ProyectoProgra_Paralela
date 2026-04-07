from __future__ import annotations

import json
import os
import sys

import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.io import (
    ExecutionMetadata,
    ExperimentSummary,
    MethodResult,
    TimingBreakdown,
    save_history_csv,
    save_iteration_metrics_csv,
    save_summary_json,
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
    )
    save_summary_json(summary, str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["execution"]["python_version"] == "3.11"
    assert data["v0"]["auc"] == 1.23


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
