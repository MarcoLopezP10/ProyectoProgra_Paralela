from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_benchmarks import (
    _resolve_suite_paths,
    _validate_report_ready,
    parse_args,
)


def test_suite_name_rewrites_default_output_paths():
    args = parse_args(["--suite-name", "final_report"])

    _resolve_suite_paths(args)

    assert args.out_dir.endswith("results/benchmark_suites/final_report/runs")
    assert args.plots_dir.endswith("results/benchmark_suites/final_report/plots/convergence")
    assert args.log_dir.endswith("results/benchmark_suites/final_report/logs")
    assert args.summary_csv.endswith("results/benchmark_suites/final_report/benchmark_summary.csv")


def test_report_ready_requires_core_objectives_dims_and_seeds():
    args = parse_args([
        "--objective", "sphere", "ackley", "rosenbrock", "rastrigin",
        "--dims", "2", "10", "30",
        "--seeds", "0", "1", "7", "42", "123",
        "--report-ready",
    ])

    _validate_report_ready(args)


def test_report_ready_rejects_incomplete_protocol():
    args = parse_args([
        "--objective", "sphere", "ackley",
        "--dims", "2", "10",
        "--seeds", "42", "7",
        "--report-ready",
    ])

    with pytest.raises(ValueError):
        _validate_report_ready(args)
