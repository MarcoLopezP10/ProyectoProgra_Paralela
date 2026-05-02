from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from experiment.run_single import RunConfig, run_one_objective
from objectives.sphere import sphere
from parallel.evaluator import ProcessPoolEvaluator


def test_bounds_for_dim_preserves_explicit_per_dimension_values():
    cfg = RunConfig(
        dim=3,
        bounds=([-5.0, -1.0, 2.0], [5.0, 1.0, 3.0]),
    )

    lower, upper = cfg.bounds_for_dim()

    assert lower == [-5.0, -1.0, 2.0]
    assert upper == [5.0, 1.0, 3.0]


def test_bounds_for_dim_rejects_ambiguous_non_uniform_shapes():
    cfg = RunConfig(
        dim=3,
        bounds=([-5.0, -1.0], [5.0, 1.0]),
    )

    with pytest.raises(ValueError, match="one lower/upper value per dimension"):
        cfg.bounds_for_dim()


def test_run_one_objective_marks_v2_and_baseline_unavailable(monkeypatch, tmp_path):
    original_open = ProcessPoolEvaluator.open

    def fake_open(self) -> None:
        raise PermissionError("sandbox blocked multiprocessing")

    def fake_baseline(**kwargs):
        raise ModuleNotFoundError("No module named 'pyswarm'")

    monkeypatch.setattr(ProcessPoolEvaluator, "open", fake_open)
    monkeypatch.setattr("experiment.run_single.run_pyswarm_baseline", fake_baseline)
    try:
        result = run_one_objective(
            sphere,
            RunConfig(
                seed=7,
                dim=2,
                bounds=([-5.0], [5.0]),
                n_particles=8,
                max_iters=10,
                patience=10,
                out_dir=str(tmp_path / "results"),
                plots_dir=str(tmp_path / "plots"),
                log_dir=str(tmp_path / "logs"),
                save_files=False,
                log_every=0,
            ),
        )
    finally:
        monkeypatch.setattr(ProcessPoolEvaluator, "open", original_open)

    assert result["v0"]["status"] == "ok"
    assert result["v1"]["status"] == "ok"
    assert result["v2"]["status"] == "unavailable"
    assert result["v3"]["status"] == "ok"
    assert result["v4"]["status"] == "ok"
    assert "sandbox blocked multiprocessing" in result["v2"]["error"]
    assert result["baseline"]["status"] == "unavailable"
    assert "pyswarm" in result["baseline"]["error"]
    assert result["winner"] in {"V0 Sequential", "V1 Threading", "V3 Asyncio", "V4 Vectorized", "Tie"}
