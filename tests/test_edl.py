"""tests.test_edl

Unit tests for the Economic Load Dispatch add-on.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from experiment.run_edl import (
    EDLRunConfig,
    _format_dispatch_lines,
    _group_dispatches,
    _summarize_powers,
    run_one_edl_variant,
)
from objectives.economic_dispatch import (
    EconomicDispatchCase,
    EconomicDispatchObjective,
    GeneratorUnit,
    LossModel,
    load_case,
)
from parallel.evaluator import ProcessPoolEvaluator
from utils.edl_io import EDLMethodResult


def _make_case() -> EconomicDispatchCase:
    return EconomicDispatchCase(
        case_name="toy_case",
        demand=120.0,
        generators=(
            GeneratorUnit("G1", 50.0, 100.0, 0.01, 2.0, 10.0, 5.0, 0.1),
            GeneratorUnit("G2", 20.0, 80.0, 0.02, 1.5, 5.0, 3.0, 0.2),
        ),
        loss_model=LossModel(
            B=np.array([[0.001, 0.0], [0.0, 0.002]], dtype=float),
            B0=np.array([0.01, 0.02], dtype=float),
            B00=1.0,
        ),
    )


def test_load_case_reads_bounds_and_losses() -> None:
    case = load_case(Path("cases/edl_case_3u.json"))

    assert case.case_name == "edl_3u_demo"
    assert case.dim == 3
    assert case.lower_bounds == [100.0, 100.0, 50.0]
    assert case.upper_bounds == [600.0, 400.0, 200.0]
    assert case.loss_model is not None
    assert case.loss_model.B.shape == (3, 3)
    assert case.loss_model.quadratic_base_mva == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "expected_name", "expected_dim", "has_losses"),
    [
        ("cases/edl_case_3u.json", "edl_3u_demo", 3, True),
        ("cases/edl_case_6u.json", "edl_6u_vpe_losses", 6, True),
        ("cases/edl_case_13u.json", "edl_13u_vpe", 13, False),
        ("cases/edl_case_40u.json", "edl_40u_vpe", 40, False),
    ],
)
def test_standard_ladder_cases_load(path, expected_name, expected_dim, has_losses) -> None:
    case = load_case(Path(path))

    assert case.case_name == expected_name
    assert case.dim == expected_dim
    assert (case.loss_model is not None) is has_losses


def test_6u_case_uses_100_mva_base_for_quadratic_loss_term() -> None:
    case = load_case(Path("cases/edl_case_6u.json"))

    assert case.loss_model is not None
    assert case.loss_model.quadratic_base_mva == pytest.approx(100.0)


def test_base_cost_without_valve_point() -> None:
    case = _make_case()
    objective = EconomicDispatchObjective(case=case, use_valve_point=False, use_losses=False)
    powers = np.array([60.0, 40.0])

    expected = 0.01 * 60.0**2 + 2.0 * 60.0 + 10.0
    expected += 0.02 * 40.0**2 + 1.5 * 40.0 + 5.0

    assert objective.base_cost(powers) == pytest.approx(expected)
    assert objective.fuel_cost(powers) == pytest.approx(expected)


def test_valve_point_cost_is_added_when_enabled() -> None:
    case = _make_case()
    powers = np.array([60.0, 40.0])
    objective = EconomicDispatchObjective(case=case, use_valve_point=True, use_losses=False)

    base = objective.base_cost(powers)
    valve = abs(5.0 * np.sin(0.1 * (50.0 - 60.0)))
    valve += abs(3.0 * np.sin(0.2 * (20.0 - 40.0)))

    assert objective.valve_point_cost(powers) == pytest.approx(valve)
    assert objective.fuel_cost(powers) == pytest.approx(base + valve)


def test_transmission_loss_and_balance_error_with_losses() -> None:
    case = _make_case()
    objective = EconomicDispatchObjective(case=case, use_valve_point=False, use_losses=True)
    powers = np.array([70.0, 55.0])

    expected_loss = float(
        powers @ case.loss_model.B @ powers + case.loss_model.B0 @ powers + case.loss_model.B00
    )
    expected_residual = float(np.sum(powers) - (case.demand + expected_loss))

    assert objective.transmission_loss(powers) == pytest.approx(expected_loss)
    assert objective.power_balance_residual(powers) == pytest.approx(expected_residual)
    assert objective.power_balance_error(powers) == pytest.approx(abs(expected_residual))


def test_transmission_loss_respects_quadratic_base_mva() -> None:
    case = EconomicDispatchCase(
        case_name="scaled_loss_case",
        demand=120.0,
        generators=(
            GeneratorUnit("G1", 50.0, 100.0, 0.01, 2.0, 10.0),
            GeneratorUnit("G2", 20.0, 80.0, 0.02, 1.5, 5.0),
        ),
        loss_model=LossModel(
            B=np.array([[0.5, 0.0], [0.0, 0.25]], dtype=float),
            B0=np.array([0.1, 0.2], dtype=float),
            B00=3.0,
            quadratic_base_mva=10.0,
        ),
    )
    objective = EconomicDispatchObjective(case=case, use_losses=True)
    powers = np.array([20.0, 10.0])

    expected_loss = float((powers @ case.loss_model.B @ powers) / 10.0 + case.loss_model.B0 @ powers + 3.0)

    assert objective.transmission_loss(powers) == pytest.approx(expected_loss)


def test_summarize_powers_is_compact() -> None:
    summary = _summarize_powers([10.0, 20.0, 30.0, 40.0])

    assert "n=4" in summary
    assert "sum=100.0000" in summary
    assert "min=10.0000" in summary
    assert "max=40.0000" in summary


def test_dispatch_lines_wrap_long_vectors_by_count() -> None:
    lines = _format_dispatch_lines(
        ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        entries_per_line=3,
    )

    assert lines == [
        "G1=10.0000 | G2=20.0000 | G3=30.0000",
        "G4=40.0000 | G5=50.0000 | G6=60.0000",
        "G7=70.0000",
    ]


def test_dispatch_lines_wrap_long_vectors_by_width() -> None:
    lines = _format_dispatch_lines(
        ["G1", "G2", "G3", "G4"],
        [10.0, 20.0, 30.0, 40.0],
        max_width=28,
    )

    assert lines == [
        "G1=10.0000 | G2=20.0000",
        "G3=30.0000 | G4=40.0000",
    ]


def test_group_dispatches_merges_identical_solutions() -> None:
    methods = [
        EDLMethodResult(strategy="V0 Sequential", best_position=[10.0, 20.0]),
        EDLMethodResult(strategy="V1 Threading", best_position=[10.0, 20.0]),
        EDLMethodResult(strategy="V2 Multiprocessing", best_position=[11.0, 19.0]),
    ]

    groups = _group_dispatches(methods)

    assert len(groups) == 2
    assert groups[0]["strategies"] == ["V0 Sequential", "V1 Threading"]
    assert groups[1]["strategies"] == ["V2 Multiprocessing"]


def test_objective_is_picklable_for_process_evaluator() -> None:
    objective = EconomicDispatchObjective(case=_make_case(), use_valve_point=True, use_losses=True)

    payload = pickle.dumps(objective)
    restored = pickle.loads(payload)

    assert restored(np.array([70.0, 50.0])) == pytest.approx(objective(np.array([70.0, 50.0])))


def test_runner_marks_v2_unavailable_when_process_pool_is_blocked(monkeypatch, tmp_path) -> None:
    original_open = ProcessPoolEvaluator.open

    def fake_open(self) -> None:
        raise PermissionError("sandbox blocked multiprocessing")

    monkeypatch.setattr(ProcessPoolEvaluator, "open", fake_open)
    try:
        case = _make_case()
        cfg = EDLRunConfig(
            seeds=[7],
            variants=["edl_1"],
            n_particles=6,
            max_iters=5,
            patience=5,
            out_dir=str(tmp_path / "results"),
            log_dir=str(tmp_path / "logs"),
            save_files=False,
        )
        summary = run_one_edl_variant(case, "edl_1", cfg, seed=7)
    finally:
        monkeypatch.setattr(ProcessPoolEvaluator, "open", original_open)

    assert summary.v0.status == "ok"
    assert summary.v1.status == "ok"
    assert summary.v2.status == "unavailable"
    assert summary.v2.error is not None


def test_runner_marks_loss_variants_unavailable_when_case_has_no_loss_model(tmp_path) -> None:
    case = load_case(Path("cases/edl_case_13u.json"))
    cfg = EDLRunConfig(
        seeds=[11],
        variants=["edl_3"],
        n_particles=6,
        max_iters=5,
        patience=5,
        out_dir=str(tmp_path / "results"),
        log_dir=str(tmp_path / "logs"),
        save_files=False,
    )
    summary = run_one_edl_variant(case, "edl_3", cfg, seed=11)

    assert summary.v0.status == "unavailable"
    assert summary.v1.status == "unavailable"
    assert summary.v2.status == "unavailable"
    assert summary.winner == "Unavailable"


def test_6u_loss_variant_stays_feasible(monkeypatch, tmp_path) -> None:
    original_open = ProcessPoolEvaluator.open

    def fake_open(self) -> None:
        raise PermissionError("sandbox blocked multiprocessing")

    monkeypatch.setattr(ProcessPoolEvaluator, "open", fake_open)
    try:
        case = load_case(Path("cases/edl_case_6u.json"))
        cfg = EDLRunConfig(
            seeds=[42],
            variants=["edl_3"],
            n_particles=40,
            max_iters=120,
            patience=40,
            out_dir=str(tmp_path / "results"),
            log_dir=str(tmp_path / "logs"),
            save_files=False,
        )
        summary = run_one_edl_variant(case, "edl_3", cfg, seed=42)
    finally:
        monkeypatch.setattr(ProcessPoolEvaluator, "open", original_open)

    assert summary.v0.status == "ok"
    assert summary.v0.power_balance_error is not None
    assert summary.v0.power_balance_error < 1e-2
    assert summary.v0.transmission_loss is not None
    assert 1.0 < summary.v0.transmission_loss < 50.0
    assert summary.v0.best_fitness is not None
    assert summary.v0.best_fitness < 1e6
