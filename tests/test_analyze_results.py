from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analyze_results import _load_summaries


def test_load_summaries_accepts_legacy_and_backfills_metrics(tmp_path):
    legacy_dir = tmp_path / "sphere_d2_s42"
    legacy_dir.mkdir()

    summary = {
        "objective": "sphere",
        "dim": 2,
        "seed": 42,
        "tol": 1e-8,
        "v0": {
            "best_fitness": 0.01,
            "iterations": 3,
            "timing": {"total_s": 1.0},
        },
        "v1": {
            "best_fitness": 0.01,
            "iterations": 3,
            "timing": {"total_s": 1.2},
        },
        "v2": {
            "best_fitness": 0.01,
            "iterations": 3,
            "timing": {"total_s": 1.5},
        },
        "v3": {
            "best_fitness": 0.01,
            "iterations": 3,
            "timing": {"total_s": 0.9},
        },
        "baseline": {
            "best_fitness": 0.1,
            "iterations": 3,
            "timing": {"total_s": 0.8},
        },
    }
    (legacy_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for method_key in ("v0", "v1", "v2", "v3"):
        (legacy_dir / f"history_{method_key}.csv").write_text(
            "iter,best_fitness\n0,1.0\n1,0.1\n2,0.01\n",
            encoding="utf-8",
        )

    summaries, stats = _load_summaries(str(tmp_path))

    assert stats["loaded"] == 1
    assert len(summaries) == 1
    loaded = summaries[0]
    assert loaded["execution"] == {}
    assert loaded["v0"]["auc"] is not None
    assert loaded["v0"]["convergence_iteration"] is not None
    assert loaded["v3"]["auc"] is not None
    assert loaded["v3"]["convergence_iteration"] is not None
    assert loaded["baseline"]["auc"] is None
    assert loaded["baseline"]["convergence_iteration"] is None
    assert loaded["baseline"]["timing"]["eval_s"] == 0.0


def test_load_summaries_skips_missing_core_fields(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "summary.json").write_text(
        json.dumps({"objective": "sphere", "dim": 2, "seed": 42}),
        encoding="utf-8",
    )

    summaries, stats = _load_summaries(str(tmp_path))

    assert summaries == []
    assert stats["skipped_missing_core"] == 1
