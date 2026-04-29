"""Shared metadata for PSO execution methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str
    linestyle: str = "-"
    marker: str = "o"
    has_history: bool = True


PSO_METHOD_SPECS = [
    MethodSpec("v0", "V0 Sequential", "#4c78a8", "-", "o"),
    MethodSpec("v1", "V1 Threading", "#f58518", "--", "s"),
    MethodSpec("v2", "V2 Multiprocessing", "#54a24b", ":", "^"),
    MethodSpec("v3", "V3 Asyncio", "#e45756", "-.", "D"),
]

BASELINE_METHOD_SPEC = MethodSpec(
    "baseline",
    "PySwarm baseline",
    "#7f7f7f",
    "--",
    "x",
    has_history=False,
)

SUMMARY_METHOD_SPECS = PSO_METHOD_SPECS + [BASELINE_METHOD_SPEC]
PSO_METHOD_KEYS = tuple(spec.key for spec in PSO_METHOD_SPECS)
SUMMARY_METHOD_KEYS = tuple(spec.key for spec in SUMMARY_METHOD_SPECS)
STRATEGY_LABELS = {spec.key: spec.label for spec in PSO_METHOD_SPECS}

