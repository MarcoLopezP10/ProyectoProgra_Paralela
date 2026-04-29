"""Objective with asymmetric latency to make asyncio.gather meaningful."""

from __future__ import annotations

import asyncio
import time

import numpy as np
from numpy.typing import NDArray

_BASE_LATENCY_S = 0.0004
_SPREAD_LATENCY_S = 0.0024


def _latency_scale(position: NDArray[np.float64]) -> float:
    """Deterministic per-particle latency weight in [0, 1]."""
    x = np.asarray(position, dtype=float)
    if x.size == 0:
        return 0.0

    weights = np.arange(1, x.size + 1, dtype=float)
    signal_a = abs(np.sin(float(np.dot(x, weights * 0.73))))
    signal_b = abs(np.cos(float(np.sum(np.abs(x)) * 0.41 + x.size * 0.13)))
    signal_c = float(np.mod(abs(x[0]), 1.0))
    return min(1.0, 0.55 * signal_a + 0.35 * signal_b + 0.10 * signal_c)


def latency_seconds(position: NDArray[np.float64]) -> float:
    """Latency used by sync and async paths so the comparison stays fair."""
    return _BASE_LATENCY_S + _SPREAD_LATENCY_S * _latency_scale(position)


def _fitness(position: NDArray[np.float64]) -> float:
    """Smooth bowl plus ripple so optimisation still has a real landscape."""
    x = np.asarray(position, dtype=float)
    shifted = x - 0.75
    bowl = float(np.sum(shifted**2))
    ripple = float(0.08 * np.sum(1.0 - np.cos(3.0 * x)))
    return bowl + ripple


def latency_mix(position: NDArray[np.float64]) -> float:
    """Blocking objective path used by baseline, V0, V1 and V2."""
    time.sleep(latency_seconds(position))
    return _fitness(position)


async def latency_mix_async(position: NDArray[np.float64]) -> float:
    """Cooperative path used by V3 via asyncio.gather."""
    await asyncio.sleep(latency_seconds(position))
    return _fitness(position)


latency_mix.async_evaluate = latency_mix_async  # type: ignore[attr-defined]
latency_mix.describe_latency = latency_seconds  # type: ignore[attr-defined]

