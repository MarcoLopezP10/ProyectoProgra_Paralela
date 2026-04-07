"""objectives.rastrigin

Highly multimodal Rastrigin benchmark.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rastrigin(x: ArrayLike) -> float:
    """Return the Rastrigin objective value for one point."""
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.size
    return float(10 * n + np.sum(x_arr**2 - 10 * np.cos(2 * np.pi * x_arr)))
