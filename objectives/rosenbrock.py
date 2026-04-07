"""objectives.rosenbrock

Rosenbrock valley benchmark.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rosenbrock(x: ArrayLike) -> float:
    """Return the Rosenbrock objective value for one point."""
    x_arr = np.asarray(x, dtype=float)
    return float(
        np.sum(100 * (x_arr[1:] - x_arr[:-1] ** 2) ** 2 + (x_arr[:-1] - 1) ** 2)
    )
