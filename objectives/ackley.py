"""objectives.ackley

Ackley benchmark with the global minimum at the origin.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ackley(
    x: ArrayLike,
    a: float = 20.0,
    b: float = 0.2,
    c: float = 2 * np.pi,
) -> float:
    """Return the Ackley objective value for one point."""
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.size

    sum_sq = np.sum(x_arr**2)
    sum_cos = np.sum(np.cos(c * x_arr))

    return float(
        -a * np.exp(-b * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + a + np.e
    )
