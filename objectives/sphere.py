"""objectives.sphere

Canonical convex Sphere benchmark.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def sphere(x: ArrayLike) -> float:
    """Return the Sphere objective value for one point."""
    return float(np.sum(np.asarray(x, dtype=float) ** 2))
