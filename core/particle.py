"""core.particle

Particle state and update rules used by the PSO swarm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Particle:
    """
    Single particle of the swarm.

    Each particle stores:
    - current position
    - current velocity
    - personal best position
    - personal best fitness
    """

    def __init__(
        self,
        dim: int,
        bounds: tuple[list[float], list[float]],
        rng: np.random.Generator,
        vmax_ratio: float = 0.2,
    ) -> None:
        """
        Initialize a particle inside the search space.

        Parameters
        ----------
        dim : int
            Problem dimension.
        bounds : tuple
            Tuple (lower, upper) with one bound per dimension.
        rng : np.random.Generator
            Random generator for reproducibility.
        """
        self.dim: int = dim
        self.bounds: tuple[list[float], list[float]] = bounds
        self.rng: np.random.Generator = rng
        self.vmax_ratio: float = vmax_ratio

        low, high = bounds
        self.low: NDArray[np.float64] = np.array(low, dtype=float)
        self.high: NDArray[np.float64] = np.array(high, dtype=float)

        # Random initial position inside the allowed search box.
        self.position: NDArray[np.float64] = rng.uniform(self.low, self.high, dim)

        # Use a moderate initial velocity instead of the full search range.
        # This keeps the algorithm much more stable in dimensions such as 30.
        search_range = self.high - self.low
        self.vmax: NDArray[np.float64] = self.vmax_ratio * search_range
        self.velocity: NDArray[np.float64] = rng.uniform(-self.vmax, self.vmax, dim)

        # Initialize personal best with the starting point.
        self.best_position: NDArray[np.float64] = self.position.copy()
        self.best_fitness: float = float("inf")

    def update_velocity(
        self,
        global_best_position: NDArray[np.float64],
        w: float,
        c1: float,
        c2: float,
    ) -> None:
        """
        Update particle velocity using the standard PSO equation.

        Parameters
        ----------
        global_best_position : np.ndarray
            Best position known by the swarm.
        w : float
            Inertia coefficient.
        c1 : float
            Cognitive coefficient.
        c2 : float
            Social coefficient.
        """
        r1 = self.rng.random(self.dim)
        r2 = self.rng.random(self.dim)

        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

        # Limit the step size to avoid unstable jumps.
        self.velocity = np.clip(self.velocity, -self.vmax, self.vmax)

    def update_position(self) -> None:
        """Move the particle according to its current velocity."""
        self.position = self.position + self.velocity
