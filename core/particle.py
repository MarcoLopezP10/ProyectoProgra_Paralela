from __future__ import annotations

import numpy as np


class Particle:
    """
    Single particle of the swarm.

    Each particle stores:
    - current position
    - current velocity
    - personal best position
    - personal best fitness
    """

    def __init__(self, dim, bounds, rng, vmax_ratio: float = 0.2):
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
        self.dim = dim
        self.bounds = bounds
        self.rng = rng
        self.vmax_ratio = vmax_ratio

        low, high = bounds
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)

        # Random initial position inside the allowed search box.
        self.position = rng.uniform(self.low, self.high, dim)

        # Use a moderate initial velocity instead of the full search range.
        # This keeps the algorithm much more stable in dimensions such as 30.
        search_range = self.high - self.low
        self.vmax = self.vmax_ratio * search_range
        self.velocity = rng.uniform(-self.vmax, self.vmax, dim)

        # Initialize personal best with the starting point.
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")

    def update_velocity(self, global_best_position, w, c1, c2):
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

    def update_position(self):
        """
        Move the particle according to its current velocity.
        """
        self.position = self.position + self.velocity
