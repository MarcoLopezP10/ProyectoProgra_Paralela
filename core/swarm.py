"""core.swarm

Swarm container responsible for particle collection state and global-best
bookkeeping. The mathematical PSO loop lives in `core.pso`, while this class
keeps the mutable swarm state focused and testable.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from .particle import Particle

class Swarm:
    """Collection of particles plus the swarm-wide best solution."""

    def __init__(
        self,
        n_particles: int,
        dim: int,
        bounds: tuple[list[float], list[float]],
        rng: np.random.Generator,
        vmax_ratio: float = 0.2,
    ):
        """
        Initialize a swarm with multiple particles.

        Args:
            n_particles (int): Number of particles in the swarm.
            dim (int): Dimension of the search space.
            bounds (tuple[list, list]): Lower and upper bounds.
            rng (np.random.Generator): Random number generator.
        """
        if n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if len(bounds[0]) != dim or len(bounds[1]) != dim:
            raise ValueError("bounds must provide one lower/upper value per dimension")

        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if np.any(lower >= upper):
            raise ValueError("each lower bound must be strictly less than the upper bound")

        self.dim = dim
        self.n_particles = n_particles
        self.particles: List[Particle] = [
            Particle(dim, bounds, rng, vmax_ratio=vmax_ratio)
            for _ in range(n_particles)
        ]
        self.global_best_position: NDArray[np.float64] | None = None
        self.global_best_fitness: float = np.inf

    def get_positions(self) -> List[NDArray[np.float64]]:
        """
        Get current positions of all particles.

        Returns:
            List[NDArray]: Positions of particles.
        """
        return [p.position for p in self.particles]

    def update_global_best(self, positions: List[NDArray[np.float64]], fitness_values: List[float]) -> None:
        """
        Update each particle's best and the swarm's global best.

        Args:
            positions (List[NDArray]): Current positions of particles.
            fitness_values (List[float]): Fitness values corresponding to positions.
        """
        if len(positions) != self.n_particles:
            raise ValueError(
                f"positions must contain exactly {self.n_particles} entries, "
                f"got {len(positions)}."
            )
        if len(fitness_values) != self.n_particles:
            raise ValueError(
                f"fitness_values must contain exactly {self.n_particles} entries, "
                f"got {len(fitness_values)}."
            )

        for p, pos, fit in zip(self.particles, positions, fitness_values):
            pos_array = np.asarray(pos, dtype=float)
            if pos_array.shape != (self.dim,):
                raise ValueError(
                    f"each position must have shape ({self.dim},), got {pos_array.shape}."
                )
            if fit < p.best_fitness:
                p.best_fitness = fit
                p.best_position = pos_array.copy()

            # Swarm best is monotonic: once a better point is found, it becomes
            # the shared reference for the next velocity update step.
            if fit < self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best_position = pos_array.copy()
