from .particle import Particle
import numpy as np
from numpy.typing import NDArray
from typing import List

class Swarm:
    """
    Class representing a PSO swarm.
    """

    def __init__(self, n_particles: int, dim: int, bounds: tuple[list[float], list[float]], rng: np.random.Generator):
        """
        Initialize a swarm with multiple particles.

        Args:
            n_particles (int): Number of particles in the swarm.
            dim (int): Dimension of the search space.
            bounds (tuple[list, list]): Lower and upper bounds.
            rng (np.random.Generator): Random number generator.
        """
        self.particles: List[Particle] = [Particle(dim, bounds, rng) for _ in range(n_particles)]
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
        for p, pos, fit in zip(self.particles, positions, fitness_values):
            if fit < p.best_fitness:
                p.best_fitness = fit
                p.best_position = pos.copy()

            if fit < self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best_position = pos.copy()

