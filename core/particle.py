import numpy as np
from numpy.typing import NDArray

class Particle:
    """
    Class representing a particle in the PSO swarm.
    """

    def __init__(self, dim: int, bounds: tuple[list[float], list[float]], rng: np.random.Generator):
        """
        Initialize a particle.

        Args:
            dim (int): Dimension of the search space.
            bounds (tuple[list, list]): Lower and upper bounds for each dimension.
            rng (np.random.Generator): Random number generator.
        """
        self.dim: int = dim
        self.bounds: tuple[np.ndarray, np.ndarray] = (np.array(bounds[0]), np.array(bounds[1]))
        self.rng: np.random.Generator = rng

        low, high = self.bounds
        self.position: NDArray[np.float_] = rng.uniform(low, high, dim)
        self.velocity: NDArray[np.float_] = rng.uniform(-(high - low), (high - low))

        self.best_position: NDArray[np.float_] = self.position.copy()
        self.best_fitness: float = np.inf

    def update_velocity(self, best_position: NDArray[np.float_], w: float, c1: float, c2: float) -> None:
        """
        Update the velocity of the particle.

        Args:
            best_position (NDArray): The reference best position (global or local).
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
        """
        r1 = self.rng.random(self.dim)
        r2 = self.rng.random(self.dim)

        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

    def update_position(self) -> None:
        """
        Update the particle's position based on its velocity.
        """
        self.position += self.velocity
