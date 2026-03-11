from typing import Callable, Tuple, List
import numpy as np
import time
from pyswarm import pso


def run_pyswarm_baseline(
    objective_function: Callable[[np.ndarray], float],
    bounds: Tuple[List[float], List[float]],
    n_particles: int,
    max_iters: int
) -> Tuple[np.ndarray, float, float, int]:
    """Run PySwarm's PSO with a configuration comparable to our implementation.

Args:
    objective_function: Objective function to minimize.
    bounds: (lower, upper) bounds per dimension.
    n_particles: Swarm size.
    max_iters: Maximum number of iterations.

Returns:
    best_position, best_fitness, elapsed_time_seconds, iterations
"""

    lower_bounds, upper_bounds = bounds

    start_time = time.time()

    best_position, best_fitness = pso(
        objective_function,
        lb=lower_bounds,
        ub=upper_bounds,
        swarmsize=n_particles,
        maxiter=max_iters,
        debug=False
    )

    elapsed_time = time.time() - start_time

    return best_position, best_fitness, elapsed_time, max_iters
