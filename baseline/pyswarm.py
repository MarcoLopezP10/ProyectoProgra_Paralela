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
    """
    Ejecuta el PSO de pyswarm con configuración comparable a tu implementación.

    Args:
        objective_function: función objetivo.
        bounds: tupla (lower_bounds, upper_bounds).
        n_particles: número de partículas.
        max_iters: número máximo de iteraciones.

    Returns:
        best_position
        best_fitness
        elapsed_time
        iterations (max_iters, ya que pyswarm no devuelve las reales)
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
