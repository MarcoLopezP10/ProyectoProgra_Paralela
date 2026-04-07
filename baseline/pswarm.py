"""baseline.pswarm

Thin wrapper around the external PySwarm implementation used as a reference
baseline in the experiment scripts.
"""

from __future__ import annotations

import random
import time
from typing import Callable, List, Optional, Tuple

import numpy as np


def run_pyswarm_baseline(
    objective_function: Callable[[np.ndarray], float],
    bounds: Tuple[List[float], List[float]],
    w: float,
    c1: float,
    c2: float,
    n_particles: int,
    max_iters: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, float, int]:
    """Run PySwarm's PSO with a configuration comparable to our implementation.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        Objective function to minimize.
    bounds : tuple[list[float], list[float]]
        Lower and upper bounds per dimension.
    w, c1, c2 : float
        PSO hyperparameters forwarded to PySwarm.
    n_particles : int
        Swarm size.
    max_iters : int
        Maximum number of iterations.
    seed : int | None, optional
        Seed forwarded to NumPy/Python RNGs before running PySwarm.

    Returns
    -------
    tuple[np.ndarray, float, float, int]
        Best position, best fitness, elapsed time in seconds, and iterations.
    """
    from pyswarm import pso

    lower_bounds, upper_bounds = bounds

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_time = time.perf_counter()

    best_position, best_fitness = pso(
        objective_function,
        lb=lower_bounds,
        ub=upper_bounds,
        swarmsize=n_particles,
        omega=w,
        phip=c1,
        phig=c2,
        maxiter=max_iters,
        debug=False
    )

    elapsed_time = time.perf_counter() - start_time

    return best_position, best_fitness, elapsed_time, max_iters
