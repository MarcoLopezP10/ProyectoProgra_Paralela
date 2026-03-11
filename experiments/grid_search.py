"""experiment.grid_search

Lightweight grid search for PSO hyperparameters.

"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from core.swarm import Swarm
from core.pso import PSO
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology

# In V0 we keep the evaluator import from parallel so the future swap point stays stable.
from parallel.evaluator import SequentialEvaluator


def simple_grid_search(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[List[float], List[float]],
    max_iters: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a small grid search for PSO hyperparameters.

    Parameters
    ----------
    objective_fn : Callable[[np.ndarray], float]
        Objective function to minimize.
    dim : int
        Problem dimension.
    bounds : Tuple[List[float], List[float]]
        Lower and upper bounds.
    max_iters : int, default=200
        Reference maximum iterations from the main experiment.
    seed : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Best hyperparameter combination found.
    """

    
    name = getattr(objective_fn, "__name__", "objective")

    if name == "rosenbrock":
        w_list = np.linspace(0.6, 0.9, 4)
        c1_list = np.linspace(1.2, 1.8, 3)
        c2_list = np.linspace(1.2, 1.8, 3)
        n_particles_list = np.linspace(80, 140, 4)

    elif name == "rastrigin":
        w_list = np.linspace(0.5, 0.8, 4)
        c1_list = np.linspace(1.0, 1.6, 4)
        c2_list = np.linspace(1.4, 2.0, 4)
        n_particles_list = np.linspace(80, 140, 4)

    else:
        w_list = np.linspace(0.4, 0.8, 3)
        c1_list = np.linspace(1.2, 1.8, 3)
        c2_list = np.linspace(1.2, 1.8, 3)
        n_particles_list = np.linspace(60, 120, 3)

    best_fitness = float("inf")
    best_params: Dict[str, Any] = {}

    # The grid search should be cheaper than the final run.
    grid_max_iters = max(40, int(max_iters * 0.4))

    for w in w_list:
        for c1 in c1_list:
            for c2 in c2_list:
                for n_particles in n_particles_list:
                    w_value = float(w)
                    c1_value = float(c1)
                    c2_value = float(c2)

                    
                    n_particles_value = int(round(float(n_particles)))

                    rng = np.random.default_rng(seed)

                    swarm = Swarm(
                        n_particles=n_particles_value,
                        dim=dim,
                        bounds=bounds,
                        rng=rng,
                    )

                    evaluator = SequentialEvaluator(objective_fn)
                    bounds_handler = ClampBounds(bounds[0], bounds[1])
                    topology = GlobalBestTopology()

                    pso = PSO(
                        swarm=swarm,
                        evaluator=evaluator,
                        bounds_handler=bounds_handler,
                        topology=topology,
                        w=w_value,
                        c1=c1_value,
                        c2=c2_value,
                        max_iters=grid_max_iters,
                        tol=1e-8,
                        patience=40,
                    )

                    _, best_fit, _, _ = pso.run()

                    if best_fit < best_fitness:
                        best_fitness = float(best_fit)
                        best_params = {
                            "w": w_value,
                            "c1": c1_value,
                            "c2": c2_value,
                            "n_particles": n_particles_value,
                            "best_fitness": best_fitness,
                        }

    return best_params
