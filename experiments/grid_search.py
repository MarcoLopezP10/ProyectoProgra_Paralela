import numpy as np
from core.swarm import Swarm
from core.pso import PSO
from options.evaluator import SequentialEvaluator
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from typing import Dict, Any

def simple_grid_search(
    objective_fn,
    dim: int,
    bounds: tuple[list[float], list[float]],
    max_iters: int = 200,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Simple grid search for PSO hyperparameters.

    Returns the combination with the best fitness.
    """
    # Small hyperparameter grid
    w_list = [0.5, 0.7, 0.9]
    c1_list = [1.0, 1.5, 2.0]
    c2_list = [1.0, 1.5, 2.0]
    n_particles_list = [50, 100, 150]

    best_config: Dict[str, Any] = {"fitness": np.inf}

    for w in w_list:
        for c1 in c1_list:
            for c2 in c2_list:
                for n_particles in n_particles_list:
                    rng = np.random.default_rng(seed)
                    swarm = Swarm(n_particles=n_particles, dim=dim, bounds=bounds, rng=rng)
                    evaluator = SequentialEvaluator(objective_fn)
                    bounds_handler = ClampBounds(bounds[0], bounds[1])
                    topology = GlobalBestTopology()

                    pso = PSO(swarm, evaluator, bounds_handler, topology,
                              w, c1, c2, max_iters=max_iters)

                    _, fitness, _, _ = pso.run()

                    if fitness < best_config["fitness"]:
                        best_config = {"w": w, "c1": c1, "c2": c2, "n_particles": n_particles, "fitness": fitness}

    return best_config
