import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .swarm import Swarm
from options.topology import Topology
from options.bounds import ClampBounds
from options.evaluator import FitnessEvaluator
import logging


class PSO:
    """
    Particle Swarm Optimization algorithm.
    """

    def __init__(
        self,
        swarm: Swarm,
        evaluator: FitnessEvaluator,
        bounds_handler: ClampBounds,
        topology: Topology,
        w: float,
        c1: float,
        c2: float,
        max_iters: int,
        tol: float = 1e-8,
        patience: int = 20,
        logger: Optional[logging.Logger] = None
    ):
        self.swarm = swarm
        self.evaluator = evaluator
        self.bounds_handler = bounds_handler
        self.topology = topology

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.tol = tol
        self.patience = patience

        # Convergence history
        self.history: list[float] = []

        self.logger = logger

    def run(self) -> tuple[NDArray[np.float64], float, float, int]:
        """
        Run the PSO optimization.

        Returns:
            best_position
            best_fitness
            elapsed_time
            iterations_executed
        """

        start = time.perf_counter()
        no_improve_counter = 0
        prev_best = np.inf

        for it in range(self.max_iters):

            positions = self.swarm.get_positions()
            fitness = self.evaluator.evaluate(positions)

            self.swarm.update_global_best(positions, fitness)

            # Store convergence history
            self.history.append(self.swarm.global_best_fitness)

            # Early stopping 
            improvement = abs(prev_best - self.swarm.global_best_fitness)

            if improvement < self.tol:
                no_improve_counter += 1
            else:
                no_improve_counter = 0

            if no_improve_counter >= self.patience:
                break

            prev_best = self.swarm.global_best_fitness

            #  Update particles 
            for p in self.swarm.particles:
                best_pos = self.topology.get_best_position(p, self.swarm)

                p.update_velocity(best_pos, self.w, self.c1, self.c2)
                p.update_position()

                p.position, p.velocity = self.bounds_handler.apply(
                    p.position, p.velocity
                )
    
        elapsed = time.perf_counter() - start

        # Logging for results and iterations
        if self.logger:
            self.logger.info(
                f"PSO finished: Best fitness={self.swarm.global_best_fitness}, "
                f"Iterations={len(self.history)}, Elapsed={elapsed:.4f}s"
            )

        return (
            self.swarm.global_best_position,
            self.swarm.global_best_fitness,
            elapsed,
            len(self.history),
        )
