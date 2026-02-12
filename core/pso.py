import time
import numpy as np


class PSO:
    def __init__(self, swarm, evaluator, bounds_handler,
                 topology,
                 w, c1, c2,
                 max_iters,
                 tol=1e-8,
                 patience=20):

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

        self.history = []

    def run(self):
        start = time.perf_counter()

        no_improve_counter = 0
        prev_best = np.inf

        for it in range(self.max_iters):

            positions = self.swarm.get_positions()
            fitness = self.evaluator.evaluate(positions)

            self.swarm.update_global_best(positions, fitness)

            self.history.append(self.swarm.global_best_fitness)

            # Early stopping
            improvement = abs(prev_best - self.swarm.global_best_fitness)

            if improvement < self.tol:
                no_improve_counter += 1
            else:
                no_improve_counter = 0

            if no_improve_counter >= self.patience:
                print(f"Early stopping at iteration {it}")
                break

            prev_best = self.swarm.global_best_fitness

            # Update particles
            for p in self.swarm.particles:
                best_position = self.topology.get_best_position(p, self.swarm)

                p.update_velocity(best_position, self.w, self.c1, self.c2)
                p.update_position()
                p.position, p.velocity = self.bounds_handler.apply(
                    p.position, p.velocity
                )

        elapsed = time.perf_counter() - start

        return (
            self.swarm.global_best_position,
            self.swarm.global_best_fitness,
            elapsed,
            len(self.history)
        )

    
