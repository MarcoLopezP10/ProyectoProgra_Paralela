import time

class PSO:
    def __init__(self, swarm, evaluator, bounds_handler,
                 w, c1, c2, max_iters, rng):
        self.swarm = swarm
        self.evaluator = evaluator
        self.bounds_handler = bounds_handler
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.rng = rng
        self.history = []

    def run(self):
        start = time.perf_counter()

        for it in range(self.max_iters):
            positions = self.swarm.get_positions()
            fitness = self.evaluator.evaluate(positions)

            self.swarm.update_global_best(positions, fitness)

            for p in self.swarm.particles:
                p.update_velocity(
                    self.swarm.global_best_position,
                    self.w, self.c1, self.c2, self.rng
                )
                p.update_position()
                p.position, p.velocity = (
                    self.bounds_handler.apply(p.position, p.velocity)
                )

            self.history.append(self.swarm.global_best_fitness)

        elapsed = time.perf_counter() - start
        return self.swarm.global_best_position, self.swarm.global_best_fitness, elapsed

    
