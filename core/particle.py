import numpy as np

class Particle:
    def __init__(self, dim, bounds, rng):
        self.position = rng.uniform(bounds[0], bounds[1], size=dim)
        self.velocity = rng.uniform(-1, 1, size=dim)
        self.best_position = self.position.copy()
        self.best_fitness = np.inf

    def update_velocity(self, gbest, w, c1, c2, rng):
        r1 = rng.random(len(self.position))
        r2 = rng.random(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (gbest - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        self.position = self.position + self.velocity




