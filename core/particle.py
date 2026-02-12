import numpy as np


class Particle:
    def __init__(self, dim, bounds, rng):
        self.dim = dim
        self.bounds = bounds
        self.rng = rng

        low, high = bounds

        self.position = rng.uniform(low, high, dim)
        self.velocity = rng.uniform(-(np.array(high) - np.array(low)),
                                    (np.array(high) - np.array(low)))

        self.best_position = self.position.copy()
        self.best_fitness = np.inf

    def update_velocity(self, best_position, w, c1, c2):
        r1 = self.rng.random(self.dim)
        r2 = self.rng.random(self.dim)

        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        self.position = self.position + self.velocity



