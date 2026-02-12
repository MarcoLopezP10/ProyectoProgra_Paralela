from .particle import Particle
import numpy as np


class Swarm:
    def __init__(self, n_particles, dim, bounds, rng):
        self.particles = [
            Particle(dim, bounds, rng) for _ in range(n_particles)
        ]
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def get_positions(self):
        return [p.position for p in self.particles]

    def update_global_best(self, positions, fitness_values):
        for p, pos, fit in zip(self.particles, positions, fitness_values):

            if fit < p.best_fitness:
                p.best_fitness = fit
                p.best_position = pos.copy()

            if fit < self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best_position = pos.copy()
