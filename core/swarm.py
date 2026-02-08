import numpy as np
from core.particle import Particle

class Swarm:
    def __init__(self,n_particles,dim,bounds,rng): 
        self.particles=[
            Particle(dim,bounds,rng) for _ in range(n_particles)
        ]
        self.global_best_position=None
        self.global_best_fitness=np.inf

    def get_position(self):
        return np.array([p.position for p in self.particles])
    
    def update_global_best(self,particle,fitness):
        if fitness<self.global_best_fitness:
            self.global_best_fitness=fitness
            self.global_best_position=particle.position.copy()

