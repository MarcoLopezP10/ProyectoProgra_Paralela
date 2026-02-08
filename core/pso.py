import numpy as np

class PSO: #Clase coordinadora 
    def __init__(self,swarm,evaluator,topology,bounds_handler,w,c1,c2,max_iter,seed=None):
        self.swarm=swarm
        self.evaluator=evaluator
        self.topology=topology
        self.bounds_handler=bounds_handler

        self.w=w
        self.c1=c1
        self.c2=c2
        self.max_iter = max_iter

        self.rng=np.random.default_rng(seed) #Reproducibilidad garantizada
    
    def run(self):
        history=[]

        for _ in range (self.max_iter):
            positions = self.swarm.get_positions()
            fitness_values= self.evaluator.evaluate(positions)

            for particle,fitness in zip(self.swarm.particles, fitness_values):
                if fitness < particle.best_fitness: #Actualizamos pbests y gbest
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                self.swarm.update_global_best(particle,fitness)
            
            for particle in self.swarm.particles:
                best_pos=self.topology.get_best_position( #Movimeinto de particulas
                    particle, self.swarm
                )
                particle.update_velocity( #Topologia explicita
                    best_pos, self.w, self.c1,self.c2,self.rng
                )
                particle.update_position()
                particle.position, particle.velocity = self.bounds_handler.apply( #Limites explicitos
                    particle.position, particle.velocity
                )
            history.append(self.swarm.global_best_fitness)
        return self.swarm.global_best_position, history
    
