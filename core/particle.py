import numpy as np

class Particle:
    def __init__(self,dim,bounds,rng): #Estado Propio, dimension, limites y generador aleatorio
        self.position=rng.uniform(bounds[0],bounds[1],size=dim)
        self.velocity=rng.uniform(-(bounds[1]-bounds[0]), (bounds[1]-bounds[0]),size=dim) 
        self.best_position=self.position.copy()
        self.best_fitness=np.inf
    
    def update_velocity(self,global_best_position,w,c1,c2,rng): #Actualizacion de velocidad(ecuacion PSO)
        #Coeficientes aleatorios independientes
        r1=rng.random(self.position.shape)
        r2=rng.random(self.position.shape)

        #Parte cognitiva y Social(enjambre)
        cognitive=c1*r1*(self.best_position - self.position)
        social=c2*r2*(global_best_position - self.position)

        #Formula velocidad
        self.velocity=w*self.velocity+cognitive+social
    
    def update_position(self):
        self.position=self.position + self.velocity
        



