from abc import ABC,abstractmethod
import numpy as np

class BoundsHandler(ABC):
    @abstractmethod
    def apply(self,position,velocity):
        pass

class ClampBounds(BoundsHandler):
    def __init__(self,bounds):
        self.min,self.max=bounds
    
    def apply(self,position,velocity):
        position =np.clip(position,self,min,self.max) #No tocamos la velocidad
        return position,velocity 
    
