from abc import ABC,abstractmethod

class Topology(ABC):
    @abstractmethod
    def get_best_position(self,particle,swarm):
        pass

class GlobalBestTopology(Topology):
    def get_best_position(self, particle, swarm):
        return swarm.global_best_position
        
