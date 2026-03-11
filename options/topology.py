from abc import ABC, abstractmethod


class Topology(ABC):
    """
    Abstract base class for swarm topologies.
     Defines the interface for determining the best position for a particle based on the swarm's topology.
    """
    @abstractmethod
    def get_best_position(self, particle, swarm):
        pass


class GlobalBestTopology(Topology):
    """
    Global best topology where each particle is influenced by the best position found by the entire swarm.
    """
    def get_best_position(self, particle, swarm):
        """
        Get the global best position from the swarm.
        Args:
            particle: The particle for which to determine the best position (not used in global topology).
            swarm: The swarm containing all particles and the global best information.
        Returns:
             The global best position found by the swarm.
        """
        return swarm.global_best_position
