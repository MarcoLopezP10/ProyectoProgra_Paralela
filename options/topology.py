"""options.topology

Neighbourhood policies used by the PSO core.

The project currently ships only the canonical global-best topology, but the
abstraction keeps the core loop ready for ring/local-best variants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Topology(ABC):
    """Return the social reference position used by one particle."""

    @abstractmethod
    def get_best_position(self, particle, swarm) -> NDArray[np.float64]:
        """Return the position that should guide the particle update."""
        raise NotImplementedError


class GlobalBestTopology(Topology):
    """Every particle is attracted to the swarm-wide best position."""

    def get_best_position(self, particle, swarm) -> NDArray[np.float64]:
        # `particle` is unused here, but keeping the same signature makes it
        # easy to plug in local neighbourhood topologies later.
        return swarm.global_best_position
