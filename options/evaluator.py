"""options.evaluator

Fitness evaluation strategies.

PSO itself should not care *how* the objective function is evaluated (sequential,
threads, processes, asyncio, vectorization, ...). This module defines a small
interface (`FitnessEvaluator`) and a baseline implementation (`SequentialEvaluator`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

from numpy.typing import NDArray

if TYPE_CHECKING:
    from core.swarm import Swarm
    from options.bounds import BoundsPolicy
    from options.topology import Topology


class FitnessEvaluator(ABC):
    """Interface for computing fitness values for a batch of positions."""

    def open(self) -> None:
        """Allocate resources needed for a run."""
        return None

    def close(self) -> None:
        """Release resources allocated for a run."""
        return None

    def step(
        self,
        swarm: "Swarm",
        bounds_handler: "BoundsPolicy",
        topology: "Topology",
        w: float,
        c1: float,
        c2: float,
    ) -> Optional[dict[str, float]]:
        """Optionally execute one full PSO step and return timing metrics."""
        return None

    @abstractmethod
    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        """Return the fitness for each position (same order as input)."""
        raise NotImplementedError


class SequentialEvaluator(FitnessEvaluator):
    """Compute fitness values one-by-one in Python."""

    def __init__(self, objective_fn: Callable[[NDArray], float]) -> None:
        """
        Parameters
        ----------
        objective_fn : Callable
            Function to optimize. Should take a single position and return a scalar fitness.
        """
        self.objective_fn = objective_fn

    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        """Evaluate positions serially while preserving input order."""
        return [self.objective_fn(x) for x in positions]
