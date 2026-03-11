"""options.evaluator

Fitness evaluation strategies.

PSO itself should not care *how* the objective function is evaluated (sequential,
threads, processes, asyncio, vectorization, ...). This module defines a small
interface (`FitnessEvaluator`) and a baseline implementation (`SequentialEvaluator`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, List

from numpy.typing import NDArray


class FitnessEvaluator(ABC):
    """Interface for computing fitness values for a batch of positions."""

    @abstractmethod
    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        """Return the fitness for each position (same order as input)."""
        raise NotImplementedError


class SequentialEvaluator(FitnessEvaluator):
    """Compute fitness values one-by-one in Python."""

    def __init__(self, objective_fn: Callable[[NDArray], float]):
        """Parameters
        ----------
        objective_fn : Callable
            Function to optimize. Should take a single position and return a scalar fitness.
        """
        
        self.objective_fn = objective_fn

    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        return [self.objective_fn(x) for x in positions]
