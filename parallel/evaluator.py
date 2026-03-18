"""parallel.evaluator

Evaluator implementations that enable concurrency / parallelism.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional

from numpy.typing import NDArray

from options.evaluator import FitnessEvaluator, SequentialEvaluator


class ThreadPoolEvaluator(FitnessEvaluator):
    """Compute fitness values in parallel using Python threads."""

    def __init__(
        self,
        objective_fn: Callable[[NDArray], float],
        max_workers: Optional[int] = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.max_workers = max_workers

    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        positions_list = list(positions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.objective_fn, positions_list))


__all__ = ["FitnessEvaluator", "SequentialEvaluator", "ThreadPoolEvaluator"]
