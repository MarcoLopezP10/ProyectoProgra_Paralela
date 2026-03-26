"""parallel.evaluator

Evaluator implementations that enable concurrency / parallelism.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

from numpy.typing import NDArray

from options.evaluator import FitnessEvaluator, SequentialEvaluator


def _evaluate_batch(
    objective_fn: Callable[[NDArray], float],
    positions_batch: Sequence[NDArray],
) -> List[float]:
    """Evaluate a batch inside a worker process."""
    return [objective_fn(position) for position in positions_batch]


def _chunked(items: Sequence[NDArray], batch_size: int) -> Iterator[List[NDArray]]:
    """Yield fixed-size batches while preserving the original order."""
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            break
        yield chunk


class ThreadPoolEvaluator(FitnessEvaluator):
    """Compute fitness values in parallel using Python threads."""

    def __init__(
        self,
        objective_fn: Callable[[NDArray], float],
        max_workers: Optional[int] = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.objective_fn = objective_fn
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None

    def open(self) -> None:
        """Create the pool once and reuse it across the full PSO run."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def close(self) -> None:
        """Shut down the pool after the PSO run finishes."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        positions_list = list(positions)
        if self._executor is None:
            self.open()
        return list(self._executor.map(self.objective_fn, positions_list))


class ProcessPoolEvaluator(FitnessEvaluator):
    """Compute fitness values in parallel using processes and batched IPC."""

    def __init__(
        self,
        objective_fn: Callable[[NDArray], float],
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.objective_fn = objective_fn
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._executor: Optional[ProcessPoolExecutor] = None

    def open(self) -> None:
        """Create the process pool once and reuse it across the full PSO run."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def close(self) -> None:
        """Shut down the pool after the PSO run finishes."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        positions_list = list(positions)
        if not positions_list:
            return []

        if self._executor is None:
            self.open()

        effective_workers = self.max_workers or os.cpu_count() or 1
        batch_size = self.batch_size or max(1, len(positions_list) // (effective_workers * 2))
        batches = list(_chunked(positions_list, batch_size))
        batch_results = self._executor.map(
            _evaluate_batch,
            [self.objective_fn] * len(batches),
            batches,
        )

        fitness: List[float] = []
        for result in batch_results:
            fitness.extend(result)
        return fitness


__all__ = [
    "FitnessEvaluator",
    "SequentialEvaluator",
    "ThreadPoolEvaluator",
    "ProcessPoolEvaluator",
]
