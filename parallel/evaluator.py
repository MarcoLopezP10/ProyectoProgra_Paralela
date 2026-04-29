"""parallel.evaluator

Evaluator implementations that enable concurrency / parallelism.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import pickle
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
    """Compute fitness values in parallel using processes and batched IPC.

    The objective function must be picklable, which in practice means using
    a top-level function rather than a lambda or local closure.
    """

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

    def _validate_objective(self) -> None:
        """Fail fast with a clear error if the objective cannot be pickled."""
        try:
            pickle.dumps(self.objective_fn)
        except Exception as exc:
            raise TypeError(
                "V2 multiprocessing requires a picklable top-level objective function."
            ) from exc

    def open(self) -> None:
        """Create the process pool once and reuse it across the full PSO run."""
        if self._executor is None:
            self._validate_objective()
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


class AsyncioEvaluator(FitnessEvaluator):
    """Compute fitness values with cooperative concurrency via asyncio.gather."""

    def __init__(self, objective_fn: Callable[[NDArray], float]) -> None:
        self.objective_fn = objective_fn

    def _resolve_async_evaluator(self) -> Callable[[NDArray], object]:
        async_evaluate = getattr(self.objective_fn, "async_evaluate", None)
        if callable(async_evaluate):
            return async_evaluate
        if inspect.iscoroutinefunction(self.objective_fn):
            return self.objective_fn

        async def _sync_wrapper(position: NDArray) -> float:
            return float(self.objective_fn(position))

        return _sync_wrapper

    async def _evaluate_async(self, positions: Sequence[NDArray]) -> List[float]:
        async_evaluate = self._resolve_async_evaluator()
        coroutines = [async_evaluate(position) for position in positions]
        results = await asyncio.gather(*coroutines)
        return [float(value) for value in results]

    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        positions_list = list(positions)
        if not positions_list:
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self._evaluate_async(positions_list))
            finally:
                new_loop.close()

        return asyncio.run(self._evaluate_async(positions_list))


def build_evaluator(
    strategy: str,
    objective_fn: Callable[[NDArray], float],
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> FitnessEvaluator:
    """Factory shared by scripts and grid search to keep strategies aligned."""
    strategy = strategy.lower()
    if strategy in {"v0", "sequential"}:
        return SequentialEvaluator(objective_fn)
    if strategy in {"v1", "thread", "threading"}:
        return ThreadPoolEvaluator(objective_fn, max_workers=max_workers)
    if strategy in {"v2", "process", "multiprocessing"}:
        return ProcessPoolEvaluator(
            objective_fn,
            max_workers=max_workers,
            batch_size=batch_size,
        )
    if strategy in {"v3", "async", "asyncio"}:
        return AsyncioEvaluator(objective_fn)
    raise ValueError(f"Unknown evaluation strategy: {strategy}")


__all__ = [
    "FitnessEvaluator",
    "SequentialEvaluator",
    "ThreadPoolEvaluator",
    "ProcessPoolEvaluator",
    "AsyncioEvaluator",
    "build_evaluator",
]
