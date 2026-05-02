"""parallel.evaluator

Evaluator implementations that enable concurrency / parallelism.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from options.bounds import ClampBounds
from options.evaluator import FitnessEvaluator, SequentialEvaluator
from options.topology import GlobalBestTopology


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


def _objective_key(objective_fn: Callable[[NDArray], float]) -> tuple[str, str]:
    """Return stable module/name identifiers for objective dispatch."""
    return (
        getattr(objective_fn, "__module__", ""),
        getattr(objective_fn, "__name__", ""),
    )


def _sphere_batch(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(positions**2, axis=1, dtype=float)


def _ackley_batch(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    n = positions.shape[1]
    sum_sq = np.sum(positions**2, axis=1, dtype=float)
    sum_cos = np.sum(np.cos(2 * np.pi * positions), axis=1, dtype=float)
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20.0
        + np.e
    )


def _rastrigin_batch(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    n = positions.shape[1]
    return 10.0 * n + np.sum(
        positions**2 - 10.0 * np.cos(2 * np.pi * positions),
        axis=1,
        dtype=float,
    )


def _rosenbrock_batch(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    left = positions[:, :-1]
    right = positions[:, 1:]
    return np.sum(100.0 * (right - left**2) ** 2 + (left - 1.0) ** 2, axis=1, dtype=float)


_VECTORIZED_OBJECTIVES: dict[tuple[str, str], Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    ("objectives.sphere", "sphere"): _sphere_batch,
    ("objectives.ackley", "ackley"): _ackley_batch,
    ("objectives.rastrigin", "rastrigin"): _rastrigin_batch,
    ("objectives.rosenbrock", "rosenbrock"): _rosenbrock_batch,
}


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


class VectorizedNumpyEvaluator(FitnessEvaluator):
    """Compute fitness and particle updates with NumPy batch operations."""

    def __init__(self, objective_fn: Callable[[NDArray], float]) -> None:
        self.objective_fn = objective_fn
        self._vectorized_objective = _VECTORIZED_OBJECTIVES.get(_objective_key(objective_fn))

    def _positions_to_array(self, positions: Iterable[NDArray]) -> NDArray[np.float64]:
        positions_list = list(positions)
        if not positions_list:
            return np.empty((0, 0), dtype=float)
        return np.asarray(positions_list, dtype=float)

    def _evaluate_array(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        if positions.size == 0:
            return np.empty((0,), dtype=float)
        if self._vectorized_objective is not None:
            return np.asarray(self._vectorized_objective(positions), dtype=float)
        return np.asarray([self.objective_fn(position) for position in positions], dtype=float)

    def evaluate(self, positions: Iterable[NDArray]) -> List[float]:
        positions_array = self._positions_to_array(positions)
        return self._evaluate_array(positions_array).astype(float).tolist()

    def _supports_vectorized_step(self, swarm, bounds_handler, topology) -> bool:
        return (
            self._vectorized_objective is not None
            and isinstance(bounds_handler, ClampBounds)
            and isinstance(topology, GlobalBestTopology)
            and len(swarm.particles) > 0
        )

    def _stack_swarm_state(self, swarm) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        positions = np.asarray([particle.position for particle in swarm.particles], dtype=float)
        velocities = np.asarray([particle.velocity for particle in swarm.particles], dtype=float)
        best_positions = np.asarray(
            [particle.best_position for particle in swarm.particles],
            dtype=float,
        )
        best_fitness = np.asarray(
            [particle.best_fitness for particle in swarm.particles],
            dtype=float,
        )
        vmax = np.asarray([particle.vmax for particle in swarm.particles], dtype=float)
        return positions, velocities, best_positions, best_fitness, vmax

    def _sample_random_coefficients(self, swarm) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r1 = np.empty((len(swarm.particles), swarm.dim), dtype=float)
        r2 = np.empty((len(swarm.particles), swarm.dim), dtype=float)
        for idx, particle in enumerate(swarm.particles):
            r1[idx] = particle.rng.random(particle.dim)
            r2[idx] = particle.rng.random(particle.dim)
        return r1, r2

    def _apply_clamp_vectorized(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        bounds_handler: ClampBounds,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.asarray(bounds_handler.lower, dtype=float)
        upper = np.asarray(bounds_handler.upper, dtype=float)
        clipped_positions = np.clip(positions, lower, upper)
        clipped_velocities = velocities.copy()
        hit_mask = clipped_positions != positions
        clipped_velocities[hit_mask] = 0.0
        return clipped_positions, clipped_velocities

    def _write_back_swarm_state(
        self,
        swarm,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        best_positions: NDArray[np.float64],
        best_fitness: NDArray[np.float64],
    ) -> None:
        for idx, particle in enumerate(swarm.particles):
            particle.position = positions[idx].copy()
            particle.velocity = velocities[idx].copy()
            particle.best_position = best_positions[idx].copy()
            particle.best_fitness = float(best_fitness[idx])

    def step(
        self,
        swarm,
        bounds_handler,
        topology,
        w: float,
        c1: float,
        c2: float,
    ) -> Optional[dict[str, float]]:
        if not self._supports_vectorized_step(swarm, bounds_handler, topology):
            return None

        positions, velocities, best_positions, best_fitness, vmax = self._stack_swarm_state(swarm)

        eval_start = time.perf_counter()
        fitness = self._evaluate_array(positions)
        improve_mask = fitness < best_fitness
        if np.any(improve_mask):
            best_positions[improve_mask] = positions[improve_mask]
            best_fitness[improve_mask] = fitness[improve_mask]

        global_best_position = (
            None
            if swarm.global_best_position is None
            else np.asarray(swarm.global_best_position, dtype=float).copy()
        )
        global_best_fitness = float(swarm.global_best_fitness)
        current_best_idx = int(np.argmin(fitness))
        current_best_fitness = float(fitness[current_best_idx])
        if global_best_position is None or current_best_fitness < global_best_fitness:
            global_best_position = positions[current_best_idx].copy()
            global_best_fitness = current_best_fitness
        eval_end = time.perf_counter()

        update_start = time.perf_counter()
        r1, r2 = self._sample_random_coefficients(swarm)
        cognitive = c1 * r1 * (best_positions - positions)
        social = c2 * r2 * (global_best_position - positions)
        velocities = w * velocities + cognitive + social
        velocities = np.clip(velocities, -vmax, vmax)

        positions = positions + velocities
        positions, velocities = self._apply_clamp_vectorized(positions, velocities, bounds_handler)
        update_end = time.perf_counter()

        self._write_back_swarm_state(swarm, positions, velocities, best_positions, best_fitness)
        swarm.global_best_position = global_best_position
        swarm.global_best_fitness = global_best_fitness

        return {
            "eval_s": float(eval_end - eval_start),
            "update_s": float(update_end - update_start),
        }


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
    if strategy in {"v4", "vectorized", "numpy"}:
        return VectorizedNumpyEvaluator(objective_fn)
    raise ValueError(f"Unknown evaluation strategy: {strategy}")


__all__ = [
    "FitnessEvaluator",
    "SequentialEvaluator",
    "ThreadPoolEvaluator",
    "ProcessPoolEvaluator",
    "AsyncioEvaluator",
    "VectorizedNumpyEvaluator",
    "build_evaluator",
]
