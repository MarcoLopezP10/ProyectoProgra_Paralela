"""core.pso

PSO algorithm core shared by V0, V1, V2, V3, and V4.

Instruments:
- Per-iteration timing: fitness evaluation, particle update, overhead
- Structured per-iteration logging
- Convergence history and iteration-level records for persistence/analysis
"""

from __future__ import annotations

import copy
import time
import logging
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from core.swarm import Swarm
from options.topology import Topology
from options.bounds import BoundsPolicy
from options.evaluator import FitnessEvaluator


def _trapezoid_area(values: list[float]) -> float:
    """Compute AUC without tripping NumPy 2.x deprecation warnings."""
    arr = np.asarray(values, dtype=float)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(arr))


class PSO:
    """
    Particle Swarm Optimization algorithm.

    The PSO class is strategy-agnostic: it delegates fitness evaluation,
    bounds enforcement, and neighbourhood topology to injected objects.
    This allows swapping V0/V1/V2/… without touching the core loop.
    """

    def __init__(
        self,
        swarm: Swarm,
        evaluator: FitnessEvaluator,
        bounds_handler: BoundsPolicy,
        topology: Topology,
        w: float,
        c1: float,
        c2: float,
        max_iters: int,
        seed: int = 42,
        tol: float = 1e-8,
        patience: int = 20,
        log_every: int = 10,
        logger: Optional[logging.Logger] = None,
        on_iteration: Optional[Callable[[int, Swarm, dict[str, float]], None]] = None,
    ):
        """
        Parameters
        ----------
        swarm : Swarm
        evaluator : FitnessEvaluator
        bounds_handler : BoundsPolicy
        topology : Topology
        w : float            Inertia weight.
        c1 : float           Cognitive coefficient.
        c2 : float           Social coefficient.
        max_iters : int      Maximum number of iterations.
        seed : int           Seed used to build the swarm (stored for reproducibility).
        tol : float          Minimum improvement considered significant.
        patience : int       Early-stop after this many non-improving iterations.
        log_every : int      Log a per-iteration line every N iterations (0 = never).
        logger : Logger      Optional Python logger.
        """
        self.swarm = swarm
        self.evaluator = evaluator
        self.bounds_handler = bounds_handler
        self.topology = topology

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.seed = seed
        self.tol = tol
        self.patience = patience
        self.log_every = log_every
        self.logger = logger
        self.on_iteration = on_iteration

        # Convergence history: best fitness per iteration
        self.history: list[float] = []
        self.iteration_records: list[dict[str, float]] = []

        # Timing breakdown accumulators
        self.time_eval: float = 0.0
        self.time_update: float = 0.0
        self.time_total: float = 0.0

        # Capture the initial optimiser state so repeated run() calls are deterministic.
        self._initial_swarm_state = self._snapshot_swarm_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _snapshot_swarm_state(self) -> dict[str, object]:
        """Capture the full mutable swarm state needed to rerun from scratch."""
        return {
            "global_best_position": (
                None
                if self.swarm.global_best_position is None
                else self.swarm.global_best_position.copy()
            ),
            "global_best_fitness": float(self.swarm.global_best_fitness),
            "particles": [
                {
                    "position": p.position.copy(),
                    "velocity": p.velocity.copy(),
                    "best_position": p.best_position.copy(),
                    "best_fitness": float(p.best_fitness),
                    "rng_state": copy.deepcopy(p.rng.bit_generator.state),
                }
                for p in self.swarm.particles
            ],
        }

    def _restore_swarm_state(self) -> None:
        """Restore the optimiser to the exact state it had before the first run."""
        snapshot = self._initial_swarm_state
        self.swarm.global_best_position = (
            None
            if snapshot["global_best_position"] is None
            else snapshot["global_best_position"].copy()
        )
        self.swarm.global_best_fitness = float(snapshot["global_best_fitness"])

        for particle, particle_state in zip(
            self.swarm.particles,
            snapshot["particles"],
        ):
            particle.position = particle_state["position"].copy()
            particle.velocity = particle_state["velocity"].copy()
            particle.best_position = particle_state["best_position"].copy()
            particle.best_fitness = float(particle_state["best_fitness"])
            particle.rng.bit_generator.state = copy.deepcopy(particle_state["rng_state"])

    def _reset_run_state(self) -> None:
        """Reset time series, timers, and swarm state before every run."""
        self._restore_swarm_state()
        self.history = []
        self.iteration_records = []
        self.time_eval = 0.0
        self.time_update = 0.0
        self.time_total = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[NDArray[np.float64], float, float, int]:
        """
        Run the PSO optimisation.

        Returns
        -------
        best_position : np.ndarray
        best_fitness : float
        elapsed_time : float   Wall-clock seconds for the full run.
        iterations : int       Number of iterations actually executed.
        """
        self._reset_run_state()

        start_total = time.perf_counter()
        no_improve_counter = 0
        prev_best = np.inf

        self._log(
            f"event=start seed={self.seed} w={self.w:.3f} c1={self.c1:.3f} "
            f"c2={self.c2:.3f} particles={len(self.swarm.particles)} "
            f"max_iters={self.max_iters} tol={self.tol:.1e} patience={self.patience}"
        )

        self.evaluator.open()
        try:
            for it in range(self.max_iters):
                start_iter = time.perf_counter()
                step_metrics = self.evaluator.step(
                    self.swarm,
                    self.bounds_handler,
                    self.topology,
                    self.w,
                    self.c1,
                    self.c2,
                )

                if step_metrics is None:
                    # ── 1. Evaluate fitness ───────────────────────────────
                    positions = self.swarm.get_positions()
                    t0 = time.perf_counter()
                    fitness = self.evaluator.evaluate(positions)
                    t1 = time.perf_counter()
                    iter_eval_time = t1 - t0
                    self.time_eval += iter_eval_time

                    # ── 2. Update personal and global bests ──────────────
                    self.swarm.update_global_best(positions, fitness)
                    self.history.append(self.swarm.global_best_fitness)

                    # ── 5. Update velocities and positions ───────────────
                    t2 = time.perf_counter()
                    for p in self.swarm.particles:
                        best_pos = self.topology.get_best_position(p, self.swarm)
                        p.update_velocity(best_pos, self.w, self.c1, self.c2)
                        p.update_position()
                        p.position, p.velocity = self.bounds_handler.apply(
                            p.position, p.velocity
                        )
                    t3 = time.perf_counter()
                    iter_update_time = t3 - t2
                else:
                    iter_eval_time = float(step_metrics.get("eval_s", 0.0))
                    iter_update_time = float(step_metrics.get("update_s", 0.0))
                    self.time_eval += iter_eval_time
                    self.time_update += iter_update_time
                    self.history.append(self.swarm.global_best_fitness)

                # ── 3. Early stopping check ───────────────────────────────
                improvement = abs(prev_best - self.swarm.global_best_fitness)
                if improvement < self.tol:
                    no_improve_counter += 1
                else:
                    no_improve_counter = 0

                if step_metrics is None:
                    self.time_update += iter_update_time

                iter_total_time = time.perf_counter() - start_iter
                iter_overhead_time = max(
                    iter_total_time - iter_eval_time - iter_update_time,
                    0.0,
                )

                iteration_record = {
                    "iter": float(it),
                    "best_fitness": float(self.swarm.global_best_fitness),
                    "eval_s": float(iter_eval_time),
                    "update_s": float(iter_update_time),
                    "overhead_s": float(iter_overhead_time),
                    "iter_s": float(iter_total_time),
                    "stall_count": float(no_improve_counter),
                }
                self.iteration_records.append(iteration_record)

                if self.on_iteration is not None:
                    self.on_iteration(it, self.swarm, iteration_record)

                # ── 6. Per-iteration structured log ───────────────────────
                if self.log_every > 0 and it % self.log_every == 0:
                    self._log(
                        f"event=iter iter={it:04d}/{self.max_iters} "
                        f"best={self.swarm.global_best_fitness:.6e} "
                        f"eval_ms={iter_eval_time*1000:.2f} "
                        f"update_ms={iter_update_time*1000:.2f} "
                        f"overhead_ms={iter_overhead_time*1000:.2f} "
                        f"iter_ms={iter_total_time*1000:.2f} "
                        f"stall={no_improve_counter}/{self.patience}"
                    )

                if no_improve_counter >= self.patience:
                    self._log(
                        f"event=stop reason=early_stop iter={it} "
                        f"stall={no_improve_counter}/{self.patience}"
                    )
                    break

                prev_best = self.swarm.global_best_fitness
        finally:
            self.evaluator.close()

        self.time_total = time.perf_counter() - start_total

        self._log(
            f"event=done best={self.swarm.global_best_fitness:.6e} "
            f"iters={len(self.history)} total_s={self.time_total:.4f} "
            f"eval_s={self.time_eval:.4f} update_s={self.time_update:.4f} "
            f"overhead_s={max(self.time_total - self.time_eval - self.time_update, 0.0):.4f} "
            f"pct_eval={100*self.time_eval/self.time_total:.1f} "
            f"pct_update={100*self.time_update/self.time_total:.1f}"
        )

        return (
            self.swarm.global_best_position,
            self.swarm.global_best_fitness,
            self.time_total,
            len(self.history),
        )

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------

    def timing_summary(self) -> dict[str, float]:
        """
        Return a breakdown of wall-clock time after run() has been called.

        Returns
        -------
        dict with keys: total, eval, update, overhead, pct_eval, pct_update
        """
        overhead = self.time_total - self.time_eval - self.time_update
        return {
            "total_s": self.time_total,
            "eval_s": self.time_eval,
            "update_s": self.time_update,
            "overhead_s": max(overhead, 0.0),
            "pct_eval": 100 * self.time_eval / self.time_total if self.time_total else 0.0,
            "pct_update": 100 * self.time_update / self.time_total if self.time_total else 0.0,
        }

    def area_under_curve(self, normalize: bool = False) -> float:
        """Return the convergence AUC over best-fitness history."""
        if not self.history:
            return 0.0
        if len(self.history) == 1:
            auc = float(self.history[0])
        else:
            auc = _trapezoid_area(self.history)
        if normalize and len(self.history) > 1:
            return auc / float(len(self.history) - 1)
        return auc

    def convergence_iteration(
        self,
        abs_tol: Optional[float] = None,
        rel_tol: float = 0.01,
    ) -> int:
        """
        Return the first iteration that reaches the final best value within tolerance.
        """
        if not self.history:
            return 0

        final_best = float(self.history[-1])
        margin = max(
            abs_tol if abs_tol is not None else self.tol,
            abs(final_best) * rel_tol,
        )
        for idx, value in enumerate(self.history):
            if value <= final_best + margin:
                return idx
        return len(self.history) - 1
