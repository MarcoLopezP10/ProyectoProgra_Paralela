"""core.pso

PSO algorithm — sequential baseline (V0).

Instruments:
- Per-iteration timing: fitness evaluation vs particle update
- Structured per-iteration logging (iter, best_fitness, t_eval, t_update)
- Total elapsed time
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from core.swarm import Swarm
from options.topology import Topology
from options.bounds import BoundsPolicy
from options.evaluator import FitnessEvaluator


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

        # Convergence history: best fitness per iteration
        self.history: list[float] = []

        # Timing breakdown accumulators
        self.time_eval: float = 0.0        # total time spent in fitness evaluation
        self.time_update: float = 0.0      # total time spent updating velocities/positions
        self.time_total: float = 0.0       # wall-clock time for the full run

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

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
        start_total = time.perf_counter()
        no_improve_counter = 0
        prev_best = np.inf

        self._log(
            f"PSO start | seed={self.seed} w={self.w} c1={self.c1} c2={self.c2} "
            f"n_particles={len(self.swarm.particles)} max_iters={self.max_iters}"
        )

        for it in range(self.max_iters):

            # ── 1. Evaluate fitness ───────────────────────────────────
            positions = self.swarm.get_positions()
            t0 = time.perf_counter()
            fitness = self.evaluator.evaluate(positions)
            t1 = time.perf_counter()
            iter_eval_time = t1 - t0
            self.time_eval += iter_eval_time

            # ── 2. Update personal and global bests ──────────────────
            self.swarm.update_global_best(positions, fitness)
            self.history.append(self.swarm.global_best_fitness)

            # ── 3. Early stopping check ───────────────────────────────
            improvement = abs(prev_best - self.swarm.global_best_fitness)
            if improvement < self.tol:
                no_improve_counter += 1
            else:
                no_improve_counter = 0

            # ── 4. Per-iteration structured log ───────────────────────
            if self.log_every > 0 and it % self.log_every == 0:
                self._log(
                    f"iter={it:>5d} | best={self.swarm.global_best_fitness:.6e} "
                    f"| t_eval={iter_eval_time*1000:.2f}ms "
                    f"| no_improve={no_improve_counter}/{self.patience}"
                )

            if no_improve_counter >= self.patience:
                self._log(f"Early stop at iter={it} (no improvement for {self.patience} iters)")
                break

            prev_best = self.swarm.global_best_fitness

            # ── 5. Update velocities and positions ────────────────────
            t2 = time.perf_counter()
            for p in self.swarm.particles:
                best_pos = self.topology.get_best_position(p, self.swarm)
                p.update_velocity(best_pos, self.w, self.c1, self.c2)
                p.update_position()
                p.position, p.velocity = self.bounds_handler.apply(
                    p.position, p.velocity
                )
            t3 = time.perf_counter()
            self.time_update += t3 - t2

        self.time_total = time.perf_counter() - start_total

        self._log(
            f"PSO done  | best={self.swarm.global_best_fitness:.6e} "
            f"| iters={len(self.history)} | total={self.time_total:.4f}s "
            f"| t_eval={self.time_eval:.4f}s ({100*self.time_eval/self.time_total:.1f}%) "
            f"| t_update={self.time_update:.4f}s ({100*self.time_update/self.time_total:.1f}%)"
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