"""tests.test_pso

Minimal unit tests required by the project specification:

1. Reproducibility by seed        — same seed → identical results.
2. Bounds enforcement             — no particle ever leaves the search box.
3. Global-best monotonicity       — best fitness never gets worse.
4. Sphere convergence             — converges to ≈ 0 with reasonable params.
"""

from __future__ import annotations

import sys
import os

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from core.particle import Particle
from core.swarm import Swarm
from core.pso import PSO
from options.bounds import ClampBounds
from options.evaluator import SequentialEvaluator
from options.topology import GlobalBestTopology
from parallel.evaluator import AsyncioEvaluator, ProcessPoolEvaluator
from objectives.latency_mix import latency_mix
from objectives.sphere import sphere
from objectives.ackley import ackley
from objectives.rosenbrock import rosenbrock
from objectives.rastrigin import rastrigin


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_pso(
    objective_fn,
    dim: int = 2,
    bounds_lo: float = -5.0,
    bounds_hi: float = 5.0,
    n_particles: int = 30,
    max_iters: int = 200,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int = 42,
    tol: float = 1e-10,
    patience: int = 50,
    evaluator=None,
) -> PSO:
    """Build a ready-to-run PSO instance."""
    bounds = ([bounds_lo] * dim, [bounds_hi] * dim)
    rng = np.random.default_rng(seed)
    swarm = Swarm(n_particles=n_particles, dim=dim, bounds=bounds, rng=rng)
    return PSO(
        swarm=swarm,
        evaluator=SequentialEvaluator(objective_fn) if evaluator is None else evaluator,
        bounds_handler=ClampBounds(bounds[0], bounds[1]),
        topology=GlobalBestTopology(),
        w=w, c1=c1, c2=c2,
        max_iters=max_iters,
        seed=seed,
        tol=tol,
        patience=patience,
        log_every=0,   # silence logs during tests
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Reproducibility by seed
# ──────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """Same seed must produce byte-identical results across two independent runs."""

    def test_same_seed_same_fitness(self):
        """Two runs with the same seed reach the same best fitness."""
        pso_a = _make_pso(sphere, seed=42)
        pso_b = _make_pso(sphere, seed=42)
        _, fit_a, _, iters_a = pso_a.run()
        _, fit_b, _, iters_b = pso_b.run()
        assert fit_a == fit_b, (
            f"Same seed produced different fitness: {fit_a} vs {fit_b}"
        )
        assert iters_a == iters_b, (
            f"Same seed produced different iteration count: {iters_a} vs {iters_b}"
        )

    def test_same_seed_same_history(self):
        """Full convergence histories must be identical."""
        pso_a = _make_pso(ackley, seed=99)
        pso_b = _make_pso(ackley, seed=99)
        pso_a.run()
        pso_b.run()
        assert pso_a.history == pso_b.history, (
            "Same seed produced different convergence histories."
        )

    def test_different_seeds_different_results(self):
        """Different seeds should (almost always) produce different histories."""
        pso_a = _make_pso(sphere, seed=1)
        pso_b = _make_pso(sphere, seed=2)
        pso_a.run()
        pso_b.run()
        # It is astronomically unlikely that two different seeds produce
        # exactly the same full history — assert they differ.
        assert pso_a.history != pso_b.history, (
            "Different seeds produced identical histories — check RNG seeding."
        )

    def test_same_instance_can_be_rerun_from_initial_state(self):
        """Calling run() twice on the same PSO instance must restart from scratch."""
        pso = _make_pso(sphere, seed=42, dim=4, n_particles=20, max_iters=80)

        pos_a, fit_a, _, iters_a = pso.run()
        history_a = list(pso.history)

        pos_b, fit_b, _, iters_b = pso.run()
        history_b = list(pso.history)

        assert fit_b == pytest.approx(fit_a)
        assert iters_b == iters_a
        np.testing.assert_allclose(pos_b, pos_a)
        assert history_b == pytest.approx(history_a)

    @pytest.mark.parametrize("obj", [sphere, ackley, rosenbrock, rastrigin])
    def test_all_objectives_reproducible(self, obj):
        """Reproducibility holds for every benchmark function."""
        pso_a = _make_pso(obj, seed=7)
        pso_b = _make_pso(obj, seed=7)
        _, fit_a, _, _ = pso_a.run()
        _, fit_b, _, _ = pso_b.run()
        assert fit_a == fit_b


class TestMultiprocessingEvaluator:
    """V2 must preserve the same mathematical result as the sequential path."""

    def test_process_evaluator_matches_sequential_fitness_list(self):
        positions = [
            np.array([1.0, 1.0]),
            np.array([2.0, 0.0]),
            np.array([0.0, 3.0]),
            np.array([1.0, 2.0]),
        ]

        sequential = SequentialEvaluator(sphere).evaluate(positions)
        process_eval = ProcessPoolEvaluator(sphere, max_workers=2, batch_size=2)
        process_eval.open()
        try:
            parallel = process_eval.evaluate(positions)
        finally:
            process_eval.close()

        assert parallel == sequential

    def test_process_pso_matches_sequential_for_same_seed(self):
        pso_v0 = _make_pso(sphere, dim=4, n_particles=12, max_iters=60, seed=21)
        pso_v2 = _make_pso(
            sphere,
            dim=4,
            n_particles=12,
            max_iters=60,
            seed=21,
            evaluator=ProcessPoolEvaluator(sphere, max_workers=2, batch_size=3),
        )

        pos_v0, fit_v0, _, iters_v0 = pso_v0.run()
        pos_v2, fit_v2, _, iters_v2 = pso_v2.run()

        assert fit_v2 == pytest.approx(fit_v0)
        assert iters_v2 == iters_v0
        np.testing.assert_allclose(pos_v2, pos_v0)
        assert pso_v2.history == pytest.approx(pso_v0.history)

    def test_process_evaluator_rejects_non_picklable_objective(self):
        process_eval = ProcessPoolEvaluator(lambda x: float(np.sum(x)), max_workers=2)

        with pytest.raises(TypeError, match="picklable top-level objective function"):
            process_eval.open()


class TestAsyncioEvaluator:
    """V3 must preserve the same mathematical result while enabling async objectives."""

    def test_asyncio_evaluator_matches_sequential_fitness_list_for_sync_objective(self):
        positions = [
            np.array([1.0, 1.0]),
            np.array([2.0, 0.0]),
            np.array([0.0, 3.0]),
            np.array([1.0, 2.0]),
        ]

        sequential = SequentialEvaluator(sphere).evaluate(positions)
        async_eval = AsyncioEvaluator(sphere)
        parallel = async_eval.evaluate(positions)

        assert parallel == sequential

    def test_asyncio_evaluator_matches_sequential_fitness_list_for_latency_objective(self):
        positions = [
            np.array([0.5, 0.5]),
            np.array([1.5, -0.5]),
            np.array([-1.0, 2.0]),
        ]

        sequential = SequentialEvaluator(latency_mix).evaluate(positions)
        async_eval = AsyncioEvaluator(latency_mix)
        parallel = async_eval.evaluate(positions)

        assert parallel == pytest.approx(sequential)

    def test_asyncio_pso_matches_sequential_for_same_seed(self):
        pso_v0 = _make_pso(sphere, dim=4, n_particles=12, max_iters=60, seed=21)
        pso_v3 = _make_pso(
            sphere,
            dim=4,
            n_particles=12,
            max_iters=60,
            seed=21,
            evaluator=AsyncioEvaluator(sphere),
        )

        pos_v0, fit_v0, _, iters_v0 = pso_v0.run()
        pos_v3, fit_v3, _, iters_v3 = pso_v3.run()

        assert fit_v3 == pytest.approx(fit_v0)
        assert iters_v3 == iters_v0
        np.testing.assert_allclose(pos_v3, pos_v0)
        assert pso_v3.history == pytest.approx(pso_v0.history)


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Bounds enforcement
# ──────────────────────────────────────────────────────────────────────────────

class TestBoundsEnforcement:
    """No particle position should ever leave the search box."""

    def _run_and_collect_positions(self, dim, bounds_lo, bounds_hi, seed):
        """
        Run a manual PSO loop and collect every particle position at every
        iteration so we can check all of them, not just the final state.
        """
        bounds = ([bounds_lo] * dim, [bounds_hi] * dim)
        rng = np.random.default_rng(seed)
        swarm = Swarm(n_particles=30, dim=dim, bounds=bounds, rng=rng)
        evaluator    = SequentialEvaluator(sphere)
        bounds_h     = ClampBounds(bounds[0], bounds[1])
        topology     = GlobalBestTopology()
        all_positions = []

        for _ in range(100):
            positions = swarm.get_positions()
            fitness   = evaluator.evaluate(positions)
            swarm.update_global_best(positions, fitness)
            all_positions.extend(positions)

            for p in swarm.particles:
                best_pos = topology.get_best_position(p, swarm)
                p.update_velocity(best_pos, 0.7, 1.5, 1.5)
                p.update_position()
                p.position, p.velocity = bounds_h.apply(p.position, p.velocity)

        return np.array(all_positions), bounds_lo, bounds_hi

    def test_positions_stay_within_bounds_d2(self):
        all_pos, lo, hi = self._run_and_collect_positions(2, -5.0, 5.0, 42)
        assert np.all(all_pos >= lo), "Particle went below lower bound (d=2)."
        assert np.all(all_pos <= hi), "Particle went above upper bound (d=2)."

    def test_positions_stay_within_bounds_d10(self):
        all_pos, lo, hi = self._run_and_collect_positions(10, -5.0, 5.0, 42)
        assert np.all(all_pos >= lo), "Particle went below lower bound (d=10)."
        assert np.all(all_pos <= hi), "Particle went above upper bound (d=10)."

    def test_positions_stay_within_bounds_asymmetric(self):
        """Works with non-symmetric bounds too."""
        all_pos, lo, hi = self._run_and_collect_positions(2, -2.0, 8.0, 0)
        assert np.all(all_pos >= lo)
        assert np.all(all_pos <= hi)

    def test_clamp_bounds_apply_directly(self):
        """Unit test for ClampBounds.apply() in isolation."""
        cb = ClampBounds([-1.0, -1.0], [1.0, 1.0])

        # Position clearly outside bounds
        pos = np.array([2.0, -3.0])
        vel = np.array([0.5, -0.5])
        new_pos, new_vel = cb.apply(pos, vel)

        assert new_pos[0] == pytest.approx(1.0), "x₁ not clamped to upper bound."
        assert new_pos[1] == pytest.approx(-1.0), "x₂ not clamped to lower bound."
        # Velocity on hit axes must be zeroed
        assert new_vel[0] == pytest.approx(0.0), "Velocity not zeroed on x₁ collision."
        assert new_vel[1] == pytest.approx(0.0), "Velocity not zeroed on x₂ collision."

    def test_clamp_bounds_inside_box(self):
        """Position already inside box must not be modified."""
        cb = ClampBounds([-5.0, -5.0], [5.0, 5.0])
        pos = np.array([1.0, -2.0])
        vel = np.array([0.3, 0.3])
        new_pos, new_vel = cb.apply(pos, vel)
        np.testing.assert_array_equal(new_pos, pos)
        np.testing.assert_array_equal(new_vel, vel)

    def test_swarm_rejects_invalid_shape_or_size(self):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError):
            Swarm(n_particles=0, dim=2, bounds=([-5.0, -5.0], [5.0, 5.0]), rng=rng)

        with pytest.raises(ValueError):
            Swarm(n_particles=10, dim=0, bounds=([], []), rng=rng)

        with pytest.raises(ValueError):
            Swarm(n_particles=10, dim=2, bounds=([-5.0], [5.0, 5.0]), rng=rng)

        with pytest.raises(ValueError):
            Swarm(n_particles=10, dim=2, bounds=([1.0, -5.0], [1.0, 5.0]), rng=rng)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Global-best monotonicity
# ──────────────────────────────────────────────────────────────────────────────

class TestMonotonicity:
    """The global best fitness must never increase between iterations."""

    @pytest.mark.parametrize("obj", [sphere, ackley, rosenbrock, rastrigin])
    def test_history_is_non_increasing(self, obj):
        """history[i] >= history[i+1] for all i."""
        pso = _make_pso(obj, seed=42, max_iters=300)
        pso.run()
        history = pso.history
        assert len(history) > 1, "History is empty — PSO did not iterate."
        for i in range(len(history) - 1):
            assert history[i] >= history[i + 1] - 1e-15, (
                f"Global best worsened at iteration {i+1}: "
                f"{history[i]:.6e} → {history[i+1]:.6e}"
            )

    def test_monotonicity_high_dim(self):
        """Monotonicity must hold in higher dimensions too."""
        pso = _make_pso(sphere, dim=10, seed=0, max_iters=200)
        pso.run()
        history = pso.history
        for i in range(len(history) - 1):
            assert history[i] >= history[i + 1] - 1e-15, (
                f"Global best worsened at iter {i+1} (d=10)"
            )

    def test_global_best_never_worse_than_personal_bests(self):
        """Global best must be <= every particle's personal best."""
        pso = _make_pso(sphere, seed=5, max_iters=100)
        pso.run()
        gbest = pso.swarm.global_best_fitness
        for i, p in enumerate(pso.swarm.particles):
            assert gbest <= p.best_fitness + 1e-15, (
                f"Global best ({gbest:.6e}) worse than particle {i} "
                f"personal best ({p.best_fitness:.6e})."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Sphere convergence
# ──────────────────────────────────────────────────────────────────────────────

class TestSphereConvergence:
    """PSO must converge to ≈ 0 on the Sphere function with reasonable params."""

    def test_sphere_d2_converges(self):
        """Sphere d=2 must reach fitness < 1e-6 within 500 iterations."""
        pso = _make_pso(sphere, dim=2, seed=42, max_iters=500,
                        n_particles=40, w=0.7, c1=1.5, c2=1.5)
        _, fit, _, iters = pso.run()
        assert fit < 1e-6, (
            f"Sphere d=2 did not converge: fitness={fit:.6e} after {iters} iters."
        )

    def test_sphere_d10_converges(self):
        """Sphere d=10 must reach fitness < 1e-4 within 1000 iterations."""
        pso = _make_pso(sphere, dim=10, seed=42, max_iters=1000,
                        n_particles=60, w=0.7, c1=1.5, c2=1.5,
                        patience=80)
        _, fit, _, iters = pso.run()
        assert fit < 1e-4, (
            f"Sphere d=10 did not converge: fitness={fit:.6e} after {iters} iters."
        )

    def test_sphere_d30_converges(self):
        """Sphere d=30 must reach fitness < 1.0 within 2000 iterations."""
        pso = _make_pso(sphere, dim=30, seed=42, max_iters=2000,
                        n_particles=100, w=0.7, c1=1.5, c2=1.5,
                        patience=100)
        _, fit, _, iters = pso.run()
        assert fit < 1.0, (
            f"Sphere d=30 did not converge: fitness={fit:.6e} after {iters} iters."
        )

    def test_sphere_optimal_position_near_zero(self):
        """Best position found must be close to the true optimum (origin)."""
        pso = _make_pso(sphere, dim=2, seed=42, max_iters=500, n_particles=40)
        best_pos, best_fit, _, _ = pso.run()
        np.testing.assert_allclose(
            best_pos, np.zeros(2), atol=1e-3,
            err_msg=f"Best position {best_pos} far from origin. fitness={best_fit:.6e}",
        )

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
    def test_sphere_multiple_seeds(self, seed):
        """Sphere must converge regardless of seed."""
        pso = _make_pso(sphere, dim=2, seed=seed, max_iters=500, n_particles=40)
        _, fit, _, _ = pso.run()
        assert fit < 1e-5, (
            f"Sphere d=2 did not converge with seed={seed}: fitness={fit:.6e}"
        )
