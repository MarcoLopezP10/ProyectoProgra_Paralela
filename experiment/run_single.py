"""experiment.run_single

Orchestrates a single PSO experiment:
  1. Optional grid search for hyperparameters.
  2. Run V0 (sequential), V1 (threading), V2 (multiprocessing), V3 (asyncio),
     and V4 (NumPy vectorized) with the same config and seed.
  3. Run PySwarm baseline for reference.
  4. Save structured results (JSON + CSV) and convergence plot.
  5. Structured logging with per-iteration timing breakdown.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from prettytable import PrettyTable
except ModuleNotFoundError:
    PrettyTable = None

from core.swarm import Swarm
from core.pso import PSO
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from parallel.evaluator import build_evaluator
from experiment.grid_search import recommended_profile, simple_grid_search
from baseline.pswarm import run_pyswarm_baseline
from utils.logger import setup_logger
from utils.metadata import collect_execution_metadata
from utils.io import (
    ExecutionMetadata,
    ExperimentSummary,
    MethodResult,
    TimingBreakdown,
    result_dir,
    save_iteration_metrics_csv,
    save_summary_json,
)
from viz.convergence import save_convergence_plot


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    seed: int = 42
    dim: int = 2
    bounds: Tuple[List[float], List[float]] = field(
        default_factory=lambda: ([-5.0, -5.0], [5.0, 5.0])
    )

    # Default hyperparameters (overridden by grid search if enabled)
    w: Optional[float] = None
    c1: Optional[float] = None
    c2: Optional[float] = None
    n_particles: Optional[int] = None
    use_grid_search: bool = False
    grid_metric: str = "final_fitness"
    grid_strategy: str = "v0"
    thread_max_workers: Optional[int] = None
    process_max_workers: Optional[int] = None
    batch_size: Optional[int] = None

    max_iters: Optional[int] = None
    tol: Optional[float] = None
    patience: Optional[int] = None
    vmax_ratio: Optional[float] = None
    log_every: int = 10       # log a line every N iters inside PSO (0 = off)

    # Output paths
    out_dir: str = "results/runs"
    plots_dir: str = "logs/convergence"
    log_dir: str = "logs"
    log_file: str = "pso_summary.log"
    repo_root: str = "."

    save_files: bool = True

    def bounds_for_dim(self) -> Tuple[List[float], List[float]]:
        """Return bounds resolved for cfg.dim while preserving explicit per-dim inputs."""
        lower = [float(value) for value in self.bounds[0]]
        upper = [float(value) for value in self.bounds[1]]

        if len(lower) == self.dim and len(upper) == self.dim:
            return lower, upper

        if len(lower) == 1 and len(upper) == 1:
            return lower * self.dim, upper * self.dim

        if (
            len(lower) == len(upper)
            and len(lower) > 1
            and all(value == lower[0] for value in lower)
            and all(value == upper[0] for value in upper)
        ):
            return [lower[0]] * self.dim, [upper[0]] * self.dim

        raise ValueError(
            "bounds must either provide one lower/upper value per dimension "
            "or a uniform bound that can be safely replicated."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_pso(
    cfg: RunConfig,
    bounds: Tuple[List[float], List[float]],
    w: float,
    c1: float,
    c2: float,
    n_particles: int,
    evaluator,
    logger: Optional[logging.Logger] = None,
) -> PSO:
    """Construct a PSO instance with the given evaluator, same seed always."""
    rng = np.random.default_rng(cfg.seed)
    swarm = Swarm(
        n_particles=n_particles,
        dim=cfg.dim,
        bounds=bounds,
        rng=rng,
        vmax_ratio=cfg.vmax_ratio if cfg.vmax_ratio is not None else 0.2,
    )
    return PSO(
        swarm=swarm,
        evaluator=evaluator,
        bounds_handler=ClampBounds(bounds[0], bounds[1]),
        topology=GlobalBestTopology(),
        w=w, c1=c1, c2=c2,
        max_iters=cfg.max_iters,
        seed=cfg.seed,
        tol=cfg.tol,
        patience=cfg.patience,
        log_every=cfg.log_every,
        logger=logger,
    )


def _resolve_run_config(cfg: RunConfig, objective_name: str) -> Dict[str, float]:
    """Resolve conservative defaults while keeping explicit user overrides."""
    profile = recommended_profile(objective_name, cfg.dim)
    return {
        "w": profile["w"] if cfg.w is None else cfg.w,
        "c1": profile["c1"] if cfg.c1 is None else cfg.c1,
        "c2": profile["c2"] if cfg.c2 is None else cfg.c2,
        "n_particles": profile["n_particles"] if cfg.n_particles is None else cfg.n_particles,
        "max_iters": profile["max_iters"] if cfg.max_iters is None else cfg.max_iters,
        "tol": profile["tol"] if cfg.tol is None else cfg.tol,
        "patience": profile["patience"] if cfg.patience is None else cfg.patience,
        "vmax_ratio": profile["vmax_ratio"] if cfg.vmax_ratio is None else cfg.vmax_ratio,
    }


def _resolve_winner(
    results: list[tuple[str, Optional[float], Optional[float], str]],
) -> str:
    """Best fitness first; time as tie-break; unavailable methods are ignored."""
    available = [
        (name, fitness, time_s)
        for name, fitness, time_s, status in results
        if status == "ok" and fitness is not None and time_s is not None
    ]
    if not available:
        return "Unavailable"

    best_fitness = min(f for _, f, _ in available)
    candidates = [
        (n, t)
        for n, f, t in available
        if np.isclose(f, best_fitness, atol=1e-15, rtol=0.0)
    ]
    if len(candidates) == 1:
        return candidates[0][0]
    best_time = min(t for _, t in candidates)
    winners = [
        n for n, t in candidates if np.isclose(t, best_time, atol=1e-12, rtol=0.0)
    ]
    return winners[0] if len(winners) == 1 else "Tie"


def _build_method_result(
    strategy: str,
    pso: Optional[PSO],
    best_fitness: Optional[float],
    iterations: Optional[int],
    total_s: Optional[float] = None,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    status: str = "ok",
    error: Optional[str] = None,
) -> MethodResult:
    """Create a persistable method result from a PSO run or a baseline value."""
    if pso is None:
        timing = TimingBreakdown(total_s=float(total_s or 0.0))
        return MethodResult(
            status=status,
            strategy=strategy,
            best_fitness=None if best_fitness is None else float(best_fitness),
            iterations=None if iterations is None else int(iterations),
            auc=None,
            convergence_iteration=None,
            timing=timing,
            max_workers=max_workers,
            batch_size=batch_size,
            error=error,
        )

    return MethodResult(
        status=status,
        strategy=strategy,
        best_fitness=None if best_fitness is None else float(best_fitness),
        iterations=None if iterations is None else int(iterations),
        auc=float(pso.area_under_curve()),
        convergence_iteration=int(pso.convergence_iteration()),
        timing=TimingBreakdown(**pso.timing_summary()),
        max_workers=max_workers,
        batch_size=batch_size,
        error=error,
    )


def _format_float(value: Optional[float], fmt: str) -> str:
    if value is None:
        return "-"
    return format(float(value), fmt)


def _format_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return str(int(value))


def _format_speedup(reference_time: float, method_time: Optional[float]) -> str:
    if method_time is None or method_time <= 0:
        return "-"
    return f"{reference_time / method_time:.3f}x"


def _print_table(
    name: str, dim: int, seed: int,
    w: float, c1: float, c2: float, n_particles: int,
    v0_fit: float, v0_iters: int, v0_time: float,
    v0_pct_eval: float, v0_pct_update: float, v0_conv_iter: int,
    v1_fit: float, v1_iters: int, v1_time: float,
    v1_pct_eval: float, v1_pct_update: float, v1_conv_iter: int,
    v2_fit: Optional[float], v2_iters: Optional[int], v2_time: Optional[float],
    v2_pct_eval: float, v2_pct_update: float, v2_conv_iter: Optional[int],
    v3_fit: Optional[float], v3_iters: Optional[int], v3_time: Optional[float],
    v3_pct_eval: float, v3_pct_update: float, v3_conv_iter: Optional[int],
    v4_fit: Optional[float], v4_iters: Optional[int], v4_time: Optional[float],
    v4_pct_eval: float, v4_pct_update: float, v4_conv_iter: Optional[int],
    base_fit: Optional[float], base_iters: Optional[int], base_time: Optional[float],
    winner: str,
    thread_max_workers: Optional[int],
    process_max_workers: Optional[int],
    batch_size: Optional[int],
    v2_status: str = "ok",
    v3_status: str = "ok",
    v4_status: str = "ok",
    baseline_status: str = "ok",
) -> None:
    v1_speedup = _format_speedup(v0_time, v1_time)
    v2_speedup = _format_speedup(v0_time, v2_time)
    v3_speedup = _format_speedup(v0_time, v3_time)
    v4_speedup = _format_speedup(v0_time, v4_time)
    baseline_speedup = _format_speedup(v0_time, base_time)

    hyper_rows = [
        ["objective", name],
        ["dim", dim],
        ["seed", seed],
        ["w", f"{w:.3f}"],
        ["c1", f"{c1:.3f}"],
        ["c2", f"{c2:.3f}"],
        ["n_particles", n_particles],
        ["thread_max_workers", thread_max_workers or "default"],
        ["process_max_workers", process_max_workers or "default"],
        ["batch_size", batch_size or "auto"],
    ]
    result_rows = [
        [
            "V0 Sequential",
            f"ok | {_format_float(v0_fit, '.6e')}",
            v0_iters,
            v0_conv_iter,
            f"{v0_time:.4f}",
            "1.000x",
            f"{v0_pct_eval:.1f}%",
            f"{v0_pct_update:.1f}%",
        ],
        [
            "V1 Threading",
            f"ok | {_format_float(v1_fit, '.6e')}",
            v1_iters,
            v1_conv_iter,
            f"{v1_time:.4f}",
            v1_speedup,
            f"{v1_pct_eval:.1f}%",
            f"{v1_pct_update:.1f}%",
        ],
        [
            "V2 Multiprocessing",
            f"{v2_status} | {_format_float(v2_fit, '.6e')}",
            _format_int(v2_iters),
            _format_int(v2_conv_iter),
            _format_float(v2_time, ".4f"),
            v2_speedup,
            "-" if v2_status != "ok" else f"{v2_pct_eval:.1f}%",
            "-" if v2_status != "ok" else f"{v2_pct_update:.1f}%",
        ],
        [
            "V3 Asyncio",
            f"{v3_status} | {_format_float(v3_fit, '.6e')}",
            _format_int(v3_iters),
            _format_int(v3_conv_iter),
            _format_float(v3_time, ".4f"),
            v3_speedup,
            "-" if v3_status != "ok" else f"{v3_pct_eval:.1f}%",
            "-" if v3_status != "ok" else f"{v3_pct_update:.1f}%",
        ],
        [
            "V4 Vectorized",
            f"{v4_status} | {_format_float(v4_fit, '.6e')}",
            _format_int(v4_iters),
            _format_int(v4_conv_iter),
            _format_float(v4_time, ".4f"),
            v4_speedup,
            "-" if v4_status != "ok" else f"{v4_pct_eval:.1f}%",
            "-" if v4_status != "ok" else f"{v4_pct_update:.1f}%",
        ],
        [
            "PySwarm baseline",
            f"{baseline_status} | {_format_float(base_fit, '.6e')}",
            _format_int(base_iters),
            "-",
            _format_float(base_time, ".4f"),
            baseline_speedup,
            "-",
            "-",
        ],
    ]
    summary_rows = [
        ["Winner", winner],
        ["V1 speedup vs V0", v1_speedup],
        ["V2 speedup vs V0", v2_speedup],
        ["V3 speedup vs V0", v3_speedup],
        ["V4 speedup vs V0", v4_speedup],
    ]

    if PrettyTable is not None:
        t = PrettyTable(["Hyperparameter", "Value"])
        for r in hyper_rows: t.add_row(r)
        print(t)

        t = PrettyTable(["Method", "Best fitness", "Iters", "Conv iter", "Time (s)", "Speedup", "% eval", "% update"])
        for r in result_rows: t.add_row(r)
        print(t)

        t = PrettyTable(["Metric", "Value"])
        for r in summary_rows: t.add_row(r)
        print(t)
    else:
        for section, headers, rows in [
            ("Hyperparameters", ["Param", "Value"], hyper_rows),
            ("Results", ["Method", "Best fitness", "Iters", "Conv iter", "Time (s)", "Speedup", "% eval", "% update"], result_rows),
            ("Summary", ["Metric", "Value"], summary_rows),
        ]:
            widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
            sep = "-+-".join("-" * w for w in widths)
            fmt = lambda row: " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))
            print(f"\n{section}")
            print(fmt(headers))
            print(sep)
            for r in rows:
                print(fmt(r))


def _run_optional_pso(
    *,
    cfg: RunConfig,
    bounds: Tuple[List[float], List[float]],
    w: float,
    c1: float,
    c2: float,
    n_particles: int,
    strategy: str,
    objective: Callable[[np.ndarray], float],
    logger: logging.LoggerAdapter,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    allow_unavailable: bool = False,
) -> Dict[str, Any]:
    pso = _build_pso(
        cfg,
        bounds,
        w,
        c1,
        c2,
        n_particles,
        build_evaluator(
            strategy,
            objective,
            max_workers=max_workers,
            batch_size=batch_size,
        ),
        logger,
    )
    try:
        best_pos, best_fit, time_s, iters = pso.run()
    except Exception as exc:
        if not allow_unavailable:
            raise

        error = f"{type(exc).__name__}: {exc}"
        logger.warning(f"event=unavailable reason={error}")
        return {
            "status": "unavailable",
            "error": error,
            "pso": None,
            "best_pos": None,
            "best_fit": None,
            "time_s": None,
            "iters": None,
            "timing": TimingBreakdown().__dict__.copy(),
            "auc": None,
            "convergence_iter": None,
            "history": [],
            "iteration_records": [],
        }

    return {
        "status": "ok",
        "error": None,
        "pso": pso,
        "best_pos": best_pos,
        "best_fit": best_fit,
        "time_s": time_s,
        "iters": iters,
        "timing": pso.timing_summary(),
        "auc": pso.area_under_curve(),
        "convergence_iter": pso.convergence_iteration(),
        "history": list(pso.history),
        "iteration_records": list(pso.iteration_records),
    }


def _run_optional_baseline(
    *,
    objective: Callable[[np.ndarray], float],
    bounds: Tuple[List[float], List[float]],
    w: float,
    c1: float,
    c2: float,
    n_particles: int,
    max_iters: int,
    seed: int,
    logger: logging.LoggerAdapter,
) -> Dict[str, Any]:
    try:
        best_pos, best_fit, time_s, iters = run_pyswarm_baseline(
            objective_function=objective,
            bounds=bounds,
            w=w,
            c1=c1,
            c2=c2,
            n_particles=n_particles,
            max_iters=max_iters,
            seed=seed,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning(f"event=unavailable reason={error}")
        return {
            "status": "unavailable",
            "error": error,
            "best_pos": None,
            "best_fit": None,
            "time_s": None,
            "iters": None,
        }

    return {
        "status": "ok",
        "error": None,
        "best_pos": best_pos,
        "best_fit": best_fit,
        "time_s": time_s,
        "iters": iters,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_one_objective(
    objective: Callable[[np.ndarray], float],
    cfg: RunConfig,
) -> Dict[str, object]:
    """
    Run a full V0+V1+V2+V3+V4 experiment for a single objective function.

    The seed is passed through to every PSO instance so that V0, V1, V2, V3,
    and V4 start from exactly the same swarm state and results are comparable.

    Parameters
    ----------
    objective : Callable
    cfg : RunConfig

    Returns
    -------
    dict with all results, histories, and paths.
    """
    logger = setup_logger(log_dir=cfg.log_dir, log_file=cfg.log_file)
    name = getattr(objective, "__name__", "objective")
    execution_metadata = ExecutionMetadata(**collect_execution_metadata(cfg.repo_root))
    resolved = _resolve_run_config(cfg, name)
    cfg.max_iters = int(resolved["max_iters"])
    cfg.tol = float(resolved["tol"])
    cfg.patience = int(resolved["patience"])
    cfg.vmax_ratio = float(resolved["vmax_ratio"])

    # Bounds always scaled to cfg.dim
    bounds = cfg.bounds_for_dim()

    if cfg.save_files:
        os.makedirs(cfg.plots_dir, exist_ok=True)

    # ── 1. Hyperparameter resolution ─────────────────────────────────
    if cfg.use_grid_search:
        best = simple_grid_search(
            objective_fn=objective,
            dim=cfg.dim,
            bounds=bounds,
            max_iters=cfg.max_iters,
            seed=cfg.seed,
            vmax_ratio=cfg.vmax_ratio,
            strategy=cfg.grid_strategy,
            metric=cfg.grid_metric,
            max_workers=cfg.process_max_workers if cfg.grid_strategy.lower() == "v2" else cfg.thread_max_workers,
            batch_size=cfg.batch_size,
        )
        w, c1, c2 = float(best["w"]), float(best["c1"]), float(best["c2"])
        n_particles = int(best["n_particles"])
        hyperparam_source = "grid_search"
    else:
        w = float(resolved["w"])
        c1 = float(resolved["c1"])
        c2 = float(resolved["c2"])
        n_particles = int(resolved["n_particles"])
        hyperparam_source = "fixed_config"

    tag = f"{name.upper()} d={cfg.dim} seed={cfg.seed}"
    log = logging.LoggerAdapter(logger, {"objective": tag, "method": "RUN"})
    log.info(
        f"event=config source={hyperparam_source} objective={name} dim={cfg.dim} "
        f"seed={cfg.seed} w={w:.3f} c1={c1:.3f} c2={c2:.3f} "
        f"particles={n_particles} vmax_ratio={cfg.vmax_ratio:.3f} "
        f"max_iters={cfg.max_iters} patience={cfg.patience} tol={cfg.tol:.1e} "
        f"grid_strategy={cfg.grid_strategy} grid_metric={cfg.grid_metric}"
    )

    # ── 2. V0 — Sequential ───────────────────────────────────────────
    v0_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V0"})
    v0_result = _run_optional_pso(
        cfg=cfg,
        bounds=bounds,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        strategy="v0",
        objective=objective,
        logger=v0_log,
    )

    # ── 3. V1 — Threading ────────────────────────────────────────────
    v1_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V1"})
    v1_result = _run_optional_pso(
        cfg=cfg,
        bounds=bounds,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        strategy="v1",
        objective=objective,
        logger=v1_log,
        max_workers=cfg.thread_max_workers,
    )

    # ── 4. V2 — Multiprocessing ───────────────────────────────────────
    v2_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V2"})
    v2_result = _run_optional_pso(
        cfg=cfg,
        bounds=bounds,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        strategy="v2",
        objective=objective,
        logger=v2_log,
        max_workers=cfg.process_max_workers,
        batch_size=cfg.batch_size,
        allow_unavailable=True,
    )

    # ── 5. V3 — Asyncio ───────────────────────────────────────────────
    v3_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V3"})
    v3_result = _run_optional_pso(
        cfg=cfg,
        bounds=bounds,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        strategy="v3",
        objective=objective,
        logger=v3_log,
        allow_unavailable=True,
    )

    # ── 6. V4 — NumPy vectorized ──────────────────────────────────────
    v4_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V4"})
    v4_result = _run_optional_pso(
        cfg=cfg,
        bounds=bounds,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        strategy="v4",
        objective=objective,
        logger=v4_log,
        allow_unavailable=True,
    )

    # ── 7. PySwarm baseline ───────────────────────────────────────────
    baseline_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "BASE"})
    baseline_result = _run_optional_baseline(
        objective=objective,
        bounds=bounds,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        max_iters=cfg.max_iters,
        seed=cfg.seed,
        logger=baseline_log,
    )

    # ── 8. Winner ─────────────────────────────────────────────────────
    winner = _resolve_winner([
        ("V0 Sequential", v0_result["best_fit"], v0_result["time_s"], v0_result["status"]),
        ("V1 Threading", v1_result["best_fit"], v1_result["time_s"], v1_result["status"]),
        ("V2 Multiprocessing", v2_result["best_fit"], v2_result["time_s"], v2_result["status"]),
        ("V3 Asyncio", v3_result["best_fit"], v3_result["time_s"], v3_result["status"]),
        ("V4 Vectorized", v4_result["best_fit"], v4_result["time_s"], v4_result["status"]),
        ("PySwarm", baseline_result["best_fit"], baseline_result["time_s"], baseline_result["status"]),
    ])

    # ── 9. Log summary ────────────────────────────────────────────────
    log.info(
        f"event=summary method=V0 fit={v0_result['best_fit']:.6e} iters={v0_result['iters']} "
        f"time_s={v0_result['time_s']:.4f} eval_pct={v0_result['timing']['pct_eval']:.1f} "
        f"update_pct={v0_result['timing']['pct_update']:.1f} auc={v0_result['auc']:.6e} "
        f"conv_iter={v0_result['convergence_iter']}"
    )
    log.info(
        f"event=summary method=V1 fit={v1_result['best_fit']:.6e} iters={v1_result['iters']} "
        f"time_s={v1_result['time_s']:.4f} eval_pct={v1_result['timing']['pct_eval']:.1f} "
        f"update_pct={v1_result['timing']['pct_update']:.1f} auc={v1_result['auc']:.6e} "
        f"conv_iter={v1_result['convergence_iter']} "
        f"speedup_vs_v0={_format_speedup(v0_result['time_s'], v1_result['time_s'])}"
    )
    if v2_result["status"] == "ok":
        log.info(
            f"event=summary method=V2 fit={v2_result['best_fit']:.6e} iters={v2_result['iters']} "
            f"time_s={v2_result['time_s']:.4f} eval_pct={v2_result['timing']['pct_eval']:.1f} "
            f"update_pct={v2_result['timing']['pct_update']:.1f} auc={v2_result['auc']:.6e} "
            f"conv_iter={v2_result['convergence_iter']} "
            f"speedup_vs_v0={_format_speedup(v0_result['time_s'], v2_result['time_s'])} "
            f"process_workers={cfg.process_max_workers or 'default'} "
            f"batch_size={cfg.batch_size or 'auto'}"
        )
    if v3_result["status"] == "ok":
        log.info(
            f"event=summary method=V3 fit={v3_result['best_fit']:.6e} iters={v3_result['iters']} "
            f"time_s={v3_result['time_s']:.4f} eval_pct={v3_result['timing']['pct_eval']:.1f} "
            f"update_pct={v3_result['timing']['pct_update']:.1f} auc={v3_result['auc']:.6e} "
            f"conv_iter={v3_result['convergence_iter']} "
            f"speedup_vs_v0={_format_speedup(v0_result['time_s'], v3_result['time_s'])}"
        )
    if v4_result["status"] == "ok":
        log.info(
            f"event=summary method=V4 fit={v4_result['best_fit']:.6e} iters={v4_result['iters']} "
            f"time_s={v4_result['time_s']:.4f} eval_pct={v4_result['timing']['pct_eval']:.1f} "
            f"update_pct={v4_result['timing']['pct_update']:.1f} auc={v4_result['auc']:.6e} "
            f"conv_iter={v4_result['convergence_iter']} "
            f"speedup_vs_v0={_format_speedup(v0_result['time_s'], v4_result['time_s'])}"
        )
    if baseline_result["status"] == "ok":
        log.info(
            f"event=summary method=PySwarm fit={baseline_result['best_fit']:.6e} "
            f"iters={baseline_result['iters']} time_s={baseline_result['time_s']:.4f}"
        )
    log.info(f"event=summary winner={winner}")

    # ── 10. Console table ─────────────────────────────────────────────
    _print_table(
        name=name, dim=cfg.dim, seed=cfg.seed,
        w=w, c1=c1, c2=c2, n_particles=n_particles,
        v0_fit=float(v0_result["best_fit"]), v0_iters=int(v0_result["iters"]), v0_time=float(v0_result["time_s"]),
        v0_pct_eval=v0_result["timing"]["pct_eval"], v0_pct_update=v0_result["timing"]["pct_update"],
        v0_conv_iter=int(v0_result["convergence_iter"]),
        v1_fit=float(v1_result["best_fit"]), v1_iters=int(v1_result["iters"]), v1_time=float(v1_result["time_s"]),
        v1_pct_eval=v1_result["timing"]["pct_eval"], v1_pct_update=v1_result["timing"]["pct_update"],
        v1_conv_iter=int(v1_result["convergence_iter"]),
        v2_fit=v2_result["best_fit"], v2_iters=v2_result["iters"], v2_time=v2_result["time_s"],
        v2_pct_eval=v2_result["timing"]["pct_eval"], v2_pct_update=v2_result["timing"]["pct_update"],
        v2_conv_iter=v2_result["convergence_iter"],
        v3_fit=v3_result["best_fit"], v3_iters=v3_result["iters"], v3_time=v3_result["time_s"],
        v3_pct_eval=v3_result["timing"]["pct_eval"], v3_pct_update=v3_result["timing"]["pct_update"],
        v3_conv_iter=v3_result["convergence_iter"],
        v4_fit=v4_result["best_fit"], v4_iters=v4_result["iters"], v4_time=v4_result["time_s"],
        v4_pct_eval=v4_result["timing"]["pct_eval"], v4_pct_update=v4_result["timing"]["pct_update"],
        v4_conv_iter=v4_result["convergence_iter"],
        base_fit=baseline_result["best_fit"], base_iters=baseline_result["iters"], base_time=baseline_result["time_s"],
        winner=winner, thread_max_workers=cfg.thread_max_workers,
        process_max_workers=cfg.process_max_workers, batch_size=cfg.batch_size,
        v2_status=v2_result["status"], v3_status=v3_result["status"], v4_status=v4_result["status"],
        baseline_status=baseline_result["status"],
    )

    # ── 11. Save results ──────────────────────────────────────────────
    if cfg.save_files:
        # Convergence plot
        plot_path = os.path.join(cfg.plots_dir, f"{name}_d{cfg.dim}_s{cfg.seed}_convergence.png")
        save_convergence_plot(
            history=v0_result["history"],
            baseline_final_fitness=baseline_result["best_fit"],
            title=f"Convergence — {name} d={cfg.dim} seed={cfg.seed}",
            out_path=plot_path,
            threaded_history=v1_result["history"],
            process_history=v2_result["history"],
            asyncio_history=v3_result["history"],
            vectorized_history=v4_result["history"],
        )

        # Structured results directory: results/runs/sphere_d2_s42/
        rdir = result_dir(cfg.out_dir, name, cfg.dim, cfg.seed)
        os.makedirs(rdir, exist_ok=True)

        summary = ExperimentSummary(
            objective=name,
            dim=cfg.dim,
            bounds_lower=list(bounds[0]),
            bounds_upper=list(bounds[1]),
            seed=cfg.seed,
            hyperparam_source=hyperparam_source,
            w=w, c1=c1, c2=c2,
            n_particles=n_particles,
            max_iters=cfg.max_iters,
            tol=cfg.tol,
            patience=cfg.patience,
            v0=_build_method_result(
                "V0 Sequential",
                v0_result["pso"],
                v0_result["best_fit"],
                v0_result["iters"],
                status=v0_result["status"],
                error=v0_result["error"],
            ),
            v1=_build_method_result(
                "V1 Threading",
                v1_result["pso"],
                v1_result["best_fit"],
                v1_result["iters"],
                max_workers=cfg.thread_max_workers,
                status=v1_result["status"],
                error=v1_result["error"],
            ),
            v2=_build_method_result(
                "V2 Multiprocessing",
                v2_result["pso"],
                v2_result["best_fit"],
                v2_result["iters"],
                max_workers=cfg.process_max_workers,
                batch_size=cfg.batch_size,
                total_s=v2_result["time_s"],
                status=v2_result["status"],
                error=v2_result["error"],
            ),
            v3=_build_method_result(
                "V3 Asyncio",
                v3_result["pso"],
                v3_result["best_fit"],
                v3_result["iters"],
                total_s=v3_result["time_s"],
                status=v3_result["status"],
                error=v3_result["error"],
            ),
            v4=_build_method_result(
                "V4 Vectorized",
                v4_result["pso"],
                v4_result["best_fit"],
                v4_result["iters"],
                total_s=v4_result["time_s"],
                status=v4_result["status"],
                error=v4_result["error"],
            ),
            baseline=_build_method_result(
                "PySwarm baseline",
                None,
                baseline_result["best_fit"],
                baseline_result["iters"],
                total_s=baseline_result["time_s"],
                status=baseline_result["status"],
                error=baseline_result["error"],
            ),
            winner=winner,
            selected_grid_metric=cfg.grid_metric if cfg.use_grid_search else None,
            execution=execution_metadata,
            notes=(
                "V1 uses ThreadPoolExecutor and is mainly a concurrency baseline. "
                "V2 uses ProcessPoolExecutor with batched particle evaluation to "
                "reduce pickling and IPC overhead while preserving the same PSO "
                "search behaviour as V0. V3 uses asyncio.gather and is mainly "
                "expected to help when the objective exposes cooperative latency. "
                "V4 uses NumPy vectorization for the evaluation/update path on "
                "supported numeric objectives and otherwise falls back safely."
            ),
        )
        summary_path = os.path.join(rdir, "summary.json")
        history_v0_path = os.path.join(rdir, "history_v0.csv")
        history_v1_path = os.path.join(rdir, "history_v1.csv")
        history_v2_path = os.path.join(rdir, "history_v2.csv")
        history_v3_path = os.path.join(rdir, "history_v3.csv")
        history_v4_path = os.path.join(rdir, "history_v4.csv")

        save_summary_json(summary, summary_path)
        save_iteration_metrics_csv(v0_result["iteration_records"], history_v0_path)
        save_iteration_metrics_csv(v1_result["iteration_records"], history_v1_path)
        if v2_result["status"] == "ok":
            save_iteration_metrics_csv(v2_result["iteration_records"], history_v2_path)
        if v3_result["status"] == "ok":
            save_iteration_metrics_csv(v3_result["iteration_records"], history_v3_path)
        if v4_result["status"] == "ok":
            save_iteration_metrics_csv(v4_result["iteration_records"], history_v4_path)

    else:
        plot_path = None

    return {
        "objective": name,
        "dim": cfg.dim,
        "seed": cfg.seed,
        "hyperparams": {"w": w, "c1": c1, "c2": c2, "n_particles": n_particles},
        "v0": {"status": v0_result["status"], "error": v0_result["error"], "best_pos": v0_result["best_pos"], "best_fit": v0_result["best_fit"],
               "time_s": v0_result["time_s"], "iters": v0_result["iters"], "timing": v0_result["timing"],
               "auc": v0_result["auc"],
               "convergence_iter": v0_result["convergence_iter"]},
        "v1": {"status": v1_result["status"], "error": v1_result["error"], "best_pos": v1_result["best_pos"], "best_fit": v1_result["best_fit"],
               "time_s": v1_result["time_s"], "iters": v1_result["iters"], "timing": v1_result["timing"],
               "max_workers": cfg.thread_max_workers,
               "auc": v1_result["auc"],
               "convergence_iter": v1_result["convergence_iter"]},
        "v2": {"status": v2_result["status"], "error": v2_result["error"], "best_pos": v2_result["best_pos"], "best_fit": v2_result["best_fit"],
               "time_s": v2_result["time_s"], "iters": v2_result["iters"], "timing": v2_result["timing"],
               "max_workers": cfg.process_max_workers, "batch_size": cfg.batch_size,
               "auc": v2_result["auc"],
               "convergence_iter": v2_result["convergence_iter"]},
        "v3": {"status": v3_result["status"], "error": v3_result["error"], "best_pos": v3_result["best_pos"], "best_fit": v3_result["best_fit"],
               "time_s": v3_result["time_s"], "iters": v3_result["iters"], "timing": v3_result["timing"],
               "auc": v3_result["auc"],
               "convergence_iter": v3_result["convergence_iter"]},
        "v4": {"status": v4_result["status"], "error": v4_result["error"], "best_pos": v4_result["best_pos"], "best_fit": v4_result["best_fit"],
               "time_s": v4_result["time_s"], "iters": v4_result["iters"], "timing": v4_result["timing"],
               "auc": v4_result["auc"],
               "convergence_iter": v4_result["convergence_iter"]},
        "baseline": {"status": baseline_result["status"], "error": baseline_result["error"], "best_fit": baseline_result["best_fit"], "time_s": baseline_result["time_s"], "iters": baseline_result["iters"]},
        "winner": winner,
        "selected_grid_metric": cfg.grid_metric if cfg.use_grid_search else None,
        "history_v0": v0_result["history"],
        "history_v1": v1_result["history"],
        "history_v2": v2_result["history"],
        "history_v3": v3_result["history"],
        "history_v4": v4_result["history"],
        "iteration_records_v0": v0_result["iteration_records"],
        "iteration_records_v1": v1_result["iteration_records"],
        "iteration_records_v2": v2_result["iteration_records"],
        "iteration_records_v3": v3_result["iteration_records"],
        "iteration_records_v4": v4_result["iteration_records"],
        "plot_path": plot_path,
    }
