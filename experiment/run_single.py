"""experiment.run_single

Orchestrates a single PSO experiment:
  1. Optional grid search for hyperparameters.
  2. Run V0 (sequential), V1 (threading), and V2 (multiprocessing)
     with the same config and seed.
  3. Run PySwarm baseline for reference.
  4. Save structured results (JSON + CSV) and convergence plot.
  5. Structured logging with per-iteration timing breakdown.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from prettytable import PrettyTable
except ModuleNotFoundError:
    PrettyTable = None

from core.swarm import Swarm
from core.pso import PSO
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from parallel.evaluator import ProcessPoolEvaluator, SequentialEvaluator, ThreadPoolEvaluator
from experiment.grid_search import recommended_profile, simple_grid_search
from baseline.pswarm import run_pyswarm_baseline
from utils.logger import setup_logger
from utils.io import (
    ExperimentSummary,
    MethodResult,
    TimingBreakdown,
    result_dir,
    save_history_csv,
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
    thread_max_workers: Optional[int] = None
    process_max_workers: Optional[int] = None
    batch_size: Optional[int] = None

    max_iters: Optional[int] = None
    tol: Optional[float] = None
    patience: Optional[int] = None
    vmax_ratio: Optional[float] = None
    log_every: int = 10       # log a line every N iters inside PSO (0 = off)

    # Output paths
    out_dir: str = "results"
    plots_dir: str = "logs/convergence"
    log_dir: str = "logs"
    log_file: str = "pso_summary.log"

    save_files: bool = True

    def bounds_for_dim(self) -> Tuple[List[float], List[float]]:
        """Return bounds scaled to cfg.dim (replicating the first element)."""
        lo = float(self.bounds[0][0])
        hi = float(self.bounds[1][0])
        return ([lo] * self.dim, [hi] * self.dim)


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


def _resolve_winner(results: list[tuple[str, float, float]]) -> str:
    """Best fitness first; time as tie-break."""
    best_fitness = min(f for _, f, _ in results)
    candidates = [(n, t) for n, f, t in results if np.isclose(f, best_fitness, atol=1e-15)]
    if len(candidates) == 1:
        return candidates[0][0]
    best_time = min(t for _, t in candidates)
    winners = [n for n, t in candidates if np.isclose(t, best_time, atol=1e-12)]
    return winners[0] if len(winners) == 1 else "Tie"


def _print_table(
    name: str, dim: int, seed: int,
    w: float, c1: float, c2: float, n_particles: int,
    v0_fit: float, v0_iters: int, v0_time: float,
    v0_pct_eval: float, v0_pct_update: float,
    v1_fit: float, v1_iters: int, v1_time: float,
    v1_pct_eval: float, v1_pct_update: float,
    v2_fit: float, v2_iters: int, v2_time: float,
    v2_pct_eval: float, v2_pct_update: float,
    base_fit: float, base_iters: int, base_time: float,
    winner: str,
    thread_max_workers: Optional[int],
    process_max_workers: Optional[int],
    batch_size: Optional[int],
) -> None:
    v1_speedup = v0_time / v1_time if v1_time > 0 else float("inf")
    v2_speedup = v0_time / v2_time if v2_time > 0 else float("inf")
    baseline_speedup = v0_time / base_time if base_time > 0 else float("inf")

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
        ["V0 Sequential", f"{v0_fit:.6e}", v0_iters, f"{v0_time:.4f}",
         "1.000x", f"{v0_pct_eval:.1f}%", f"{v0_pct_update:.1f}%"],
        ["V1 Threading", f"{v1_fit:.6e}", v1_iters, f"{v1_time:.4f}",
         f"{v1_speedup:.3f}x", f"{v1_pct_eval:.1f}%", f"{v1_pct_update:.1f}%"],
        ["V2 Multiprocessing", f"{v2_fit:.6e}", v2_iters, f"{v2_time:.4f}",
         f"{v2_speedup:.3f}x", f"{v2_pct_eval:.1f}%", f"{v2_pct_update:.1f}%"],
        ["PySwarm baseline", f"{base_fit:.6e}", base_iters, f"{base_time:.4f}",
         f"{baseline_speedup:.3f}x", "-", "-"],
    ]
    summary_rows = [
        ["Winner", winner],
        ["V1 speedup vs V0", f"{v1_speedup:.3f}x"],
        ["V2 speedup vs V0", f"{v2_speedup:.3f}x"],
    ]

    if PrettyTable is not None:
        t = PrettyTable(["Hyperparameter", "Value"])
        for r in hyper_rows: t.add_row(r)
        print(t)

        t = PrettyTable(["Method", "Best fitness", "Iters", "Time (s)", "Speedup", "% eval", "% update"])
        for r in result_rows: t.add_row(r)
        print(t)

        t = PrettyTable(["Metric", "Value"])
        for r in summary_rows: t.add_row(r)
        print(t)
    else:
        for section, headers, rows in [
            ("Hyperparameters", ["Param", "Value"], hyper_rows),
            ("Results", ["Method", "Best fitness", "Iters", "Time (s)", "Speedup", "% eval", "% update"], result_rows),
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


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_one_objective(
    objective: Callable[[np.ndarray], float],
    cfg: RunConfig,
) -> Dict[str, object]:
    """
    Run a full V0+V1+V2 experiment for a single objective function.

    The seed is passed through to every PSO instance so that V0, V1, and V2
    start from exactly the same swarm state and results are comparable.

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
        f"max_iters={cfg.max_iters} patience={cfg.patience} tol={cfg.tol:.1e}"
    )

    # ── 2. V0 — Sequential ───────────────────────────────────────────
    v0_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V0"})
    v0_pso = _build_pso(cfg, bounds, w, c1, c2, n_particles,
                        SequentialEvaluator(objective), v0_log)
    v0_pos, v0_fit, v0_time, v0_iters = v0_pso.run()
    v0_timing = v0_pso.timing_summary()
    v0_history = list(v0_pso.history)

    # ── 3. V1 — Threading ────────────────────────────────────────────
    v1_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V1"})
    v1_pso = _build_pso(cfg, bounds, w, c1, c2, n_particles,
                        ThreadPoolEvaluator(objective, max_workers=cfg.thread_max_workers),
                        v1_log)
    v1_pos, v1_fit, v1_time, v1_iters = v1_pso.run()
    v1_timing = v1_pso.timing_summary()
    v1_history = list(v1_pso.history)

    # ── 4. V2 — Multiprocessing ───────────────────────────────────────
    v2_log = logging.LoggerAdapter(logger, {"objective": tag, "method": "V2"})
    v2_pso = _build_pso(
        cfg,
        bounds,
        w,
        c1,
        c2,
        n_particles,
        ProcessPoolEvaluator(
            objective,
            max_workers=cfg.process_max_workers,
            batch_size=cfg.batch_size,
        ),
        v2_log,
    )
    v2_pos, v2_fit, v2_time, v2_iters = v2_pso.run()
    v2_timing = v2_pso.timing_summary()
    v2_history = list(v2_pso.history)

    # ── 5. PySwarm baseline ───────────────────────────────────────────
    _, base_fit, base_time, base_iters = run_pyswarm_baseline(
        objective_function=objective,
        bounds=bounds,
        w=w, c1=c1, c2=c2,
        n_particles=n_particles,
        max_iters=cfg.max_iters,
    )

    # ── 6. Winner ─────────────────────────────────────────────────────
    winner = _resolve_winner([
        ("V0 Sequential", float(v0_fit), float(v0_time)),
        ("V1 Threading", float(v1_fit), float(v1_time)),
        ("V2 Multiprocessing", float(v2_fit), float(v2_time)),
        ("PySwarm", float(base_fit), float(base_time)),
    ])

    # ── 7. Log summary ────────────────────────────────────────────────
    log.info(
        f"event=summary method=V0 fit={v0_fit:.6e} iters={v0_iters} "
        f"time_s={v0_time:.4f} eval_pct={v0_timing['pct_eval']:.1f} "
        f"update_pct={v0_timing['pct_update']:.1f}"
    )
    log.info(
        f"event=summary method=V1 fit={v1_fit:.6e} iters={v1_iters} "
        f"time_s={v1_time:.4f} eval_pct={v1_timing['pct_eval']:.1f} "
        f"update_pct={v1_timing['pct_update']:.1f} speedup_vs_v0={v0_time/v1_time:.3f}x"
    )
    log.info(
        f"event=summary method=V2 fit={v2_fit:.6e} iters={v2_iters} "
        f"time_s={v2_time:.4f} eval_pct={v2_timing['pct_eval']:.1f} "
        f"update_pct={v2_timing['pct_update']:.1f} speedup_vs_v0={v0_time/v2_time:.3f}x "
        f"process_workers={cfg.process_max_workers or 'default'} "
        f"batch_size={cfg.batch_size or 'auto'}"
    )
    log.info(
        f"event=summary method=PySwarm fit={base_fit:.6e} iters={base_iters} time_s={base_time:.4f}"
    )
    log.info(f"event=summary winner={winner}")

    # ── 8. Console table ──────────────────────────────────────────────
    _print_table(
        name=name, dim=cfg.dim, seed=cfg.seed,
        w=w, c1=c1, c2=c2, n_particles=n_particles,
        v0_fit=float(v0_fit), v0_iters=v0_iters, v0_time=v0_time,
        v0_pct_eval=v0_timing["pct_eval"], v0_pct_update=v0_timing["pct_update"],
        v1_fit=float(v1_fit), v1_iters=v1_iters, v1_time=v1_time,
        v1_pct_eval=v1_timing["pct_eval"], v1_pct_update=v1_timing["pct_update"],
        v2_fit=float(v2_fit), v2_iters=v2_iters, v2_time=v2_time,
        v2_pct_eval=v2_timing["pct_eval"], v2_pct_update=v2_timing["pct_update"],
        base_fit=float(base_fit), base_iters=base_iters, base_time=base_time,
        winner=winner, thread_max_workers=cfg.thread_max_workers,
        process_max_workers=cfg.process_max_workers, batch_size=cfg.batch_size,
    )

    # ── 9. Save results ───────────────────────────────────────────────
    if cfg.save_files:
        # Convergence plot
        plot_path = os.path.join(cfg.plots_dir, f"{name}_d{cfg.dim}_s{cfg.seed}_convergence.png")
        save_convergence_plot(
            history=v0_history,
            baseline_final_fitness=float(base_fit),
            title=f"Convergence — {name} d={cfg.dim} seed={cfg.seed}",
            out_path=plot_path,
            threaded_history=v1_history,
            process_history=v2_history,
        )

        # Structured results directory: results/sphere_d2_s42/
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
            v0=MethodResult(
                best_fitness=float(v0_fit),
                iterations=v0_iters,
                timing=TimingBreakdown(**v0_timing),
            ),
            v1=MethodResult(
                best_fitness=float(v1_fit),
                iterations=v1_iters,
                timing=TimingBreakdown(**v1_timing),
                max_workers=cfg.thread_max_workers,
            ),
            v2=MethodResult(
                best_fitness=float(v2_fit),
                iterations=v2_iters,
                timing=TimingBreakdown(**v2_timing),
                max_workers=cfg.process_max_workers,
                batch_size=cfg.batch_size,
            ),
            baseline=MethodResult(
                best_fitness=float(base_fit),
                iterations=base_iters,
                timing=TimingBreakdown(total_s=base_time),
            ),
            winner=winner,
            notes=(
                "V1 uses ThreadPoolExecutor and is mainly a concurrency baseline. "
                "V2 uses ProcessPoolExecutor with batched particle evaluation to "
                "reduce pickling and IPC overhead while preserving the same PSO "
                "search behaviour as V0."
            ),
        )
        summary_path = os.path.join(rdir, "summary.json")
        history_v0_path = os.path.join(rdir, "history_v0.csv")
        history_v1_path = os.path.join(rdir, "history_v1.csv")
        history_v2_path = os.path.join(rdir, "history_v2.csv")

        save_summary_json(summary, summary_path)
        save_history_csv(v0_history, history_v0_path)
        save_history_csv(v1_history, history_v1_path)
        save_history_csv(v2_history, history_v2_path)

    else:
        plot_path = None

    return {
        "objective": name,
        "dim": cfg.dim,
        "seed": cfg.seed,
        "hyperparams": {"w": w, "c1": c1, "c2": c2, "n_particles": n_particles},
        "v0": {"best_pos": v0_pos, "best_fit": v0_fit,
               "time_s": v0_time, "iters": v0_iters, "timing": v0_timing},
        "v1": {"best_pos": v1_pos, "best_fit": v1_fit,
               "time_s": v1_time, "iters": v1_iters, "timing": v1_timing,
               "max_workers": cfg.thread_max_workers},
        "v2": {"best_pos": v2_pos, "best_fit": v2_fit,
               "time_s": v2_time, "iters": v2_iters, "timing": v2_timing,
               "max_workers": cfg.process_max_workers, "batch_size": cfg.batch_size},
        "baseline": {"best_fit": base_fit, "time_s": base_time, "iters": base_iters},
        "winner": winner,
        "history_v0": v0_history,
        "history_v1": v1_history,
        "history_v2": v2_history,
        "plot_path": plot_path,
    }
