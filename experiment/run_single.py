"""experiment.run_single

Orchestrates a single PSO experiment:
  1. Optional grid search for hyperparameters.
  2. Run V0, V1, V2 and V3 with the same config and seed.
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
from parallel.evaluator import build_evaluator
from experiment.grid_search import recommended_profile, simple_grid_search
from baseline.pswarm import run_pyswarm_baseline
from utils.logger import setup_logger
from utils.metadata import collect_execution_metadata
from utils.methods import PSO_METHOD_SPECS, STRATEGY_LABELS
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


def _build_method_result(
    strategy: str,
    pso: Optional[PSO],
    best_fitness: float,
    iterations: int,
    total_s: Optional[float] = None,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> MethodResult:
    """Create a persistable method result from a PSO run or a baseline value."""
    if pso is None:
        timing = TimingBreakdown(total_s=float(total_s or 0.0))
        return MethodResult(
            strategy=strategy,
            best_fitness=float(best_fitness),
            iterations=int(iterations),
            auc=None,
            convergence_iteration=None,
            timing=timing,
            max_workers=max_workers,
            batch_size=batch_size,
        )

    return MethodResult(
        strategy=strategy,
        best_fitness=float(best_fitness),
        iterations=int(iterations),
        auc=float(pso.area_under_curve()),
        convergence_iteration=int(pso.convergence_iteration()),
        timing=TimingBreakdown(**pso.timing_summary()),
        max_workers=max_workers,
        batch_size=batch_size,
    )


def _method_runtime_config(
    method_key: str,
    cfg: RunConfig,
) -> tuple[Optional[int], Optional[int]]:
    """Return per-method runtime knobs while keeping the PSO loop generic."""
    if method_key == "v1":
        return cfg.thread_max_workers, None
    if method_key == "v2":
        return cfg.process_max_workers, cfg.batch_size
    return None, None


def _print_table(
    name: str,
    dim: int,
    seed: int,
    w: float,
    c1: float,
    c2: float,
    n_particles: int,
    method_rows: List[Dict[str, object]],
    winner: str,
    thread_max_workers: Optional[int],
    process_max_workers: Optional[int],
    batch_size: Optional[int],
) -> None:
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
            str(row["label"]),
            f"{float(row['best_fit']):.6e}",
            int(row["iters"]),
            row["conv_iter"],
            f"{float(row['time_s']):.4f}",
            str(row["speedup"]),
            str(row["pct_eval"]),
            str(row["pct_update"]),
        ]
        for row in method_rows
    ]
    summary_rows = [["Winner", winner]]
    for row in method_rows:
        if row["key"] == "v0":
            continue
        summary_rows.append([f"{row['label']} speedup vs V0", str(row["speedup"])])

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


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_one_objective(
    objective: Callable[[np.ndarray], float],
    cfg: RunConfig,
) -> Dict[str, object]:
    """
    Run a full baseline+V0+V1+V2+V3 experiment for a single objective function.

    The seed is passed through to every PSO instance so that V0, V1, V2, and V3
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

    method_results: Dict[str, Dict[str, object]] = {}
    pso_runs: Dict[str, PSO] = {}
    histories: Dict[str, List[float]] = {}

    for spec in PSO_METHOD_SPECS:
        max_workers, batch_size = _method_runtime_config(spec.key, cfg)
        method_log = logging.LoggerAdapter(
            logger,
            {"objective": tag, "method": spec.key.upper()},
        )
        pso = _build_pso(
            cfg,
            bounds,
            w,
            c1,
            c2,
            n_particles,
            build_evaluator(
                spec.key,
                objective,
                max_workers=max_workers,
                batch_size=batch_size,
            ),
            method_log,
        )
        best_pos, best_fit, time_s, iters = pso.run()
        method_results[spec.key] = {
            "label": spec.label,
            "best_pos": best_pos,
            "best_fit": float(best_fit),
            "time_s": float(time_s),
            "iters": int(iters),
            "timing": pso.timing_summary(),
            "max_workers": max_workers,
            "batch_size": batch_size,
            "auc": pso.area_under_curve(),
            "convergence_iter": pso.convergence_iteration(),
        }
        pso_runs[spec.key] = pso
        histories[spec.key] = list(pso.history)

    # ── 3. PySwarm baseline ───────────────────────────────────────────
    _, base_fit, base_time, base_iters = run_pyswarm_baseline(
        objective_function=objective,
        bounds=bounds,
        w=w, c1=c1, c2=c2,
        n_particles=n_particles,
        max_iters=cfg.max_iters,
        seed=cfg.seed,
    )

    # ── 6. Winner ─────────────────────────────────────────────────────
    winner_candidates = [
        (
            str(payload["label"]),
            float(payload["best_fit"]),
            float(payload["time_s"]),
        )
        for payload in method_results.values()
    ]
    winner_candidates.append(("PySwarm baseline", float(base_fit), float(base_time)))
    winner = _resolve_winner(winner_candidates)

    # ── 5. Log summary ────────────────────────────────────────────────
    v0_time = float(method_results["v0"]["time_s"])
    for spec in PSO_METHOD_SPECS:
        payload = method_results[spec.key]
        timing = payload["timing"]
        speedup = v0_time / float(payload["time_s"]) if float(payload["time_s"]) > 0 else float("inf")
        extra = ""
        if spec.key == "v2":
            extra = (
                f" speedup_vs_v0={speedup:.3f}x "
                f"process_workers={cfg.process_max_workers or 'default'} "
                f"batch_size={cfg.batch_size or 'auto'}"
            )
        elif spec.key != "v0":
            extra = f" speedup_vs_v0={speedup:.3f}x"
        log.info(
            f"event=summary method={spec.key.upper()} fit={float(payload['best_fit']):.6e} "
            f"iters={int(payload['iters'])} time_s={float(payload['time_s']):.4f} "
            f"eval_pct={timing['pct_eval']:.1f} update_pct={timing['pct_update']:.1f} "
            f"auc={float(payload['auc']):.6e} conv_iter={int(payload['convergence_iter'])}{extra}"
        )
    log.info(
        f"event=summary method=PySwarm fit={base_fit:.6e} iters={base_iters} time_s={base_time:.4f}"
    )
    log.info(f"event=summary winner={winner}")

    # ── 6. Console table ──────────────────────────────────────────────
    table_rows: List[Dict[str, object]] = []
    for spec in PSO_METHOD_SPECS:
        payload = method_results[spec.key]
        speedup = v0_time / float(payload["time_s"]) if float(payload["time_s"]) > 0 else float("inf")
        timing = payload["timing"]
        table_rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "best_fit": payload["best_fit"],
                "iters": payload["iters"],
                "conv_iter": payload["convergence_iter"],
                "time_s": payload["time_s"],
                "speedup": f"{speedup:.3f}x",
                "pct_eval": f"{timing['pct_eval']:.1f}%",
                "pct_update": f"{timing['pct_update']:.1f}%",
            }
        )
    baseline_speedup = v0_time / base_time if base_time > 0 else float("inf")
    table_rows.append(
        {
            "key": "baseline",
            "label": "PySwarm baseline",
            "best_fit": float(base_fit),
            "iters": int(base_iters),
            "conv_iter": "-",
            "time_s": float(base_time),
            "speedup": f"{baseline_speedup:.3f}x",
            "pct_eval": "-",
            "pct_update": "-",
        }
    )
    _print_table(
        name=name,
        dim=cfg.dim,
        seed=cfg.seed,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        method_rows=table_rows,
        winner=winner,
        thread_max_workers=cfg.thread_max_workers,
        process_max_workers=cfg.process_max_workers,
        batch_size=cfg.batch_size,
    )

    # ── 7. Save results ───────────────────────────────────────────────
    if cfg.save_files:
        # Convergence plot
        plot_path = os.path.join(cfg.plots_dir, f"{name}_d{cfg.dim}_s{cfg.seed}_convergence.png")
        save_convergence_plot(
            history=histories["v0"],
            baseline_final_fitness=float(base_fit),
            title=f"Convergence — {name} d={cfg.dim} seed={cfg.seed}",
            out_path=plot_path,
            threaded_history=histories.get("v1"),
            process_history=histories.get("v2"),
            asyncio_history=histories.get("v3"),
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
                STRATEGY_LABELS["v0"],
                pso_runs["v0"],
                float(method_results["v0"]["best_fit"]),
                int(method_results["v0"]["iters"]),
            ),
            v1=_build_method_result(
                STRATEGY_LABELS["v1"],
                pso_runs["v1"],
                float(method_results["v1"]["best_fit"]),
                int(method_results["v1"]["iters"]),
                max_workers=cfg.thread_max_workers,
            ),
            v2=_build_method_result(
                STRATEGY_LABELS["v2"],
                pso_runs["v2"],
                float(method_results["v2"]["best_fit"]),
                int(method_results["v2"]["iters"]),
                max_workers=cfg.process_max_workers,
                batch_size=cfg.batch_size,
            ),
            v3=_build_method_result(
                STRATEGY_LABELS["v3"],
                pso_runs["v3"],
                float(method_results["v3"]["best_fit"]),
                int(method_results["v3"]["iters"]),
            ),
            baseline=_build_method_result(
                "PySwarm baseline",
                None,
                float(base_fit),
                base_iters,
                total_s=base_time,
            ),
            winner=winner,
            selected_grid_metric=cfg.grid_metric if cfg.use_grid_search else None,
            execution=execution_metadata,
            notes=(
                "V1 uses ThreadPoolExecutor and is mainly a concurrency baseline. "
                "V2 uses ProcessPoolExecutor with batched particle evaluation to "
                "reduce pickling and IPC overhead while preserving the same PSO "
                "search behaviour as V0. V3 uses asyncio.gather and is mainly "
                "expected to help when the objective exposes cooperative latency."
            ),
        )
        summary_path = os.path.join(rdir, "summary.json")

        save_summary_json(summary, summary_path)
        for method_key, pso in pso_runs.items():
            save_iteration_metrics_csv(
                pso.iteration_records,
                os.path.join(rdir, f"history_{method_key}.csv"),
            )

    else:
        plot_path = None

    return {
        "objective": name,
        "dim": cfg.dim,
        "seed": cfg.seed,
        "hyperparams": {"w": w, "c1": c1, "c2": c2, "n_particles": n_particles},
        "v0": method_results["v0"],
        "v1": method_results["v1"],
        "v2": method_results["v2"],
        "v3": method_results["v3"],
        "baseline": {"best_fit": base_fit, "time_s": base_time, "iters": base_iters},
        "winner": winner,
        "selected_grid_metric": cfg.grid_metric if cfg.use_grid_search else None,
        "history_v0": histories["v0"],
        "history_v1": histories["v1"],
        "history_v2": histories["v2"],
        "history_v3": histories["v3"],
        "iteration_records_v0": list(pso_runs["v0"].iteration_records),
        "iteration_records_v1": list(pso_runs["v1"].iteration_records),
        "iteration_records_v2": list(pso_runs["v2"].iteration_records),
        "iteration_records_v3": list(pso_runs["v3"].iteration_records),
        "plot_path": plot_path,
    }
