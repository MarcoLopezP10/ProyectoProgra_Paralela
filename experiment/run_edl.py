"""experiment.run_edl

Run Economic Load Dispatch variants with the existing PSO implementations.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from prettytable import PrettyTable
except ModuleNotFoundError:
    PrettyTable = None

from core.pso import PSO
from core.swarm import Swarm
from objectives.economic_dispatch import (
    EconomicDispatchCase,
    EconomicDispatchObjective,
    load_case,
)
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from parallel.evaluator import build_evaluator
from baseline.pswarm import run_pyswarm_baseline
from utils.edl_io import (
    EDLMethodResult,
    EDLRunSummary,
    edl_case_dir,
    edl_result_dir,
    save_edl_results_csv,
    save_edl_summary_json,
)
from utils.io import ExecutionMetadata, TimingBreakdown, save_iteration_metrics_csv
from utils.logger import setup_logger
from utils.methods import PSO_METHOD_KEYS, STRATEGY_LABELS
from utils.metadata import collect_execution_metadata

EDL_VARIANTS: Dict[str, Dict[str, object]] = {
    "edl_1": {
        "label": "Base",
        "use_valve_point": False,
        "use_losses": False,
    },
    "edl_2": {
        "label": "Valve-point",
        "use_valve_point": True,
        "use_losses": False,
    },
    "edl_3": {
        "label": "Losses",
        "use_valve_point": False,
        "use_losses": True,
    },
    "edl_4": {
        "label": "Valve-point + Losses",
        "use_valve_point": True,
        "use_losses": True,
    },
}

@dataclass
class EDLRunConfig:
    """Config for one EDL execution suite."""

    case_path: str = "cases/edl_case_3u.json"
    seeds: List[int] = field(default_factory=lambda: [42])
    variants: List[str] = field(default_factory=lambda: list(EDL_VARIANTS))
    w: float = 0.6
    c1: float = 1.4
    c2: float = 1.6
    n_particles: int = 80
    max_iters: int = 500
    tol: float = 1e-8
    patience: int = 80
    vmax_ratio: float = 0.2
    thread_max_workers: Optional[int] = None
    process_max_workers: Optional[int] = None
    batch_size: Optional[int] = None
    penalty_power_balance: float = 1e6
    out_dir: str = "results/edl"
    plots_dir: str = "reports/edl"
    log_dir: str = "logs"
    log_file: str = "edl_summary.log"
    repo_root: str = "."
    save_files: bool = True
    make_plots: bool = True


def _build_pso(
    cfg: EDLRunConfig,
    case: EconomicDispatchCase,
    evaluator,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> PSO:
    bounds = (case.lower_bounds, case.upper_bounds)
    rng = np.random.default_rng(seed)
    swarm = Swarm(
        n_particles=cfg.n_particles,
        dim=case.dim,
        bounds=bounds,
        rng=rng,
        vmax_ratio=cfg.vmax_ratio,
    )
    return PSO(
        swarm=swarm,
        evaluator=evaluator,
        bounds_handler=ClampBounds(bounds[0], bounds[1]),
        topology=GlobalBestTopology(),
        w=cfg.w,
        c1=cfg.c1,
        c2=cfg.c2,
        max_iters=cfg.max_iters,
        seed=seed,
        tol=cfg.tol,
        patience=cfg.patience,
        log_every=0,
        logger=logger,
    )


def _build_method_result(
    method_key: str,
    strategy: Optional[str] = None,
    best_position: Optional[np.ndarray] = None,
    best_fitness: Optional[float] = None,
    iterations: Optional[int] = None,
    components: Optional[dict[str, float]] = None,
    pso: Optional[PSO] = None,
    total_s: Optional[float] = None,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    status: str = "ok",
    error: Optional[str] = None,
) -> EDLMethodResult:
    timing = TimingBreakdown(total_s=float(total_s or 0.0))
    auc = None
    convergence_iteration = None
    if pso is not None:
        timing = TimingBreakdown(**pso.timing_summary())
        auc = float(pso.area_under_curve())
        convergence_iteration = int(pso.convergence_iteration())

    components = components or {}
    return EDLMethodResult(
        method_key=method_key,
        strategy=strategy or STRATEGY_LABELS.get(method_key, "PySwarm baseline"),
        status=status,
        best_fitness=None if best_fitness is None else float(best_fitness),
        fuel_cost=components.get("fuel_cost"),
        transmission_loss=components.get("transmission_loss"),
        power_balance_error=components.get("power_balance_error"),
        power_balance_residual=components.get("power_balance_residual"),
        penalty=components.get("penalty"),
        iterations=None if iterations is None else int(iterations),
        auc=auc,
        convergence_iteration=convergence_iteration,
        timing=timing,
        best_position=[]
        if best_position is None
        else [float(value) for value in np.asarray(best_position, dtype=float).tolist()],
        max_workers=max_workers,
        batch_size=batch_size,
        error=error,
    )


def _resolve_winner(methods: List[EDLMethodResult]) -> str:
    available = [
        method
        for method in methods
        if method.status == "ok" and method.best_fitness is not None
    ]
    if not available:
        return "Unavailable"

    best_fitness = min(float(method.best_fitness) for method in available)
    candidates = [
        method
        for method in available
        if np.isclose(float(method.best_fitness), best_fitness, atol=1e-15)
    ]
    if len(candidates) == 1:
        return candidates[0].strategy

    best_time = min(float(method.timing.total_s) for method in candidates)
    fastest = [
        method
        for method in candidates
        if np.isclose(float(method.timing.total_s), best_time, atol=1e-12)
    ]
    return fastest[0].strategy if len(fastest) == 1 else "Tie"


def _format_value(value: Optional[float], fmt: str = ".6f") -> str:
    if value is None:
        return "-"
    return format(float(value), fmt)


def _format_powers(values: List[float]) -> str:
    if not values:
        return "-"
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _summarize_powers(values: List[float]) -> str:
    """Compact one-line summary for a dispatch vector."""
    if not values:
        return "-"

    powers = np.asarray(values, dtype=float)
    return (
        f"n={len(values)} | "
        f"sum={float(np.sum(powers)):.4f} | "
        f"min={float(np.min(powers)):.4f} | "
        f"max={float(np.max(powers)):.4f}"
    )


def _format_dispatch_lines(
    generator_names: List[str],
    values: List[float],
    entries_per_line: Optional[int] = None,
    max_width: Optional[int] = None,
) -> List[str]:
    """Wrapped generator dispatch lines for wide solutions."""
    if not values:
        return ["-"]

    labels = generator_names if len(generator_names) == len(values) else [
        f"P{i + 1}" for i in range(len(values))
    ]
    entries = [f"{labels[idx]}={values[idx]:.4f}" for idx in range(len(values))]
    if entries_per_line is not None:
        lines = []
        for start in range(0, len(entries), entries_per_line):
            end = min(start + entries_per_line, len(entries))
            lines.append(" | ".join(entries[start:end]))
        return lines

    if max_width is None:
        terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        max_width = max(40, terminal_width - 8)

    lines: List[str] = []
    current = entries[0]
    for entry in entries[1:]:
        candidate = f"{current} | {entry}"
        if len(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = entry
    lines.append(current)
    return lines


def _group_dispatches(methods: List[EDLMethodResult]) -> List[dict]:
    """Group methods that converged to the same dispatch vector."""
    groups: List[dict] = []
    for method in methods:
        if not method.best_position:
            continue
        position = np.asarray(method.best_position, dtype=float)
        matched = False
        for group in groups:
            ref = group["position"]
            if position.shape == ref.shape and np.allclose(position, ref, atol=1e-9, rtol=0.0):
                group["strategies"].append(method.strategy)
                matched = True
                break
        if not matched:
            groups.append(
                {
                    "position": position,
                    "values": list(method.best_position),
                    "strategies": [method.strategy],
                }
            )
    return groups


def _print_table(summary: EDLRunSummary) -> None:
    methods = [
        method
        for method in (summary.v0, summary.v1, summary.v2, summary.v3, summary.baseline)
        if method is not None
    ]
    has_long_dispatch = any(len(method.best_position) > 6 for method in methods if method is not None)

    if has_long_dispatch:
        headers = [
            "Version",
            "Status",
            "Fitness",
            "Fuel cost",
            "Losses",
            "|Balance|",
            "Time (s)",
            "Iters",
        ]
        rows = [
            [
                method.strategy,
                method.status,
                _format_value(method.best_fitness, ".6e"),
                _format_value(method.fuel_cost, ".6f"),
                _format_value(method.transmission_loss, ".6f"),
                _format_value(method.power_balance_error, ".6e"),
                _format_value(method.timing.total_s, ".4f"),
                method.iterations if method.iterations is not None else "-",
            ]
            for method in methods
            if method is not None
        ]
    else:
        headers = [
            "Version",
            "Status",
            "Fitness",
            "Fuel cost",
            "Losses",
            "|Balance|",
            "Time (s)",
            "Iters",
            "Best powers",
        ]
        rows = [
            [
                method.strategy,
                method.status,
                _format_value(method.best_fitness, ".6e"),
                _format_value(method.fuel_cost, ".6f"),
                _format_value(method.transmission_loss, ".6f"),
                _format_value(method.power_balance_error, ".6e"),
                _format_value(method.timing.total_s, ".4f"),
                method.iterations if method.iterations is not None else "-",
                _format_powers(method.best_position),
            ]
            for method in methods
            if method is not None
        ]

    print(f"\n{'=' * 72}")
    print(
        f"EDL | case={summary.case_name} | variant={summary.variant} "
        f"({summary.variant_label}) | seed={summary.seed}"
    )
    print(f"{'=' * 72}")

    if PrettyTable is not None:
        table = PrettyTable(headers)
        for row in rows:
            table.add_row(row)
        table.align = "r"
        table.align["Version"] = "l"
        table.align["Status"] = "c"
        print(table)
    else:
        widths = [max(len(str(item)) for item in [header] + [row[idx] for row in rows]) for idx, header in enumerate(headers)]
        sep = "-+-".join("-" * width for width in widths)

        def fmt_row(row: List[object]) -> str:
            return " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers)))

        print(fmt_row(headers))
        print(sep)
        for row in rows:
            print(fmt_row(row))

    if has_long_dispatch:
        dispatch_groups = _group_dispatches([method for method in methods if method is not None])
        if dispatch_groups:
            print("Dispatch:")
            for group in dispatch_groups:
                label = ", ".join(group["strategies"])
                print(f"  {label}")
                print(f"    {_summarize_powers(group['values'])}")
                for line in _format_dispatch_lines(summary.generator_names, group["values"]):
                    print(f"    {line}")

    print(f"Winner: {summary.winner}")


def _summary_to_rows(summary: EDLRunSummary) -> List[dict]:
    rows = []
    for method in (summary.v0, summary.v1, summary.v2, summary.v3, summary.baseline):
        if method is None:
            continue
        rows.append(
            {
                "case_name": summary.case_name,
                "variant": summary.variant,
                "variant_label": summary.variant_label,
                "seed": summary.seed,
                "version": method.method_key,
                "strategy": method.strategy,
                "status": method.status,
                "fitness": method.best_fitness,
                "fuel_cost": method.fuel_cost,
                "losses": method.transmission_loss,
                "balance_error": method.power_balance_error,
                "balance_residual": method.power_balance_residual,
                "penalty": method.penalty,
                "time_s": method.timing.total_s,
                "iters": method.iterations,
                "auc": method.auc,
                "convergence_iteration": method.convergence_iteration,
                "max_workers": method.max_workers,
                "batch_size": method.batch_size,
                "error": method.error,
                "best_powers": list(method.best_position),
            }
        )
    return rows


def _run_strategy(
    method_key: str,
    objective: EconomicDispatchObjective,
    case: EconomicDispatchCase,
    cfg: EDLRunConfig,
    seed: int,
    logger: logging.Logger,
    objective_tag: str,
) -> Tuple[EDLMethodResult, Optional[PSO]]:
    max_workers = None
    batch_size = None
    if method_key == "v1":
        max_workers = cfg.thread_max_workers
    elif method_key == "v2":
        max_workers = cfg.process_max_workers
        batch_size = cfg.batch_size

    method_logger = logging.LoggerAdapter(
        logger,
        {"objective": objective_tag, "method": method_key.upper()},
    )
    evaluator = build_evaluator(
        method_key,
        objective,
        max_workers=max_workers,
        batch_size=batch_size,
    )
    pso = _build_pso(cfg, case, evaluator, seed, method_logger)

    try:
        best_position, best_fitness, _, iterations = pso.run()
    except Exception as exc:
        if method_key != "v2":
            raise

        error = f"{type(exc).__name__}: {exc}"
        method_logger.warning(f"event=unavailable reason={error}")
        return (
            _build_method_result(
                method_key=method_key,
                status="unavailable",
                error=error,
                max_workers=max_workers,
                batch_size=batch_size,
            ),
            None,
        )

    components = objective.components(best_position)
    method_logger.info(
        "event=summary fit=%s fuel=%s losses=%s balance_error=%s time_s=%.4f",
        f"{best_fitness:.6e}",
        f"{components['fuel_cost']:.6f}",
        f"{components['transmission_loss']:.6f}",
        f"{components['power_balance_error']:.6e}",
        pso.timing_summary()["total_s"],
    )
    return (
        _build_method_result(
            method_key=method_key,
            best_position=best_position,
            best_fitness=best_fitness,
            iterations=iterations,
            components=components,
            pso=pso,
            max_workers=max_workers,
            batch_size=batch_size,
        ),
        pso,
    )


def _run_baseline(
    objective: EconomicDispatchObjective,
    case: EconomicDispatchCase,
    cfg: EDLRunConfig,
    seed: int,
) -> EDLMethodResult:
    best_position, best_fitness, time_s, iters = run_pyswarm_baseline(
        objective_function=objective,
        bounds=(case.lower_bounds, case.upper_bounds),
        w=cfg.w,
        c1=cfg.c1,
        c2=cfg.c2,
        n_particles=cfg.n_particles,
        max_iters=cfg.max_iters,
        seed=seed,
    )
    components = objective.components(best_position)
    return _build_method_result(
        method_key="baseline",
        best_position=best_position,
        best_fitness=best_fitness,
        iterations=iters,
        components=components,
        pso=None,
        total_s=time_s,
        status="ok",
        error=None,
    )


def run_one_edl_variant(
    case: EconomicDispatchCase,
    variant: str,
    cfg: EDLRunConfig,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> EDLRunSummary:
    """Run one EDL variant for one seed across baseline + V0/V1/V2/V3."""
    if variant not in EDL_VARIANTS:
        raise ValueError(
            f"Unknown EDL variant {variant!r}. Expected one of {sorted(EDL_VARIANTS)}."
        )

    variant_cfg = EDL_VARIANTS[variant]
    if bool(variant_cfg["use_losses"]) and case.loss_model is None:
        if logger is None:
            logger = setup_logger(
                name="edl_logger",
                log_dir=cfg.log_dir,
                log_file=cfg.log_file,
            )
        execution_metadata = ExecutionMetadata(**collect_execution_metadata(cfg.repo_root))
        reason = (
            "This case does not define transmission-loss coefficients. "
            "Use edl_1 or edl_2, or provide a case with a loss model."
        )
        unavailable = _build_method_result(
            method_key="v0",
            status="unavailable",
            error=reason,
        )
        unavailable_v1 = _build_method_result(
            method_key="v1",
            status="unavailable",
            error=reason,
            max_workers=cfg.thread_max_workers,
        )
        unavailable_v2 = _build_method_result(
            method_key="v2",
            status="unavailable",
            error=reason,
            max_workers=cfg.process_max_workers,
            batch_size=cfg.batch_size,
        )
        unavailable_v3 = _build_method_result(
            method_key="v3",
            status="unavailable",
            error=reason,
        )
        unavailable_baseline = _build_method_result(
            method_key="baseline",
            strategy="PySwarm baseline",
            status="unavailable",
            error=reason,
        )
        summary = EDLRunSummary(
            case_name=case.case_name,
            variant=variant,
            variant_label=str(variant_cfg["label"]),
            seed=seed,
            demand=float(case.demand),
            generator_names=case.generator_names,
            bounds_lower=case.lower_bounds,
            bounds_upper=case.upper_bounds,
            use_valve_point=bool(variant_cfg["use_valve_point"]),
            use_losses=bool(variant_cfg["use_losses"]),
            penalty_power_balance=float(cfg.penalty_power_balance),
            w=float(cfg.w),
            c1=float(cfg.c1),
            c2=float(cfg.c2),
            n_particles=int(cfg.n_particles),
            max_iters=int(cfg.max_iters),
            tol=float(cfg.tol),
            patience=int(cfg.patience),
            vmax_ratio=float(cfg.vmax_ratio),
            v0=unavailable,
            v1=unavailable_v1,
            v2=unavailable_v2,
            v3=unavailable_v3,
            baseline=unavailable_baseline,
            winner="Unavailable",
            execution=execution_metadata,
            notes=reason,
        )
        _print_table(summary)

        if cfg.save_files:
            run_dir = edl_result_dir(cfg.out_dir, case.case_name, variant, seed)
            os.makedirs(run_dir, exist_ok=True)
            save_edl_summary_json(summary, os.path.join(run_dir, "summary.json"))

        return summary

    objective = EconomicDispatchObjective(
        case=case,
        use_valve_point=bool(variant_cfg["use_valve_point"]),
        use_losses=bool(variant_cfg["use_losses"]),
        penalty_power_balance=cfg.penalty_power_balance,
    )

    if logger is None:
        logger = setup_logger(
            name="edl_logger",
            log_dir=cfg.log_dir,
            log_file=cfg.log_file,
        )

    objective_tag = f"{case.case_name}:{variant}:seed={seed}"
    execution_metadata = ExecutionMetadata(**collect_execution_metadata(cfg.repo_root))
    method_results: Dict[str, EDLMethodResult] = {}
    pso_runs: Dict[str, PSO] = {}

    for method_key in PSO_METHOD_KEYS:
        method_result, pso = _run_strategy(
            method_key=method_key,
            objective=objective,
            case=case,
            cfg=cfg,
            seed=seed,
            logger=logger,
            objective_tag=objective_tag,
        )
        method_results[method_key] = method_result
        if pso is not None:
            pso_runs[method_key] = pso
    method_results["baseline"] = _run_baseline(objective, case, cfg, seed)

    winner = _resolve_winner(list(method_results.values()))
    summary = EDLRunSummary(
        case_name=case.case_name,
        variant=variant,
        variant_label=str(variant_cfg["label"]),
        seed=seed,
        demand=float(case.demand),
        generator_names=case.generator_names,
        bounds_lower=case.lower_bounds,
        bounds_upper=case.upper_bounds,
        use_valve_point=bool(variant_cfg["use_valve_point"]),
        use_losses=bool(variant_cfg["use_losses"]),
        penalty_power_balance=float(cfg.penalty_power_balance),
        w=float(cfg.w),
        c1=float(cfg.c1),
        c2=float(cfg.c2),
        n_particles=int(cfg.n_particles),
        max_iters=int(cfg.max_iters),
        tol=float(cfg.tol),
        patience=int(cfg.patience),
        vmax_ratio=float(cfg.vmax_ratio),
        v0=method_results["v0"],
        v1=method_results["v1"],
        v2=method_results["v2"],
        v3=method_results["v3"],
        baseline=method_results["baseline"],
        winner=winner,
        execution=execution_metadata,
        notes=(
            "EDL run added as an isolated objective family. "
            "V2 is marked unavailable instead of aborting the whole run "
            "when multiprocessing is blocked by the environment. "
            "V3 stays as an additional evaluator option even though EDL itself "
            "is still a synchronous objective."
        ),
    )

    _print_table(summary)

    if cfg.save_files:
        run_dir = edl_result_dir(cfg.out_dir, case.case_name, variant, seed)
        os.makedirs(run_dir, exist_ok=True)
        save_edl_summary_json(summary, os.path.join(run_dir, "summary.json"))
        for method_key, pso in pso_runs.items():
            save_iteration_metrics_csv(
                pso.iteration_records,
                os.path.join(run_dir, f"history_{method_key}.csv"),
            )

    return summary


def run_edl_suite(cfg: EDLRunConfig) -> dict[str, object]:
    """Run the selected EDL case, variants, and seeds."""
    logger = setup_logger(
        name="edl_logger",
        log_dir=cfg.log_dir,
        log_file=cfg.log_file,
    )
    case = load_case(cfg.case_path)
    summaries: List[EDLRunSummary] = []
    rows: List[dict] = []

    for seed in cfg.seeds:
        for variant in cfg.variants:
            summary = run_one_edl_variant(case, variant, cfg, seed, logger)
            summaries.append(summary)
            rows.extend(_summary_to_rows(summary))

    aggregate_csv_path = None
    plots_root = None
    plot_artifacts: List[str] = []
    if cfg.save_files:
        aggregate_csv_path = os.path.join(
            edl_case_dir(cfg.out_dir, case.case_name),
            "comparison.csv",
        )
        save_edl_results_csv(rows, aggregate_csv_path)
        print(f"\nAggregate CSV -> {aggregate_csv_path}")

        if cfg.make_plots:
            from scripts.analyze_edl import analyze_edl_results

            analysis_result = analyze_edl_results(
                results_dir=cfg.out_dir,
                out_dir=cfg.plots_dir,
                case_names=[case.case_name],
                variants=cfg.variants,
                seeds=cfg.seeds,
                quiet=True,
            )
            plot_artifacts = list(analysis_result.get("artifacts") or [])
            plots_root = os.path.join(cfg.plots_dir, case.case_name)
            print(f"Reports -> {plots_root}")
            print(f"Essential figures -> {len(plot_artifacts)}")
    elif cfg.make_plots:
        print("\nReports skipped -> disable --no-save or rerun scripts.analyze_edl on saved results.")

    return {
        "case_name": case.case_name,
        "summaries": summaries,
        "rows": rows,
        "aggregate_csv_path": aggregate_csv_path,
        "plots_root": plots_root,
        "plot_artifacts": plot_artifacts,
    }


__all__ = [
    "EDLRunConfig",
    "EDL_VARIANTS",
    "run_edl_suite",
    "run_one_edl_variant",
]
