"""experiments.run_single

Orchestrates a single experiment run:
- lightweight grid search to pick hyperparameters
- run our sequential PSO (V0)
- run a PySwarm baseline
- save plots + results to disk
- logging

This module is intentionally small so it can be reused by multiple scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from prettytable import PrettyTable

from core.swarm import Swarm
from core.pso import PSO
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology

from parallel.evaluator import SequentialEvaluator

from experiment.grid_search import simple_grid_search
from baseline.pswarm import run_pyswarm_baseline
from utils.logger import setup_logger

from utils.io import ExperimentSummary, save_history_csv, save_summary_json
from viz.convergence import save_convergence_plot


@dataclass
class RunConfig:
    seed: int = 42
    dim: int = 2
    bounds: Tuple[List[float], List[float]] = ([-5] * 2, [5] * 2)

    max_iters: int = 1000
    tol: float = 1e-10
    patience: int = 80

    # Output paths
    out_dir: str = "results"
    plots_dir: str = "logs/convergence"
    log_dir: str = "logs"
    log_file: str = "pso_summary.log"

    save_files: bool = True


def _print_result_table(
    name: str,
    w: float,
    c1: float,
    c2: float,
    n_particles: int,
    best_fit: float,
    iters: int,
    elapsed: float,
    base_fit: float,
    base_iters: int,
    base_time: float,
    winner: str,
) -> None:
    """
    Print a clean terminal summary using PrettyTable.
    """
   
    hyper_table = PrettyTable()
    hyper_table.field_names = ["Hyperparameter", "Value"]
    hyper_table.add_row(["w", f"{w:.3f}"])
    hyper_table.add_row(["c1", f"{c1:.3f}"])
    hyper_table.add_row(["c2", f"{c2:.3f}"])
    hyper_table.add_row(["n_particles", n_particles])

    result_table = PrettyTable()
    result_table.field_names = ["Method", "Best fitness", "Iterations", "Time (s)"]
    result_table.add_row(["Custom PSO (Sequential)", f"{best_fit:.12g}", iters, f"{elapsed:.4f}"])
    result_table.add_row(["PySwarm", f"{base_fit:.12g}", base_iters, f"{base_time:.4f}"])

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    summary_table.add_row(["Winner", winner])

    print(hyper_table)
    print(result_table)
    print(summary_table)


def run_one_objective(
    objective: Callable[[np.ndarray], float],
    cfg: RunConfig,
) -> Dict[str, object]:
    """
    Run a full experiment for a single objective.

    The workflow is:
    1. Run a small grid search to select hyperparameters.
    2. Run the custom sequential PSO.
    3. Run the PySwarm baseline.
    4. Save logs, plots, and optional result files.
    """

    logger = setup_logger(log_dir=cfg.log_dir, log_file=cfg.log_file)
    os.makedirs(cfg.plots_dir, exist_ok=True)

    name = getattr(objective, "__name__", "objective")

    # 1) Grid search
    best_hyperparams = simple_grid_search(
        objective_fn=objective,
        dim=cfg.dim,
        bounds=cfg.bounds,
        max_iters=cfg.max_iters,
        seed=cfg.seed,
    )

    w = float(best_hyperparams["w"])
    c1 = float(best_hyperparams["c1"])
    c2 = float(best_hyperparams["c2"])
    n_particles = int(best_hyperparams["n_particles"])

    # 2) Custom PSO (sequential V0)
    rng = np.random.default_rng(cfg.seed)

    swarm = Swarm(
        n_particles=n_particles,
        dim=cfg.dim,
        bounds=cfg.bounds,
        rng=rng,
    )

    evaluator = SequentialEvaluator(objective)
    bounds_handler = ClampBounds(cfg.bounds[0], cfg.bounds[1])
    topology = GlobalBestTopology()

    pso = PSO(
        swarm=swarm,
        evaluator=evaluator,
        bounds_handler=bounds_handler,
        topology=topology,
        w=w,
        c1=c1,
        c2=c2,
        max_iters=cfg.max_iters,
        tol=cfg.tol,
        patience=cfg.patience,
    )

    best_pos, best_fit, elapsed, iters = pso.run()
    history = list(pso.history)

    # 3) PySwarm baseline
    _, base_fit, base_time, base_iters = run_pyswarm_baseline(
        objective_function=objective,
        bounds=cfg.bounds,
        n_particles=n_particles,
        max_iters=cfg.max_iters,
    )

    # 4) Winner
    if best_fit < base_fit:
        winner = "Custom PSO (Sequential)"
    elif base_fit < best_fit:
        winner = "PySwarm"
    else:
        winner = "Tie"

    # 5) Convergence plot
    plot_path = os.path.join(cfg.plots_dir, f"{name}_convergence.png")
    save_convergence_plot(
        history=history,
        baseline_final_fitness=float(base_fit),
        title=f"Convergence - {name}",
        out_path=plot_path,
    )

    # 6) File logging only
    logger.info(f"---{name.upper()}---")
    logger.info("[CUSTOM PSO (SEQUENTIAL)]")
    logger.info(f"Hyperparameters: w={w}, c1={c1}, c2={c2}, n_particles={n_particles}")
    logger.info(f"Best fitness: {best_fit}")
    logger.info(f"Iterations: {iters}")
    logger.info(f"Time elapsed: {elapsed:.4f} s")

    logger.info("[PySwarm Baseline]")
    logger.info(f"Best fitness: {base_fit}")
    logger.info(f"Iterations: {base_iters}")
    logger.info(f"Time elapsed: {base_time:.4f} s")

    logger.info(f"Fitness difference: {best_fit - base_fit}")
    logger.info(f"Winner: {winner}")

    # 7) Pretty console output
    _print_result_table(
        name=name,
        w=w,
        c1=c1,
        c2=c2,
        n_particles=n_particles,
        best_fit=float(best_fit),
        iters=int(iters),
        elapsed=float(elapsed),
        base_fit=float(base_fit),
        base_iters=int(base_iters),
        base_time=float(base_time),
        winner=winner,
    )

    # 8) Save summary and history
    if cfg.save_files:
        obj_dir = os.path.join(cfg.out_dir, name)
        os.makedirs(obj_dir, exist_ok=True)

        summary = ExperimentSummary(
            objective=name,
            dim=cfg.dim,
            bounds_lower=list(cfg.bounds[0]),
            bounds_upper=list(cfg.bounds[1]),
            seed=cfg.seed,
            w=w,
            c1=c1,
            c2=c2,
            n_particles=n_particles,
            max_iters=cfg.max_iters,
            tol=cfg.tol,
            patience=cfg.patience,
            custom_best_fitness=float(best_fit),
            custom_iterations=int(iters),
            custom_time_s=float(elapsed),
            baseline_best_fitness=float(base_fit),
            baseline_iterations=int(base_iters),
            baseline_time_s=float(base_time),
            winner=winner,
        )

        save_summary_json(summary, os.path.join(obj_dir, "summary.json"))
        save_history_csv(history, os.path.join(obj_dir, "history.csv"))

    # 9) Return result for scripts / notebooks
    return {
        "objective": name,
        "hyperparams": {"w": w, "c1": c1, "c2": c2, "n_particles": n_particles},
        "custom": {
            "best_pos": best_pos,
            "best_fit": best_fit,
            "time_s": elapsed,
            "iters": iters,
        },
        "baseline": {
            "best_fit": base_fit,
            "time_s": base_time,
            "iters": base_iters,
        },
        "winner": winner,
        "history": history,
        "plot_path": plot_path,
    }
