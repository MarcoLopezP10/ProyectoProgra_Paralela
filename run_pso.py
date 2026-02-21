import numpy as np
import matplotlib.pyplot as plt
import os

from core.swarm import Swarm
from core.pso import PSO
from options.evaluator import SequentialEvaluator
from options.bounds import ClampBounds
from options.topology import GlobalBestTopology
from objectives.sphere import sphere
from objectives.ackley import ackley
from objectives.rosenbrock import rosenbrock
from objectives.rastrigin import rastrigin
from experiments.grid_search import simple_grid_search
from utils.logger import setup_logger
from baseline.pswarm import run_pyswarm_baseline


def main():

    seed = 42
    dim = 2
    bounds = ([-5] * dim, [5] * dim)

    logger = setup_logger(log_dir="logs", log_file="pso_summary.log")

    objectives = [sphere, ackley, rosenbrock, rastrigin]

    # Crear carpeta de convergencia
    os.makedirs("logs/convergence", exist_ok=True)

    for objective in objectives:

        print(f"\n---{objective.__name__.upper()}---")

        # 🔹 Grid search
        best_hyperparams = simple_grid_search(objective, dim, bounds)

        w = best_hyperparams["w"]
        c1 = best_hyperparams["c1"]
        c2 = best_hyperparams["c2"]
        n_particles = best_hyperparams["n_particles"]

        rng = np.random.default_rng(seed)

        swarm = Swarm(
            n_particles=n_particles,
            dim=dim,
            bounds=bounds,
            rng=rng,
        )

        evaluator = SequentialEvaluator(objective)
        bounds_handler = ClampBounds(bounds[0], bounds[1])
        topology = GlobalBestTopology()

        pso = PSO(
            swarm,
            evaluator,
            bounds_handler,
            topology,
            w,
            c1,
            c2,
            max_iters=500,
            tol=1e-10,
            patience=30,
        )

        # =============================
        # CUSTOM PSO
        # =============================
        best_pos, best_fit, elapsed, iters = pso.run()
        history = pso.history

        print("\n[Custom PSO]")
        print("Hyperparameters:", f"w={w}, c1={c1}, c2={c2}, n_particles={n_particles}")
        print("Best position:", best_pos)
        print("Best fitness:", best_fit)
        print("Iterations:", iters)
        print("Time elapsed:", f"{elapsed:.4f} s")

        # =============================
        # PYSWARM BASELINE
        # =============================
        base_pos, base_fit, base_time, base_iters = run_pyswarm_baseline(
            objective_function=objective,
            bounds=bounds,
            n_particles=n_particles,
            max_iters=500,
        )

               # Determinar ganador
        if best_fit < base_fit:
            winner = "Custom PSO"
        elif base_fit < best_fit:
            winner = "PySwarm"
        else:
            winner = "Tie"

        print("\n" + "="*75)
        print(f"{objective.__name__.upper():^75}")
        print("="*75)

        print(f"{'Method':<20}{'Best Fitness':<18}{'Time (s)':<12}{'Iterations':<12}")
        print("-"*75)

        print(f"{'Custom PSO':<20}{best_fit:<18.6e}{elapsed:<12.4f}{iters:<12}")
        print(f"{'PySwarm':<20}{base_fit:<18.6e}{base_time:<12.4f}{base_iters:<12}")

        print("-"*75)
        print(f"{'Fitness Difference':<20}{(best_fit - base_fit):.6e}")
        print(f"{'Mejor resultado':<20}{winner}")
        print("="*75 + "\n")

        
        # =============================
        # 🔥 IMPROVED CONVERGENCE PLOT
        # =============================

        plt.figure(figsize=(8, 5))

        # Custom PSO curve
        plt.plot(
            history,
            linewidth=2.5,
            label="Custom PSO",
        )

        # PySwarm horizontal line
        plt.axhline(
            y=base_fit,
            linestyle="--",
            linewidth=2.5,
            label="PySwarm Final Fitness",
        )

        plt.xlabel("Iteration", fontsize=11)
        plt.ylabel("Best Fitness", fontsize=11)
        plt.title(f"Convergence - {objective.__name__}", fontsize=12)

        plt.grid(True, linestyle="--", alpha=0.6)

        # 🔥 Solo usar log si no hay ceros
        if min(history) > 0 and base_fit > 0:
            plt.yscale("log")

        plt.text(
            0.6 * len(history),
            max(history),
            f"Δ fitness = {best_fit - base_fit:.2e}",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8)
        )
        
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"logs/convergence/{objective.__name__}_convergence.png", dpi=300)
        plt.close()

        # =============================
        # LOGGING
        # =============================
        logger.info(f"---{objective.__name__.upper()}---")

        logger.info("[Custom PSO]")
        logger.info(f"Best fitness: {best_fit}")
        logger.info(f"Iterations: {iters}")
        logger.info(f"Time elapsed: {elapsed:.4f} s")

        logger.info("[PySwarm Baseline]")
        logger.info(f"Best fitness: {base_fit}")
        logger.info(f"Time elapsed: {base_time:.4f} s")

        logger.info(f"Fitness difference: {best_fit - base_fit}")


if __name__ == "__main__":
    main()
