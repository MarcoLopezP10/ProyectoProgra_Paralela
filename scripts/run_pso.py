"""scripts.run_pso

Minimal reproducible script (V0):
- Runs the project's sequential PSO vs a PySwarm baseline
- Uses 4 standard objectives (Sphere/Ackley/Rosenbrock/Rastrigin)


"""

from __future__ import annotations

from experiment.run_single import RunConfig, run_one_objective
from objectives.sphere import sphere
from objectives.ackley import ackley
from objectives.rosenbrock import rosenbrock
from objectives.rastrigin import rastrigin


def main() -> None:
    # Minimal test configuration (V0)
    cfg = RunConfig( #en vez de a mano con unn producto cartesiano
        seed=42,
        dim=2,
        bounds=([-5]*2, [5]*2),
        max_iters=500,
        tol=1e-10,
        patience=30,
        save_files=True,
    )

    objectives = [sphere, ackley, rosenbrock, rastrigin]

    for obj in objectives:
        print(f"\n---{obj.__name__.upper()}---") #loggin.info
        run_one_objective(obj, cfg)
        


if __name__ == "__main__":
    main()
