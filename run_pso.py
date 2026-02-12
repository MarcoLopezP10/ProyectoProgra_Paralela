import numpy as np

from core.swarm import Swarm
from core.pso import PSO
from core.evaluator import SequentialEvaluator
from core.bounds import ClampBounds
from core.topology import GlobalBestTopology

from objectives.sphere import sphere
from objectives.ackley import ackley
from objectives.rosenbrock import rosenbrock
from objectives.rastrigin import rastrigin


def main():
    seed = 42
    dim = 2
    bounds = ([-5]*dim, [5]*dim)

    objectives = [sphere, ackley, rosenbrock, rastrigin]

    for objective in objectives:

        rng = np.random.default_rng(seed)

        swarm = Swarm(
            n_particles=500,
            dim=dim,
            bounds=bounds,
            rng=rng
        )

        evaluator = SequentialEvaluator(objective)
        bounds_handler = ClampBounds(bounds[0], bounds[1])
        topology = GlobalBestTopology()

        pso = PSO(
            swarm=swarm,
            evaluator=evaluator,
            bounds_handler=bounds_handler,
            topology=topology,
            w=0.7,
            c1=1.5,
            c2=1.5,
            max_iters=500,
            tol=1e-10,
            patience=30
        )

        best_pos, best_fit, elapsed, iters = pso.run()

        print("\nObjective:", objective.__name__)
        print("Best position:", best_pos)
        print("Best fitness:", best_fit)
        print("Iterations:", iters)
        print("Elapsed time:", elapsed)


if __name__ == "__main__":
    main()
