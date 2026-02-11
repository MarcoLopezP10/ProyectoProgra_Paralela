import numpy as np
from core.swarm import Swarm
from core.pso import PSO
from core.evaluator import SequentialEvaluator
from core.bounds import ClampBounds
from objectives.sphere import sphere
from objectives.ackely import ackley
from objectives.rosenbrock import rosenbrock
from objectives.rastringin import rastrigin

def main():
    seed = 42
    rng = np.random.default_rng(seed)

    dim = 2
    bounds = ([-5]*dim, [5]*dim)

    swarm = Swarm(
        n_particles=3000,
        dim=dim,
        bounds=bounds,
        rng=rng
    )

    
    bounds_handler = ClampBounds(bounds[0], bounds[1])

    for objective in [sphere, ackley, rosenbrock, rastrigin]:
        evaluator = SequentialEvaluator(objective)
    
        pso = PSO(
        swarm=swarm,
        evaluator=evaluator,
        bounds_handler=bounds_handler,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iters=100,
        rng=rng
        )

        best_pos, best_fit, elapsed = pso.run()

        print("\nObjective:", objective.__name__)
        print("Best position:", best_pos)
        print("Best fitness:", best_fit)
        print("Elapsed time:", elapsed)

if __name__ == "__main__":
    main()
