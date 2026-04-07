# Final Report — PSO V2

## 1. Objective and Scope

This project implements a maintainable Particle Swarm Optimization (PSO) codebase in Python and uses it as a controlled testbed to compare execution strategies up to `V2`:

- `V0`: sequential evaluation
- `V1`: concurrent fitness evaluation with `ThreadPoolExecutor`
- `V2`: parallel fitness evaluation with `ProcessPoolExecutor` and batching

The delivery scope ends intentionally at `V2`. `V3` (`asyncio`) and `V4` (vectorized NumPy) are discussed as future work, but they are not part of the implemented submission.

The design goal was not just to make PSO work, but to make it reproducible, measurable, extensible, and fair to compare across strategies. For that reason, the PSO loop itself remains unchanged and only the evaluator strategy is swapped.

## 2. Methodology

### 2.1 Common experimental protocol

All benchmark runs follow the same structure:

1. Select one objective function and one dimension.
2. Build the same swarm with the same seed.
3. Run `V0`, `V1`, and `V2` with identical hyperparameters, bounds, and stopping criteria.
4. Run a PySwarm baseline as an external reference.
5. Persist summaries, per-iteration metrics, and convergence histories to disk.
6. Generate aggregate reports and plots from the saved results.

This keeps the comparison fair: any difference between `V0`, `V1`, and `V2` should come from the evaluation strategy rather than from hidden algorithmic changes.

### 2.2 Benchmarks

The benchmark suite includes the four required standard continuous optimization functions:

- Sphere
- Ackley
- Rosenbrock
- Rastrigin

Each function was executed in dimensions `d=2`, `d=10`, and `d=30`, using two reproducible seeds: `42` and `7`.

### 2.3 Hyperparameters and stopping criteria

The repository supports explicit configuration and optional grid search. For the benchmark suite used in the final report, each objective uses conservative defaults chosen by the project profiles:

- dimension-aware swarm size
- objective-aware coefficients `w`, `c1`, `c2`
- maximum iterations
- tolerance and patience
- velocity cap ratio

These defaults were selected to favor stable convergence rather than raw speed.

### 2.4 Metrics collected

For each run, the project records:

- final best fitness
- convergence AUC
- convergence iteration
- total wall-clock time
- evaluation time
- particle update time
- estimated overhead

For the PySwarm baseline, only metrics that actually exist are reported. Because PySwarm does not expose the per-iteration history used by the internal PSO implementation, AUC and convergence iteration are intentionally left blank instead of being approximated with fake values.

### 2.5 Reproducibility

The project is reproducible by design:

- every run accepts a seed
- the seed is stored in the summary files
- execution metadata is persisted
- the summary includes timing breakdown and configuration
- the repository exposes CLI scripts for the main workflows
- the same workflows are also exposed as VS Code `Run and Debug` profiles via `.vscode/launch.json`

Main reproducible commands from the terminal:

```bash
python -m scripts.run_pso
python -m scripts.run_benchmarks --dims 2 10 30 --seeds 42 7 --no-grid-search
python -m scripts.run_grid_search
python -m scripts.make_viz --objective sphere --dim 2 --seed 42
python -m scripts.analyze_results
```

Equivalent reproducible workflows are available in VS Code without typing commands manually.
The user can open `Run and Debug` and choose a predefined profile such as:

- `PSO: Run single (default)`
- `PSO: Run single (choose objective, dim, seed)`
- `PSO: Run benchmarks (quick)` or `PSO: Run benchmarks (default)`
- `PSO: Run grid search (quick)` or `PSO: Run grid search (default)`
- `PSO: Make visualization (choose values)`
- `PSO: Analyze results (default)`
- `PSO: Clean outputs`

This editor integration is only a launch layer. It still executes the same
Python entry-point scripts under `scripts/`, so reproducibility is preserved
whether the workflow is started from the terminal or from VS Code.

## 3. Implementation Summary

### 3.1 Architecture

The repository is split by responsibility:

- `.vscode/`: optional editor configuration for `Run and Debug` execution profiles
- `core/`: particle, swarm, and PSO loop
- `objectives/`: benchmark functions
- `options/`: abstract interfaces and base strategies
- `parallel/`: threading and multiprocessing evaluators
- `experiment/`: run orchestration and grid search
- `utils/`: persistence, logging, metadata
- `viz/`: convergence plots and swarm animation
- `scripts/`: user-facing reproducible entry points

The most important software-engineering decision is that the PSO core is evaluator-agnostic. `PSO` depends only on:

- `FitnessEvaluator`
- `BoundsPolicy`
- `Topology`

This makes the implementation easier to maintain and also makes the experiments more scientifically defensible.

### 3.2 Bounds and topology

The project uses a clamp-based boundary policy. When a particle hits the search box boundary, its position is clipped and the offending velocity component is reset to zero.

This choice is simple, stable, and easy to test. It may introduce some bias near the borders, but it avoids large rebounds and worked reliably across the benchmark suite.

The topology implemented is canonical global-best PSO. All particles are influenced by the swarm-wide best-known position.

### 3.3 Parallel strategies

- `V0` evaluates each particle sequentially.
- `V1` uses `ThreadPoolExecutor` to evaluate particles concurrently.
- `V2` uses `ProcessPoolExecutor` and batches particles before dispatching them to worker processes.

Batching is important in `V2` because sending one particle per task would increase pickling and IPC overhead unnecessarily.

## 4. Results

The aggregate results used here come from the generated analysis in `reports/analysis/analysis_summary.csv`.

### 4.1 Sphere

Sphere is the cleanest case. `V0`, `V1`, and `V2` reach essentially the same final fitness in every tested dimension, often down to numerical precision close to zero.

The main story is timing:

- In `d=2`, `V0` is clearly best among the in-house implementations.
- In `d=10`, `V1` gets very close to `V0` and slightly beats it on average.
- In `d=30`, `V0` remains the most reliable choice, with `V1` very close and `V2` still slower.

Interpretation: Sphere is cheap to evaluate, so parallel overhead dominates. This is exactly the kind of case where parallel execution may be mathematically correct but practically unnecessary.

### 4.2 Ackley

Ackley is more irregular than Sphere but the three internal strategies still remain aligned in optimization quality:

- In `d=2` and `d=10`, `V0`, `V1`, and `V2` converge to essentially the same final fitness.
- In `d=30`, the final fitness is no longer near machine zero and there is visible seed sensitivity.

Timing remains mixed:

- `V1` slightly improves time for `d=2`.
- `V0` and `V1` are very close in `d=10`.
- In `d=30`, neither `V1` nor `V2` gives a strong and stable advantage over `V0`.

Interpretation: the internal strategies remain comparable in quality, which supports the claim that the shared core behaves consistently. However, the benefit of concurrency is still limited because the objective is not expensive enough to amortize all overheads.

### 4.3 Rosenbrock

Rosenbrock is one of the most informative benchmarks in the suite.

- In `d=2`, all three internal methods converge very well, and `V1` is the fastest on average.
- In `d=10`, the runs often hit the iteration budget before clear convergence. Even so, `V0`, `V1`, and `V2` still end with the same best fitness for the same seed.
- In `d=30`, the problem becomes difficult enough that all strategies operate far from the global optimum, but the internal methods remain tightly grouped while PySwarm becomes much worse and much more variable.

Interpretation: Rosenbrock shows that keeping the PSO core fixed was the correct design choice. The optimization behavior is stable across `V0`, `V1`, and `V2`, while the external baseline highlights that implementation details and default behaviors matter.

### 4.4 Rastrigin

Rastrigin is the benchmark that exposes the limits of fixed defaults most clearly.

- In `d=2`, the internal methods solve the problem cleanly and reach zero.
- In `d=10`, the internal strategies remain equivalent to each other, but they get trapped at worse fitness values than PySwarm on average.
- In `d=30`, the internal methods again outperform PySwarm in quality, although timings remain mixed.

Interpretation: this is not evidence of a bug in the implementation. Instead, it shows that multimodal functions can require different hyperparameter choices, and that grid search is valuable rather than optional if the goal is strongest possible final fitness.

## 5. Critical Discussion

### 5.1 GIL and threading

`V1` uses threads, so it does not guarantee a speedup for Python-heavy fitness functions because of the Global Interpreter Lock (GIL).

In practice, the results show exactly that:

- for cheap objectives, `V1` is often slower than `V0`
- for some medium-cost cases, `V1` can be competitive or slightly faster

This makes `V1` useful as a concurrency baseline, but not as a universally better strategy.

### 5.2 IPC and multiprocessing

`V2` avoids the GIL by using processes, but this introduces other costs:

- process management
- serialization and deserialization
- inter-process communication

Batching reduces the problem, but it does not remove it. In many of the final results, `V2` remains slower than `V0` because the objective functions are still too cheap relative to the communication overhead.

### 5.3 Vectorization as a missing reference

Although `V4` was not implemented in this delivery, the results strongly suggest that vectorization would be an important next step.

For numerically cheap benchmarks like Sphere or Ackley, vectorized NumPy evaluation is likely to outperform both threads and processes because it reduces Python-level loop overhead without paying the IPC cost of multiprocessing.

So the critical lesson is not only that `V2` is not always worth it, but also that “parallelism” is not the only optimization path. Sometimes the best improvement is to remove Python loops entirely.

### 5.4 Trade-offs

The project exposes a useful practical trade-off:

- `V0` is simple, deterministic, and often fastest for cheap objectives.
- `V1` is easy to plug in and sometimes competitive, but limited by the GIL.
- `V2` is conceptually the strongest parallel model in this scope, but only helps when the evaluation cost is high enough.

That trade-off is exactly the kind of conclusion the benchmark suite was meant to reveal.

## 6. Recommendations

Based on the current results, the most reasonable recommendations are:

1. Use `V0` as the default for cheap objective functions or small dimensions.
2. Use `V1` mainly as a concurrency comparison point, or when the objective internally releases the GIL or calls vectorized native code.
3. Use `V2` only when the fitness evaluation is expensive enough to amortize process and IPC overhead.
4. Run grid search for multimodal problems such as Rastrigin when the final fitness matters more than convenience.
5. Treat PySwarm as an external reference, not as a directly equivalent implementation.
6. Prioritize `V4` vectorization as the next extension because it is likely to be more useful than raw multiprocessing for several benchmark functions.

## 7. Conclusion

The delivery meets the target scope up to `V2` and provides a coherent PSO codebase with:

- modular architecture
- reproducible scripts
- persistence of experiment data
- timing instrumentation
- tests
- visualization
- analysis of saved results

The most important experimental conclusion is that parallelism is not automatically beneficial. For the benchmark functions used here, the shared PSO core behaves consistently across `V0`, `V1`, and `V2`, but speedup depends strongly on granularity and overhead. In many realistic cases, the simplest sequential implementation remains the best choice, while multiprocessing only becomes attractive when evaluation cost is sufficiently large.

This is a strong result rather than a disappointing one: it shows that the project was designed carefully enough to reveal the real trade-offs instead of assuming that more concurrency must always mean better performance.
