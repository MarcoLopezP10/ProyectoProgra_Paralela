# Final Report — PSO `V0` to `V4`

## 1. Objective and Scope

This project implements a maintainable Particle Swarm Optimization (PSO) codebase in Python and uses it as a controlled experimental platform to compare several execution strategies under identical optimization conditions.

The implemented scope includes:

- `V0`: sequential PSO
- `V1`: threaded fitness evaluation
- `V2`: multiprocessing fitness evaluation with batching
- `V3`: `asyncio`-based cooperative evaluation
- `V4`: NumPy-vectorized evaluation and update for supported numerical objectives

The key methodological rule is that the PSO algorithm itself should remain the same across versions. Only the execution strategy is allowed to change.

That decision makes the comparison fair: if two versions differ in time but not in final fitness for the same seed and hyperparameters, the difference can be attributed to execution strategy rather than to hidden algorithmic changes.

---

## 2. Methodology

### 2.1 Shared protocol

All benchmark comparisons follow the same protocol:

1. choose one objective function and one dimension
2. initialize the same swarm with the same seed
3. run all PSO variants with the same hyperparameters, topology, bounds, and stopping criteria
4. optionally compare against a PySwarm baseline
5. persist timing, convergence, and summary metrics

This preserves comparability across methods.

### 2.2 Objectives used

The project evaluates the following synthetic objectives:

- `sphere`
- `ackley`
- `rosenbrock`
- `rastrigin`
- `latency_mix`

The first four are numerical benchmarks. `latency_mix` is intentionally different: it is designed to behave like a latency-aware evaluation workload rather than a pure arithmetic kernel.

### 2.3 Applied objective

Beyond synthetic benchmarks, the project also includes an Economic Load Dispatch (EDL) workflow.

EDL is useful because it moves the project into a constrained real-world optimization setting with:

- generator bounds
- demand balance
- optional transmission losses
- optional valve-point effects

This allows the project to demonstrate not only benchmark performance, but also engineering realism.

### 2.4 Metrics collected

For each run, the project records:

- final best fitness
- convergence iteration
- convergence AUC
- total wall-clock time
- evaluation time
- update time
- residual overhead

This timing breakdown is especially important, because the project compares strategies whose overhead structures are very different.

---

## 3. Implementation Summary

### 3.1 Core architecture

The repository is organized into:

- `core/`: particle, swarm, PSO loop
- `options/`: abstract strategy interfaces
- `parallel/`: concrete evaluator implementations
- `objectives/`: synthetic and applied objectives
- `experiment/`: orchestration
- `utils/`: persistence, metadata, logging
- `viz/`: plots
- `scripts/`: CLI entry points

The most important architecture rule is that the PSO core is strategy-agnostic.

### 3.2 Execution strategies

#### `V0`

Sequential baseline:

- particle evaluations happen one by one
- particle updates happen one by one

This is the correctness and speedup reference.

#### `V1`

Thread-based strategy:

- uses `ThreadPoolExecutor`
- attempts to improve performance through concurrency

This is most useful when evaluations can overlap waiting.

#### `V2`

Process-based strategy:

- uses `ProcessPoolExecutor`
- batches particles to reduce inter-process overhead

This is the main true-parallel baseline outside the GIL, but it pays serialization and IPC costs.

#### `V3`

Async strategy:

- uses `asyncio.gather`
- targets latency-aware objectives

It is not designed to accelerate pure numerical computation, but to overlap cooperative waiting.

#### `V4`

Vectorized strategy:

- uses NumPy arrays to evaluate supported objectives in batch
- vectorizes velocity and position updates
- falls back cleanly when the objective is unsupported

This is not "parallelism" in the thread/process sense. It is an implicit numeric acceleration strategy that reduces Python-loop overhead.

---

## 4. Main Experimental Findings

## 4.1 Small numerical benchmark: `sphere`, `d=2`

Representative result:

- `V0`: `0.0078s`
- `V1`: `0.0493s`
- `V2`: `1.7865s`
- `V3`: `0.0240s`
- `V4`: `0.0042s`

Observed fitness:

- `V0` to `V4`: identical at `1.018912e-04`

Interpretation:

- the benchmark is so cheap that concurrency overhead dominates
- `V1`, `V2`, and `V3` pay machinery cost without enough useful work to amortize it
- `V4` still improves over `V0` because vectorization reduces Python-level overhead even on a small numerical task

This is an important baseline result: it shows that faster execution is possible without changing optimization behavior.

---

## 4.2 Larger numerical benchmark: `sphere`, `d=30`

Representative result:

- `V0`: `0.3971s`
- `V1`: `0.4620s`
- `V2`: `2.4091s`
- `V3`: `0.3549s`
- `V4`: `0.0803s`

Observed fitness:

- `V0` to `V4`: identical at `3.092036e-03`

Interpretation:

- this is where `V4` becomes clearly superior
- the dimension and arithmetic workload are large enough that vectorization strongly amortizes setup costs
- `V4` achieves `4.948x` speedup over `V0`
- `V2` remains unattractive because process overhead is still too high

This result is one of the clearest demonstrations that vectorization is the most effective optimization path for structured numerical PSO workloads.

---

## 4.3 Nontrivial numerical benchmark: `ackley`, `d=30`

Representative result:

- `V0`: `0.6678s`
- `V1`: `1.0877s`
- `V2`: `2.5430s`
- `V3`: `0.5956s`
- `V4`: `0.0913s`

Observed fitness:

- `V0` to `V4`: identical at `2.124390e+00`

Interpretation:

- `ackley` is more complex than `sphere`, but still highly regular numerically
- `V4` gains even more here: `7.312x` over `V0`
- this proves that the `V4` benefit is not limited to trivial convex problems

The PySwarm baseline reaches a better final fitness in this particular run, but that does not invalidate `V4`. PySwarm is a separate external implementation and is not directly equivalent to the internal PSO family.

---

## 4.4 Highly multimodal numerical benchmark: `rastrigin`, `d=30`

Representative result:

- `V0`: `1.2815s`
- `V1`: `0.5773s`
- `V2`: `1.7934s`
- `V3`: `0.5401s`
- `V4`: `0.1064s`

Observed fitness:

- `V0` to `V4`: identical at `5.019675e+01`

Interpretation:

- the optimization landscape is difficult, but the arithmetic remains vectorizable
- `V4` is again the clear winner with `12.040x` speedup over `V0`
- this is perhaps the strongest evidence in the project that vectorization targets computation structure, not only easy landscapes

An additional interesting point is that `V1` and `V3` also outperform `V0` here. This does not change the main story, but it shows that runtime behavior can be more nuanced than a purely theoretical ranking.

---

## 4.5 Latency-aware objective: `latency_mix`

### Small case: `d=2`

Representative result:

- `V0`: `0.5734s`
- `V1`: `0.1468s`
- `V2`: `1.8342s`
- `V3`: `0.1005s`
- `V4`: `0.5366s`

### Larger case: `d=10`

Representative result:

- `V0`: `6.8641s`
- `V1`: `1.1723s`
- `V2`: `4.3803s`
- `V3`: `0.4815s`
- `V4`: `7.0432s`

Observed fitness:

- `V0` to `V4`: identical in both runs

Interpretation:

- this objective is the natural home of `V3`
- `V3` achieves `14.255x` speedup over `V0` on the larger run
- `V1` also improves substantially because threads can overlap waiting
- `V4` does not help here, because the bottleneck is not arithmetic loops but latency-like behavior
- `V4` safely falls back and remains near `V0`, which is the correct engineering behavior

This result is crucial because it shows that `V4` is not a universal winner, and that the project genuinely distinguishes between workload types.

---

## 4.6 Applied optimization: EDL

The EDL experiments are valuable because they show that the PSO framework also works on a realistic constrained optimization problem.

### Important compatibility note

Not every EDL variant is valid for every case file.

For example:

- `edl_case_40u.json` supports `edl_1` and `edl_2`
- `edl_4` is marked unavailable there because the case does not include a transmission-loss model

This is correct behavior, not a bug.

### `edl_case_40u`, `edl_1`

Representative result:

- `V0`: `1.8227s`
- `V1`: `1.7914s`
- `V2`: `3.3977s`

Observed result:

- same best fitness for all three methods
- same dispatch vector for all three methods

Interpretation:

- the PSO family remains consistent in an applied setting
- `V2` is again the slowest because multiprocessing overhead is still substantial
- `V1` slightly wins this particular case by time tie-break

### `edl_case_40u`, `edl_2`

Representative result:

- `V0`: `1.5760s`
- `V1`: `2.0943s`
- `V2`: `4.1355s`

Observed result:

- same best fitness for all three methods
- same dispatch vector for all three methods

Interpretation:

- once again, optimization behavior is preserved across strategies
- on this variant the sequential baseline retakes the lead
- the applied case confirms that concurrency is not automatically beneficial

### What EDL contributes to the report

The EDL workflow strengthens the project in three ways:

1. it proves the architecture works beyond toy mathematical benchmarks
2. it confirms the same fairness property in an applied optimization problem
3. it highlights the importance of domain-aware validation, such as checking whether a loss model is actually available

---

## 5. Critical Discussion

### 5.1 Why `V2` is often disappointing

`V2` is conceptually powerful, but in practice it pays:

- process startup cost
- serialization cost
- inter-process communication cost

In this project, many objectives are still too lightweight relative to those overheads. That is why `V2` is often slower than expected.

### 5.2 Why `V3` matters

`V3` proves that "performance optimization" is not only about CPU parallelism.

If the objective exposes latency-shaped work, `asyncio` can be by far the best strategy. The `latency_mix` results are the strongest evidence of this.

### 5.3 Why `V4` matters most for numerical workloads

`V4` consistently shows the largest gains on:

- `sphere`
- `ackley`
- `rastrigin`

especially in larger dimensions.

The main reason is simple:

- Python loops are expensive
- NumPy batch operations move the work into optimized native code

This gives the project a strong and clear final message: for structured numerical PSO workloads, vectorization is often more valuable than explicit multiprocessing.

### 5.4 Why identical fitness is one of the strongest results

Across the PSO family, the same final fitness for the same seed means:

- no hidden algorithmic drift between versions
- no unfair tuning differences
- no accidental change in optimization behavior

That is what gives the runtime comparisons credibility.

---

## 6. Recommendations

Based on the final implementation and results:

1. Use `V0` as the safe general-purpose baseline.
2. Use `V1` when the objective may benefit from concurrent waiting overlap.
3. Use `V2` only when evaluation cost is high enough to amortize process overhead.
4. Use `V3` for latency-aware or cooperative-concurrency objectives.
5. Use `V4` for structured numerical objectives, especially in medium and large dimensions.
6. Use EDL to demonstrate applied consistency, not to claim that a single execution strategy always wins.

---

## 7. Conclusion

The project now provides a coherent PSO framework with:

- modular architecture
- reproducible execution
- persistence and reporting
- benchmark and applied workflows
- multiple execution strategies
- strong validation through tests and experimental consistency

The most important final conclusion is that there is no universally best execution strategy.

Instead:

- `V3` is best when the bottleneck is waiting
- `V4` is best when the bottleneck is structured numerical computation
- `V0` remains a strong baseline
- `V2` is informative but often limited by overhead

That is a strong result, because it shows that the project reveals real trade-offs rather than forcing a simplistic “more parallelism is always better” narrative.
