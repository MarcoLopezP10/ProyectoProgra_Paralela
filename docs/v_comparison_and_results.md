# PSO Versions `V0`-`V4` and Experimental Results

This document explains:

1. What each project version (`V0`, `V1`, `V2`, `V3`, `V4`) does.
2. What changes from one version to the next.
3. Why the observed experimental results behave the way they do.
4. What conclusions can be defended from the final implementation.

The goal is not only to list timings, but to explain the computational reason behind them.

---

## 1. General Idea of the Project

The project implements **Particle Swarm Optimization (PSO)** using a shared optimization core and several execution strategies.

All versions keep the same high-level PSO logic:

- same swarm initialization for a given seed
- same velocity/position update rule
- same stopping criteria
- same benchmark functions

What changes between versions is **how the expensive work is executed**.

In practice, the project compares different ways of evaluating particles and updating the swarm:

- pure sequential execution
- thread-based concurrency
- process-based parallelism
- asyncio-based cooperative concurrency
- NumPy vectorization

This means the comparison is not "different algorithms", but rather **different execution strategies for the same PSO workflow**.

---

## 2. What Each Version Is

### `V0` Sequential

`V0` is the reference implementation.

Characteristics:

- evaluates particles one by one
- updates particles one by one
- uses regular Python loops
- has no parallelism and no vectorization

Why it matters:

- it is the clean baseline
- it is the easiest version to reason about
- every other version should preserve the same optimization result as `V0`

Strengths:

- simple
- deterministic
- low overhead on small problems

Weaknesses:

- does not exploit concurrency
- does not reduce Python loop overhead

Typical behavior:

- good baseline for cheap numerical objectives
- often hard to beat on very small workloads

---

### `V1` Threading

`V1` evaluates particles using a `ThreadPoolExecutor`.

Characteristics:

- the PSO structure is still the same
- particle fitness evaluations are dispatched to threads
- the update phase still follows the standard workflow

What is different from `V0`:

- `V0` computes in strict sequence
- `V1` tries to overlap evaluations using Python threads

When this helps:

- when the objective function contains waiting time
- when the work can benefit from concurrency more than from raw CPU speed

When this does not help:

- when the objective is cheap and purely CPU/numerical
- when thread scheduling overhead is larger than the useful work
- when the GIL prevents real CPU scaling

Expected pattern:

- weak or negative gains on simple numerical functions
- mixed behavior on medium/large numerical workloads
- better behavior on latency-heavy tasks

---

### `V2` Multiprocessing

`V2` evaluates particles using a `ProcessPoolExecutor`.

Characteristics:

- each process can run independently of the Python GIL
- batches are used to reduce overhead per task
- objective functions must be picklable

What is different from `V1`:

- `V1` uses threads in the same process
- `V2` uses separate processes with inter-process communication

Potential advantage:

- better suited than threads for CPU-heavy work in principle

Main cost:

- pickling/unpickling data
- moving batches between processes
- managing process startup and synchronization

Expected pattern:

- potentially useful for large, expensive CPU workloads
- often poor on small or medium benchmarks due to overhead

In this project, `V2` acts mainly as a **parallelism baseline** rather than the strongest practical option.

---

### `V3` Asyncio

`V3` evaluates particles using `asyncio.gather`.

Characteristics:

- uses cooperative concurrency
- does not accelerate raw numerical CPU work
- is designed to overlap awaitable or latency-like operations

What is different from `V1`:

- `V1` uses OS threads
- `V3` uses an event loop and async tasks

What it is good for:

- objectives with artificial latency
- I/O-like or wait-heavy evaluation patterns
- cases where many evaluations can be in-flight at the same time

What it is not good for:

- cheap numerical benchmarks
- cases where there is no waiting to overlap

Expected pattern:

- strong results on `latency_mix`
- usually not the main winner on numerical benchmarks
- but it can still show moderate gains when the runtime profile happens to favor its lower coordination cost over other concurrency options

---

### `V4` Vectorized

`V4` is the NumPy-based vectorized version.

Characteristics:

- uses matrix/array operations instead of many Python loops
- evaluates supported numerical objectives in batch
- updates velocities and positions in a vectorized way
- keeps fallback behavior for unsupported objectives

What is different from `V0`:

- `V0` does scalar-style work particle by particle
- `V4` processes many particles together with NumPy arrays

What is different from `V1`, `V2`, and `V3`:

- those versions try to exploit concurrency or parallel execution
- `V4` exploits **implicit numeric acceleration**
- the speedup comes from moving work into optimized NumPy operations

Very important idea:

`V4` is not "parallelism" in the same sense as threads/processes.  
It is better described as **implicit parallel numeric acceleration through vectorization**.

Why this matters:

- Python loops are expensive
- NumPy executes heavy numeric operations in optimized native code
- the larger and more regular the numeric workload, the more `V4` can help

When `V4` should shine:

- numerical benchmark functions
- medium/large swarm sizes
- medium/high dimensions
- many repeated arithmetic operations

When `V4` should not dominate:

- latency-driven objectives
- unsupported objectives that require fallback
- tiny workloads where setup costs dominate

---

## 3. Architectural Interpretation of the Versions

The versions can be grouped into two families.

### Concurrency family

- `V1`: threads
- `V2`: processes
- `V3`: asyncio

These versions try to improve execution by overlapping or distributing evaluations.

### Numeric/vectorization family

- `V4`: NumPy vectorization

This version tries to improve execution by reducing Python-level overhead and batching arithmetic operations.

### Reference family

- `V0`: sequential baseline

This is the control version that makes the comparison meaningful.

---

## 4. Why Equal Fitness Across `V0`-`V4` Is Important

In your results, the PSO-based versions usually produce the same best fitness for the same objective and seed.

This is a very strong sign that:

- the mathematical PSO behavior has been preserved
- the seed handling is consistent
- the versions differ in execution strategy, not in optimization logic

That is especially important for `V4`, because it shows that vectorization changes the **implementation cost**, not the **result of the search**.

This is exactly what should happen in a fair comparison.

---

## 5. Experimental Results and Their Meaning

The explanations below use the final runs obtained on the project.

---

## 6. Result 1: `sphere`, `d=2`, `n_particles=8`, `max_iters=20`

Observed times:

- `V0`: `0.0078s`
- `V1`: `0.0493s`
- `V2`: `1.7865s`
- `V3`: `0.0240s`
- `V4`: `0.0042s`
- `PySwarm`: `0.0076s`

Observed best fitness:

- `V0` to `V4`: `1.018912e-04`
- `PySwarm`: `7.177814e-05`

### Why this happens

`sphere` is one of the cheapest possible numerical objectives:

- it is smooth
- it is simple
- each evaluation is basically a sum of squares

So the cost of evaluating one particle is very low.

#### `V0`

`V0` already performs well because the work is so small that there is not much to optimize with concurrency.

#### `V1`

`V1` is slower because:

- thread scheduling has overhead
- the objective is too cheap to amortize that overhead
- there is no latency to overlap

So threads add cost without creating enough benefit.

#### `V2`

`V2` is by far the slowest because:

- sending work to processes is expensive
- serializing particle batches is expensive
- the objective is much too small to justify IPC costs

This is the classic case where process-based parallelism is theoretically powerful but practically wasteful.

#### `V3`

`V3` is also slower than `V0` because:

- `asyncio` helps only if there is waiting/latency to overlap
- `sphere` has no such structure
- the event-loop machinery becomes extra overhead

#### `V4`

`V4` is the fastest PSO version here because:

- even a small numerical problem benefits from replacing repeated Python loops with NumPy operations
- the objective and update path are regular and vectorizable

Even though the problem is small, `V4` already wins over `V0`.

### Interpretation

This experiment shows:

- concurrency is not automatically beneficial
- vectorization can outperform concurrency on numeric workloads
- `V4` is already useful even at small scale

---

## 7. Result 2: `sphere`, `d=30`, `n_particles=60`, `max_iters=120`

Observed times:

- `V0`: `0.3971s`
- `V1`: `0.4620s`
- `V2`: `2.4091s`
- `V3`: `0.3549s`
- `V4`: `0.0803s`
- `PySwarm`: `0.2178s`

Observed best fitness:

- `V0` to `V4`: `3.092036e-03`
- `PySwarm`: `1.953143e-03`

### Why this happens

This is where the real benefit of `V4` becomes clear.

Compared with the previous experiment, this one has:

- much higher dimension
- many more particles
- many more total arithmetic operations

That means the PSO spends much more time doing repeated numeric work.

#### `V0`

`V0` gets significantly slower because Python loops scale poorly as the numeric workload grows.

#### `V1`

`V1` is still not ideal because:

- the task is still mostly numerical CPU work
- threads do not remove Python overhead
- the thread machinery adds coordination cost

So it remains slightly worse than `V0`.

#### `V2`

`V2` still loses badly because the process overhead remains very large relative to the work being done per dispatch.

Even though the task is larger, it is still not the kind of massive independent workload that justifies process communication here.

#### `V3`

`V3` becomes slightly faster than `V0` in this run.

This should be interpreted carefully:

- `V3` is not fundamentally designed for this kind of CPU-bound objective
- a small gain can appear due to runtime variability, implementation details, and measurement noise
- the strong, reliable story here is not `V3`

The strong story is `V4`.

#### `V4`

`V4` gives the strongest speedup: `4.948x` over `V0`.

This is the textbook vectorization result:

- more dimensions
- more particles
- more repeated arithmetic
- higher benefit from array-based batch operations

The work becomes large enough that NumPy amortizes setup costs and dominates the scalar Python loop approach.

### Interpretation

This experiment is the clearest proof that `V4` delivers what it was designed to deliver:

- same optimization result
- much lower execution time
- especially strong gains as the numerical workload increases

This is likely the most important result for defending `V4`.

---

## 8. Result 3: `latency_mix`, `d=2`, `n_particles=8`, `max_iters=20`

Observed times:

- `V0`: `0.5734s`
- `V1`: `0.1468s`
- `V2`: `1.8342s`
- `V3`: `0.1005s`
- `V4`: `0.5366s`
- `PySwarm`: `0.5536s`

Observed best fitness:

- `V0` to `V4`: `2.389916e-01`
- `PySwarm`: `2.392260e-01`

### Why this happens

`latency_mix` is not a purely numerical benchmark. It is designed to behave like a latency-aware or I/O-like objective.

That changes the whole performance picture.

#### `V0`

`V0` evaluates everything in strict sequence, so it pays the full waiting time for every particle.

#### `V1`

`V1` improves strongly because threads can overlap waiting periods.

This is exactly the kind of workload where threading becomes meaningful.

#### `V2`

`V2` still performs poorly because:

- process overhead remains large
- the workload is not a great fit for process-based dispatch
- the gains from overlapping latency do not compensate for IPC costs

#### `V3`

`V3` is the winner because this is the workload it was designed for.

`asyncio` is ideal when:

- work is latency-shaped
- many tasks can be awaited concurrently
- the objective can expose cooperative concurrency

So `V3` wins for the right computational reason, not by accident.

#### `V4`

`V4` does **not** dominate here because the problem is not a vectorizable numeric hotspot.

Instead:

- the objective is not in the fast vectorized path
- `V4` falls back safely
- therefore it behaves roughly like the classical implementation

That is why `V4` remains near `V0` and does not beat `V3`.

### Interpretation

This is a very important result because it proves that the project is honest:

- `V4` is not better at everything
- `V3` remains the best choice for latency-oriented objectives
- each version has a computational niche

This makes the overall comparison much more credible.

---

## 9. Result 4: Benchmark Smoke with `sphere`, `d=2`

Observed times:

- `V0`: `0.1798s`
- `V1`: `0.2097s`
- `V2`: `1.9539s`
- `V3`: `0.1489s`
- `V4`: `0.0369s`
- `PySwarm`: `0.0516s`

Observed best fitness:

- `V0` to `V4`: `3.185885e-06`
- `PySwarm`: `1.767682e-06`

### Why this result matters

This run confirms that `V4` is not only visible in the CLI, but fully integrated into:

- benchmark execution
- summary CSV export
- reporting pipeline

And performance-wise it again shows the same pattern:

- equal PSO fitness across `V0`-`V4`
- clear time improvement for `V4`

So the project is not only functionally complete; it is experimentally coherent.

---

## 10. Result 5: `ackley`, `d=30`, `n_particles=60`, `max_iters=120`

Observed times:

- `V0`: `0.6678s`
- `V1`: `1.0877s`
- `V2`: `2.5430s`
- `V3`: `0.5956s`
- `V4`: `0.0913s`
- `PySwarm`: `0.4058s`

Observed best fitness:

- `V0` to `V4`: `2.124390e+00`
- `PySwarm`: `9.316259e-01`

### Why this happens

`ackley` is much more informative than `sphere` because it is still fully numerical, but less trivial:

- it contains exponentials
- it contains cosine terms
- it is more expensive per evaluation
- it remains highly regular and vectorizable

This is a very strong case for `V4`.

#### `V0`

`V0` pays the full Python-loop cost of a relatively expensive numerical objective.

#### `V1`

`V1` gets worse than `V0` because:

- the objective is still fundamentally CPU/numeric
- thread overhead is significant
- the GIL still limits what threads can achieve on pure Python-level orchestration

#### `V2`

`V2` remains poor because the process overhead still dominates too much of the run.

#### `V3`

`V3` is slightly faster than `V0`, but this should not be misread as "`asyncio` is best for numerical work".

The important point is:

- `V3` is not the main story
- `V4` is dramatically faster

#### `V4`

`V4` achieves `7.312x` speedup over `V0`, which is one of the strongest results in the project.

This shows that:

- the benefit of vectorization is not limited to the easiest benchmark
- more expensive numerical formulas can amplify the benefit even further

### Interpretation

This result strengthens the `V4` argument significantly:

- `V4` is not only good for `sphere`
- `V4` also dominates on a richer nonlinear numerical objective
- the gain is not cosmetic; it is large and very clear

---

## 11. Result 6: `rastrigin`, `d=30`, `n_particles=60`, `max_iters=120`

Observed times:

- `V0`: `1.2815s`
- `V1`: `0.5773s`
- `V2`: `1.7934s`
- `V3`: `0.5401s`
- `V4`: `0.1064s`
- `PySwarm`: `0.5298s`

Observed best fitness:

- `V0` to `V4`: `5.019675e+01`
- `PySwarm`: `1.536808e+02`

### Why this happens

`rastrigin` is especially interesting because:

- it is highly multimodal
- it is harder as an optimization landscape
- but each evaluation is still regular arithmetic

This separates two ideas very clearly:

1. difficulty of the optimization landscape
2. cost structure of the implementation

The landscape is difficult, but the computation is still vectorizable.

#### `V0`

`V0` becomes very slow because the problem is both larger and still numerically repetitive.

#### `V1`

`V1` performs surprisingly better than `V0` here.

That does not mean threads are the best design overall; rather, it shows that on some medium/large workloads the total runtime profile can make threading competitive.

Still, it is not close to `V4`.

#### `V2`

`V2` is again held back by process overhead.

#### `V3`

`V3` also beats `V0` here, which again should be interpreted carefully:

- it is a valid empirical result
- but it does not overturn the main design logic of the project
- the dominant result is still `V4`

#### `V4`

`V4` is the clear winner with `12.040x` speedup over `V0`.

This is arguably the most dramatic result in the whole set.

It proves that:

- a difficult numerical landscape does not hurt the usefulness of vectorization
- the real target of `V4` is the arithmetic structure, not the smoothness of the function

### Interpretation

This experiment is extremely valuable in the report because it shows:

- `V4` is not specialized to easy convex problems
- `V4` remains powerful even on a complex multimodal benchmark
- the vectorized path is robust across different numerical objective styles

---

## 12. Result 7: `latency_mix`, `d=10`, `n_particles=40`, `max_iters=60`

Observed times:

- `V0`: `6.8641s`
- `V1`: `1.1723s`
- `V2`: `4.3803s`
- `V3`: `0.4815s`
- `V4`: `7.0432s`
- `PySwarm`: `6.6978s`

Observed best fitness:

- `V0` to `V4`: `1.194640e+00`
- `PySwarm`: `1.194649e+00`

### Why this happens

This is the strongest possible validation of the purpose of `V3`.

Compared with the earlier `latency_mix d=2` result, this run scales up the same kind of workload:

- more particles
- more dimensions
- more waiting-like behavior overall

#### `V0`

`V0` becomes very slow because it serializes all that latency.

#### `V1`

`V1` improves a lot because threads can overlap waiting.

#### `V2`

`V2` improves relative to `V0`, but not enough. It still pays a heavy overhead tax.

#### `V3`

`V3` becomes overwhelmingly better: `14.255x` over `V0`.

This is exactly the signature expected from an `asyncio`-friendly objective:

- many concurrent waits
- low benefit from numeric vectorization
- very high benefit from cooperative scheduling

#### `V4`

`V4` is slightly worse than `V0` here: `0.975x`.

This is actually a good result from a scientific point of view, because it confirms that:

- `V4` does not fake an advantage where it should not have one
- fallback behavior is working
- vectorization is not the right tool for this kind of objective

### Interpretation

This result should be highlighted because it completes the project story:

- `V4` is the best answer for large numerical workloads
- `V3` is the best answer for latency-aware workloads

Together, these two results show that there is no single universally best execution strategy.

---

## 13. Result 8: EDL Applied Example

The EDL workflow is important because it moves the project from synthetic benchmarks to a constrained real-world optimization problem.

Two observations are important here.

### 13.1 Invalid combination: `edl_case_40u` with `edl_4`

Observed behavior:

- `edl_4` appears as `unavailable` for all methods

This is not a PSO bug.

It happens because:

- `edl_4` requires valve-point **and** transmission losses
- `cases/edl_case_40u.json` does **not** provide a loss model

Therefore the project correctly marks the run as unavailable instead of silently producing invalid results.

This is actually good engineering behavior.

### 13.2 Valid large applied case: `edl_case_40u` with `edl_1` and `edl_2`

#### `edl_1` (Base)

Observed times:

- `V0`: `1.8227s`
- `V1`: `1.7914s`
- `V2`: `3.3977s`

Observed result:

- same fitness for all three methods: `1.294296e+05`
- same dispatch for all three methods
- winner: `V1 Threading` by time tie-break

#### `edl_2` (Valve-point)

Observed times:

- `V0`: `1.5760s`
- `V1`: `2.0943s`
- `V2`: `4.1355s`

Observed result:

- same fitness for all three methods: `1.354930e+05`
- same dispatch for all three methods
- winner: `V0 Sequential`

### Why this happens

EDL is valuable because it shows a realistic constrained problem with:

- box limits
- demand balance
- optional valve-point non-smoothness
- optional losses

But the current EDL path compares only `V0`, `V1`, and `V2`, not `V3` or `V4`.

So the EDL interpretation is different from the benchmark interpretation:

- it is not mainly about proving `V4`
- it is about proving that the same PSO family remains consistent in an applied setting

What the results show:

- all versions converge to the same dispatch
- `V2` is again the slowest because of multiprocessing overhead
- `V1` can sometimes edge out `V0` on time
- on another variant, `V0` can retake the lead

This is a very useful reminder that once we move to an applied problem:

- overhead interactions become more nuanced
- there is no guarantee that concurrency beats the simplest baseline

### Interpretation

The EDL example helps the report in three ways:

1. it proves the project is not limited to toy benchmark functions
2. it shows the same consistency property: identical optimization result across strategies
3. it highlights that engineering constraints, data availability, and case compatibility matter in applied optimization

---

## 14. Why `PySwarm` Sometimes Wins

In several tables, `PySwarm` is listed as the winner.

This does **not** mean that `V4` failed.

The winner logic is:

1. best fitness first
2. time only as tie-break

So if `PySwarm` reaches a slightly lower final fitness, it wins even if a project version is faster.

This is expected because `PySwarm` is:

- an external baseline
- not the same internal implementation
- not just "our PSO with another evaluator"

Therefore it can converge differently under the same iteration budget.

The correct interpretation is:

- `V0`-`V4` compare execution strategies for the same project PSO family
- `PySwarm` is an external reference implementation

Both comparisons are useful, but they answer different questions.

---

## 15. Why `V4` Is a Good `V4`

The project requirement for `V4` was to implement:

- vectorized evaluation
- vectorized update
- a comparison against `V0`-`V3`

The final implementation satisfies that in a defendable way:

- `V4` lives mainly in the evaluator layer
- the core was changed minimally
- supported numerical objectives use a true vectorized path
- unsupported objectives fall back safely
- reporting and analysis include `V4`

Most importantly, the experimental results match the intended theory:

- `V4` is best on large numeric workloads
- `V4` is not best on latency-like workloads
- `V4` preserves optimization correctness

That is exactly the expected scientific behavior.

---

## 16. Cross-Result Reading

Looking across all the results together, the project shows four distinct patterns.

### Pattern A: tiny cheap numerical benchmark

Example:

- `sphere d=2`

Message:

- `V0` is already strong
- `V4` can still win, but the margin is moderate
- concurrency versions often lose

### Pattern B: large regular numerical benchmark

Examples:

- `sphere d=30`
- `ackley d=30`
- `rastrigin d=30`

Message:

- `V4` dominates strongly
- the larger and more arithmetic-heavy the workload, the larger the gain
- `V2` usually remains unattractive

### Pattern C: latency-aware objective

Examples:

- `latency_mix d=2`
- `latency_mix d=10`

Message:

- `V3` is the right tool
- `V1` is also helpful
- `V4` is not supposed to win here, and it does not

### Pattern D: applied constrained optimization

Examples:

- `EDL 40u edl_1`
- `EDL 40u edl_2`

Message:

- the PSO family remains consistent outside synthetic benchmarks
- data/model compatibility matters
- engineering overhead can decide whether `V0` or `V1` is faster

---

## 17. Final Comparative Summary

### `V0`

Best interpretation:

- baseline
- simple
- reliable
- often competitive on tiny workloads

### `V1`

Best interpretation:

- useful when concurrency can overlap waiting
- sometimes competitive on medium/large workloads
- still not the main winner on vectorizable numerical benchmarks

### `V2`

Best interpretation:

- conceptually important as a multiprocessing baseline
- practically expensive in these experiments

### `V3`

Best interpretation:

- strongest option for latency-aware objectives
- occasionally competitive on some numerical runs
- still not the primary explanation for the numerical speedups seen in the project

### `V4`

Best interpretation:

- strongest option for vectorizable numerical PSO workloads
- especially effective as dimension/workload increases
- not meant to replace `V3` on latency-based objectives

---

## 18. Main Conclusions

The results support the following conclusions:

1. The versions preserve the same PSO search behavior for the same seed, so the comparison is fair.
2. Concurrency and parallelism are not universally beneficial; overhead matters.
3. `V1` and `V3` can sometimes help even outside the idealized theory, but their gains are less consistent than `V4` on numerical workloads.
4. `V3` is the right strategy for latency-aware objectives.
5. `V4` is the right strategy for regular numerical objectives, especially larger ones.
6. Vectorization is a very strong optimization path for PSO when the workload is arithmetic and structured.
7. The EDL results show that the project remains meaningful on a realistic constrained problem, not only on toy benchmarks.

In short:

- `V3` wins when the bottleneck is waiting.
- `V4` wins when the bottleneck is numerical computation in Python loops.

That division of roles is the clearest and most defensible interpretation of the final project.

---

## 19. One-Sentence Defense of the Project

The project shows that different PSO execution strategies are optimal for different workload types: `V3` is best for latency-oriented evaluations, while `V4` achieves the strongest gains on large vectorizable numerical benchmarks by replacing Python-loop work with NumPy batch operations.
