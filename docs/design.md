# PSO Design Document

## 1. Purpose

This repository implements a maintainable PSO codebase and uses it as an
experimental platform for comparing multiple execution strategies while keeping
the optimization logic fixed.

Implemented strategies:

- `V0`: sequential baseline
- `V1`: thread-based evaluation
- `V2`: process-based evaluation with batching
- `V3`: `asyncio`-based cooperative evaluation
- `V4`: NumPy-vectorized evaluation and update

The design constraint that drives the project is fairness:

- same PSO equations
- same initialization seed
- same bounds policy
- same topology
- same stopping logic
- same benchmark configuration

Only the execution path changes.

## 2. Core Design Principle

### One optimizer, many execution paths

The repository does not implement five unrelated PSOs.

It implements one PSO core with injectable execution strategies. This keeps the
comparison scientifically meaningful: when two strategies produce the same final
fitness under the same seed, the runtime difference can be attributed to the
execution strategy rather than to an algorithmic change.

Shared across `V0`-`V4`:

- particle state
- swarm state
- velocity update equation
- global-best topology
- clamp bounds policy
- timing and persistence framework

Variable across strategies:

- how fitness values are evaluated
- whether the whole numerical step can be vectorized

## 3. Repository Architecture

The codebase is split by responsibility:

- `core/`: particle, swarm, and PSO loop
- `options/`: interfaces for bounds, topology, and evaluation
- `parallel/`: concrete evaluator implementations
- `objectives/`: synthetic benchmarks and the EDL objective
- `experiment/`: single runs, benchmark suites, grid search, EDL orchestration
- `utils/`: persistence, logging, metadata, method metadata
- `viz/`: convergence plots and swarm animations
- `scripts/`: CLI entry points
- `tests/`: correctness and compatibility tests

Dependency direction:

```text
scripts -> experiment -> core
                     -> parallel
                     -> objectives
                     -> utils
                     -> viz

core -> options
parallel -> options
```

The important architectural property is that `core/` depends on interfaces, not
on specific implementations.

## 4. Main Abstractions

### `FitnessEvaluator`

Defined in `options/evaluator.py`.

Base contract:

```python
evaluate(positions) -> list[float]
```

Extended optional contract:

```python
step(swarm, bounds_handler, topology, w, c1, c2) -> dict[str, float] | None
```

Why it exists:

- `V0`, `V1`, `V2`, and `V3` need batched evaluation only
- `V4` needs a fast path that can vectorize both evaluation and update

This optional `step()` hook allowed `V4` to be added without forking the whole
PSO implementation.

### `BoundsPolicy`

Defined in `options/bounds.py`.

Current implementation:

- `ClampBounds`

Behavior:

- clip the position to the valid search box
- zero the velocity on the coordinates that hit a wall

Reason:

- simple
- deterministic
- stable in high dimensions
- easy to reproduce in both scalar and vectorized code paths

### `Topology`

Defined in `options/topology.py`.

Current implementation:

- `GlobalBestTopology`

Reason:

- keeps the implementation canonical
- preserves a clean extension point for local-best/ring variants
- allows `V4` to explicitly check whether the active topology is compatible
  with the vectorized fast path

## 5. Core PSO Decisions

### Shared state model

Each particle stores:

- current position
- current velocity
- personal best position
- personal best fitness
- RNG state

The swarm stores:

- particles
- swarm-wide best position
- swarm-wide best fitness

This was intentionally kept common to all strategies.

### Deterministic reruns

`PSO.run()` restores the initial snapshot of the swarm before every new run.

This is critical for:

- reproducibility tests
- exact strategy comparisons
- repeated local experiments without rebuilding the optimizer object

### Monotonic global best

The global best is monotonic by contract:

- once a better point is found, it becomes the new reference
- the recorded best never worsens

This is both an algorithmic invariant and a persistence/reporting invariant.

## 6. Strategy-Specific Design

### `V0` Sequential

Purpose:

- correctness baseline
- speedup reference
- minimal overhead path

### `V1` Threading

Purpose:

- lightweight concurrency baseline
- useful when evaluations can overlap waiting

Observed trade-off in final results:

- usually close to `V0` on numerical workloads
- sometimes slightly faster, sometimes slower
- clearly useful only when evaluation latency is present

### `V2` Multiprocessing

Purpose:

- true parallel baseline outside the GIL

Important implementation details:

- objective validation through pickling
- task batching
- worker initializer now installs the objective once per process

Why the initializer matters:

- it avoids re-pickling the objective for every dispatched batch
- it removes avoidable overhead from the earlier implementation

Observed trade-off:

- still expensive on small and medium benchmarks
- sometimes mildly competitive on heavier workloads
- highly sensitive to IPC and serialization costs

### `V3` Asyncio

Purpose:

- cooperative concurrency for latency-shaped workloads

Important design update:

- the evaluator now works even if it is called while an event loop is already
  running
- this is handled by bridging the async evaluation through a worker thread
  instead of incorrectly nesting loops

Why this matters:

- it makes `V3` compatible with notebooks, async environments, and async tests
- it removes a real correctness issue from the previous implementation

Observed trade-off:

- not a universal numeric speedup strategy
- can still be competitive in some numerical runs
- is conceptually strongest for latency overlap

### `V4` Vectorized

Purpose:

- accelerate regular numerical workloads by reducing Python overhead

Supported vectorized objectives:

- `sphere`
- `ackley`
- `rastrigin`
- `rosenbrock`

Compatibility requirements for the fast path:

- vectorized objective available
- `ClampBounds`
- `GlobalBestTopology`

If any requirement is not met, `V4` falls back safely to standard evaluation.

This design preserves correctness and keeps the fast path honest.

## 7. Persistence and Observability

Each run stores:

- `summary.json`
- `history_v*.csv`
- `trajectory_v*.npz`

The summary now distinguishes:

- `winner_internal`: best method among `V0`-`V4`
- `winner_overall`: best method when the external PySwarm reference is also
  considered
- `baseline_reference`: text describing how PySwarm compares against the best
  internal result

This separation is important because PySwarm is an external implementation, not
part of the controlled internal PSO family.

Timing instrumentation includes:

- total wall-clock time
- evaluation time
- update time
- residual overhead

The animation workflow now reads persisted trajectory archives instead of
implicitly re-running a similar optimizer configuration.

## 8. Final Experimental Evidence

The final benchmark protocol executed:

- objectives: `sphere`, `ackley`, `rosenbrock`, `rastrigin`
- dimensions: `2`, `10`, `30`
- seeds: `0`, `1`, `7`, `42`, `123`
- total runs: `60`

Artifacts:

- raw suite: `results/benchmark_suites/final_protocol_check/`
- aggregated analysis: `reports/analysis_final_protocol/`

What the design enabled:

- same final fitness across internal strategies in the final suite
- direct comparison of overhead structures
- workload-sensitive interpretation instead of a one-size-fits-all claim

Representative aggregate findings:

- `sphere d=30`: `V4` mean speedup `3.86x`
- `ackley d=30`: `V4` mean speedup `7.85x`
- `rosenbrock d=10`: `V4` mean speedup `6.31x`
- `rosenbrock d=30`: `V4` mean speedup `5.47x`
- `rastrigin d=10`: `V4` mean speedup `6.01x`
- `rastrigin d=30`: `V4` mean speedup `5.99x`

Interpretation:

- the modular design successfully isolated execution strategy effects
- vectorization is the dominant optimization route for regular numerical PSO
- concurrency without arithmetic restructuring is much less reliable on these
  benchmark sizes

## 9. Applied EDL Evidence

The EDL workflow reuses the same architectural structure:

- same core PSO loop
- same strategy injection model
- different objective only

Observed default execution on the `3U` case:

- `V0`, `V1`, and `V2` reached the same dispatch vector on valid variants
- the winner changed only by runtime
- this confirms behavioral consistency in a constrained engineering problem

That is a strong sign that the abstraction boundaries are correct: the
execution strategy changes runtime characteristics without changing the solution
path of the optimizer family.

## 10. Main Limitations

- `V4` only vectorizes supported numerical objectives
- `V2` remains costly for many realistic benchmark sizes
- `V3` is meaningful mainly when the workload exposes latency or awaitable work
- the project keeps `global-best` only; a local-best topology is not yet
  included

These limitations are acceptable and explicitly documented. They do not weaken
the main conclusion of the project.

## 11. Final Design Conclusion

The architecture achieved its goal:

- one PSO core
- multiple comparable execution strategies
- reproducible runs
- meaningful persistence
- experimentally defensible conclusions

The final evidence strongly supports `V4` as the recommended strategy for
structured numerical workloads, while `V1` and `V3` remain useful reference
points for concurrency-oriented or latency-oriented scenarios.
