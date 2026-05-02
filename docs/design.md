# PSO Design Document

## 1. Purpose

This repository implements a maintainable Particle Swarm Optimization (PSO) codebase in Python and uses it as an experimental platform to compare multiple execution strategies under the same algorithmic conditions.

The implemented scope now includes:

- `V0`: sequential baseline
- `V1`: thread-based concurrent evaluation
- `V2`: process-based parallel evaluation with batching
- `V3`: `asyncio`-based cooperative concurrency
- `V4`: NumPy-vectorized evaluation and update for supported numerical objectives

The main design objective is fairness of comparison:

- the PSO logic must stay the same
- the seed must stay the same
- the swarm must start from the same initial state
- only the execution strategy should change

That requirement shaped almost every architectural decision in the project.

---

## 2. Core Design Principle

### One PSO core, many execution strategies

The central idea is that `V0`-`V4` are not five different PSO algorithms.

They are five execution paths for the same optimizer.

Shared across versions:

- particle representation
- swarm state
- velocity update equation
- topology
- bounds handling
- stopping criteria
- timing/reporting framework

Different across versions:

- how fitness values are computed
- in `V4`, how the numerically heavy evaluation/update path is executed internally

This separation is what makes the results scientifically defensible.

---

## 3. Repository Architecture

The repository is split by responsibility:

- `core/`: particle state, swarm state, PSO loop
- `options/`: abstract interfaces and strategy hooks
- `parallel/`: concrete evaluator implementations (`V1`-`V4`)
- `objectives/`: benchmark and applied objective functions
- `experiment/`: orchestration for single runs, benchmarks, and EDL
- `utils/`: persistence, metadata, logging, shared method metadata
- `viz/`: convergence plots and report figures
- `scripts/`: CLI entry points
- `tests/`: correctness, compatibility, and reporting tests

This structure was chosen to:

- keep the optimizer core small and testable
- isolate execution-strategy logic
- make reporting reusable
- support realistic workflows beyond toy benchmarks

---

## 4. Main Abstractions

### `FitnessEvaluator`

Defined in `options/evaluator.py`.

It is the most important abstraction in the project.

Baseline contract:

```python
evaluate(positions) -> list[float]
```

Extended contract for `V4`:

```python
step(swarm, bounds_handler, topology, w, c1, c2) -> dict[str, float] | None
```

Why this matters:

- `V0`, `V1`, `V2`, and `V3` only need batched fitness evaluation
- `V4` needs one extra capability: execute a full vectorized PSO step
- the optional `step()` hook allows that without forking the PSO loop into a separate implementation

This is the key design compromise of the `V4` stage: minimal core change, maximal reuse.

### `BoundsPolicy`

Defined in `options/bounds.py`.

Current implementation:

- `ClampBounds`

Behavior:

- clip position to the valid search box
- zero velocity components that hit the wall

Reason for choosing it:

- simple
- stable
- easy to reproduce in both scalar and vectorized code paths

### `Topology`

Defined in `options/topology.py`.

Current implementation:

- `GlobalBestTopology`

Reason for keeping it abstract:

- preserves extensibility
- keeps the optimizer ready for ring/local-best variants
- allows `V4` to explicitly check whether the active topology is compatible with its fast path

---

## 5. PSO Core Decisions

### Shared swarm state

The swarm stores:

- particles
- global best position
- global best fitness

Each particle stores:

- current position
- current velocity
- personal best position
- personal best fitness
- RNG state

This design was kept across all versions to preserve equivalence.

### Re-runnable optimizer instances

`PSO` snapshots its initial swarm state and restores it before each `run()`.

Why:

- repeated runs must be deterministic
- tests need exact restart behavior
- comparisons across strategies require identical initial conditions

### Monotonic global best

The swarm best is monotonic by contract:

- once a better solution is found, the global best is updated
- the recorded best never worsens

This is both an algorithmic property and a reporting guarantee.

---

## 6. Version-by-Version Design

### `V0` Sequential

`V0` uses `SequentialEvaluator`.

Design goal:

- provide the clearest correctness baseline
- minimize machinery and overhead

It is intentionally simple and serves as:

- the functional reference
- the speedup reference

### `V1` Threading

`V1` uses `ThreadPoolEvaluator`.

Design goal:

- compare sequential execution against lightweight concurrency

Why threads were included:

- explicit assignment requirement
- useful comparison point for latency-sensitive workloads
- demonstrates how concurrency can be added without touching PSO logic

Limit:

- still constrained by Python scheduling and often by the GIL

### `V2` Multiprocessing

`V2` uses `ProcessPoolEvaluator`.

Design goal:

- provide a true parallel baseline outside the GIL

Important design details:

- objective must be picklable
- work is batched before dispatch

Why batching exists:

- single-particle process tasks create too much IPC overhead
- medium-size batches amortize serialization cost better

Limit:

- process startup and IPC remain expensive on lightweight objectives

### `V3` Asyncio

`V3` uses `AsyncioEvaluator`.

Design goal:

- support objectives whose evaluations benefit from cooperative concurrency

Why `asyncio` was added:

- not all useful performance problems are CPU-bound
- some workloads are dominated by waiting or synthetic latency
- `latency_mix` was added specifically to exercise this design space

Important property:

- `V3` preserves the same PSO result as `V0`
- it only changes how independent evaluations are scheduled

Limit:

- it is not inherently a numeric acceleration strategy

### `V4` Vectorized

`V4` uses `VectorizedNumpyEvaluator`.

Design goal:

- reduce Python-loop overhead in numerical PSO workloads
- implement vectorized evaluation and update without rewriting the full optimizer

This version is the most design-sensitive stage in the project.

#### Why `V4` was not implemented as a second PSO engine

A full rewrite of the swarm as matrix-native state would have:

- changed too much of the core
- made fairness harder to defend
- increased maintenance cost

Instead, the project keeps the same PSO core and adds one optional evaluator hook.

#### Minimal-core integration strategy

The main loop in `core/pso.py` now does this:

1. ask the evaluator whether it can execute a full optimized step
2. if not, run the classic `V0`/`V1`/`V2`/`V3` path
3. if yes, accept the updated swarm state and continue with history, stopping, and reporting

This means:

- `V0`-`V3` behavior remains unchanged
- `V4` gets the extra power it needs
- the core remains the orchestration layer

#### Fast path and fallback

`V4` uses a vectorized fast path only when all of the following are true:

- the objective is one of the supported numerical benchmarks
- the bounds policy is `ClampBounds`
- the topology is `GlobalBestTopology`

Supported vectorized objectives:

- `sphere`
- `ackley`
- `rastrigin`
- `rosenbrock`

If the objective is unsupported, `V4` falls back safely.

This was a deliberate design decision:

- correctness first
- fast path where it is mathematically clean
- no fake vectorization on incompatible objectives

#### Reproducibility detail

One subtle issue in `V4` was RNG usage.

Although NumPy vectorization encourages batch random generation, the project uses shared deterministic particle RNG states. That means the order of `r1`/`r2` sampling must match the scalar implementation.

So `V4` uses vectorized arithmetic but preserves scalar RNG sampling order where needed.

This was necessary to keep exact seed-based equivalence with `V0`.

---

## 7. Timing and Observability

The project records:

- total time
- evaluation time
- update time
- residual overhead

Why this matters:

- speedup alone is not enough
- the project needs to explain *where* time is going
- `V4` especially changes the balance between evaluation cost and update cost

Examples from the final results:

- on `sphere d=30`, `V4` reduced wall-clock time dramatically because both evaluation and update benefited from vectorization
- on `latency_mix`, `V4` did not help because the bottleneck was latency, not numeric loops

This timing structure makes those interpretations possible.

---

## 8. Persistence and Reporting Design

Each saved run stores:

- structured `summary.json`
- per-iteration CSV history per method
- timing and convergence metrics

The persistence layer had to evolve when `V3` and `V4` were added:

- `MethodResult` now supports unavailable methods cleanly
- new method slots were added to summaries
- reporting tools were made backward-compatible with older summaries

This was important because:

- the schema evolved during development
- old results still needed to remain analyzable

---

## 9. EDL Design Position

The Economic Load Dispatch workflow is intentionally separate from the synthetic benchmark workflow.

Why:

- EDL is an applied constrained optimization problem
- not every benchmark-oriented execution strategy is wired into EDL
- EDL should remain honest about case compatibility and domain constraints

Current EDL comparison scope:

- `V0`
- `V1`
- `V2`

Important applied-design behavior:

- if a case does not include a loss model, variants requiring losses are marked `unavailable`
- this prevents invalid runs from being reported as if they were meaningful

Example:

- `edl_case_40u.json` supports `edl_1` and `edl_2`
- `edl_4` is correctly marked unavailable there because transmission-loss data is absent

This is not a failure of the optimizer; it is a correctness guard in the experiment layer.

---

## 10. Testing Strategy

The test suite now validates:

- seed reproducibility
- bounds enforcement
- monotonic global best
- equivalence of `V0`, `V2`, `V3`, and `V4` on shared objectives
- fallback behavior for unsupported `V4` objectives
- persistence compatibility
- reporting compatibility

Why this matters:

- the project is not only about getting one result
- it is about guaranteeing that new execution strategies do not silently change optimization semantics

`V4` in particular required tests for:

- vectorized fitness equivalence
- PSO trajectory equivalence on supported objectives
- safe fallback on unsupported objectives such as `latency_mix`

---

## 11. Main Design Trade-Offs

### Fairness vs raw specialization

The project chooses fairness:

- one common optimizer
- strategy changes isolated behind interfaces

instead of building separate hand-tuned PSO engines per version.

### Minimal core change vs maximum vectorization purity

For `V4`, the project chooses minimal core change:

- small hook in the evaluator interface
- small branch in the PSO loop

instead of a full matrix-native optimizer rewrite.

This keeps the architecture coherent and easier to defend.

### Honest fallback vs overclaiming support

The project chooses honest fallback:

- vectorize only where cleanly supported
- fall back where not supported

instead of pretending every objective is equally suited to `V4`.

---

## 12. Final Design Rationale

The final architecture supports two strong claims:

1. The repository is maintainable because algorithmic logic, execution strategy, and reporting are cleanly separated.
2. The benchmark conclusions are defensible because the versions differ mainly in execution strategy, not in hidden optimizer behavior.

The most important design result of the repository is therefore not only that it implements PSO correctly, but that it implements **comparable PSO variants correctly**.

That is what makes the `V0`-`V4` experimental story credible.
