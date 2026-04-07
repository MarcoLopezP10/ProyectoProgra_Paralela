# PSO V2 Design Document

## 1. Purpose of the Project

This repository implements a complete and maintainable Particle Swarm Optimization
(PSO) codebase in Python and uses it as an experimental platform to compare
different execution strategies under the same algorithmic conditions.

The implemented delivery scope reaches `V2`:

- `V0`: sequential PSO
- `V1`: threaded fitness evaluation with `ThreadPoolExecutor`
- `V2`: process-based fitness evaluation with `ProcessPoolExecutor` and batching

The project was designed with two goals in mind:

1. implement canonical PSO correctly and reproducibly
2. compare concurrency and parallelism strategies without forking the algorithm

That second goal shaped most of the design decisions in the repository.

## 2. Scope and Evolution

The local Git history shows three clear milestones in the project:

- `f605f60` — first version uploaded around the V1 stage
- `8a1e7f5` — improvements in fitness quality and logging
- `4bad170` — V2 implementation

This evolution matters because the final design is not just the result of a
single clean-room implementation. It reflects several rounds of refinement:

- improving optimization quality instead of chasing only raw speed
- strengthening observability and reproducibility
- keeping a single PSO core while adding new execution strategies
- making the outputs easier to analyze and defend

The final repository intentionally does **not** implement `V3` (`asyncio`) or
`V4` (vectorized NumPy). They remain documented as future work so that the
implemented submission stays coherent with the real scope.

## 3. Design Principles

The project follows a small set of explicit engineering principles.

### 3.1 One PSO core, many evaluators

The main principle is that `V0`, `V1`, and `V2` must share the same PSO loop.
Only the way fitness values are computed is allowed to change.

This avoids the common mistake of accidentally comparing several different PSO
implementations instead of comparing execution strategies on top of one common
algorithm.

### 3.2 Reproducibility first

Every experimental decision was treated as something that should be reproducible:

- explicit seed handling
- persisted configuration
- timing breakdown
- execution metadata
- stable result directory naming

If a result could not be explained or reproduced later, it was considered a weak
result no matter how good it looked.

### 3.3 Measurement with context

Raw total time is not enough to discuss parallelism. The design therefore stores
timing at three levels:

- total time
- fitness evaluation time
- particle update time
- estimated residual overhead

This makes the trade-offs around GIL, IPC, and batching much easier to explain.

### 3.4 Honest reporting over cosmetic reporting

Some outputs, especially around the external PySwarm baseline, are intentionally
conservative:

- PySwarm is treated as an external reference, not as an equivalent internal method
- metrics such as AUC or convergence iteration are left blank when the baseline
  does not expose enough history to compute them honestly

This avoids presenting invented metrics as if they were measured.

## 4. Repository Architecture

The project is split by responsibility:

- `core/`: PSO state and algorithm loop
- `objectives/`: benchmark functions
- `options/`: abstract interfaces and core strategy points
- `parallel/`: V1 and V2 evaluators
- `experiment/`: orchestration and grid search
- `utils/`: logging, persistence, metadata
- `viz/`: convergence plots and swarm animation
- `scripts/`: CLI entry points

### 4.1 Dependency structure

At a high level:

```text
scripts/ ──> experiment/ ──> core/
                           ├── options/
                           ├── parallel/
                           ├── objectives/
                           └── utils/ + viz/
```

The `scripts/` layer contains only command-line orchestration. The experiment
layer builds configurations, launches runs, and persists outputs. The `core/`
layer holds the algorithmic state. Parallel and policy modules are plugged into
the core through abstractions rather than direct imports.

### 4.2 Why this split was chosen

This structure solves several problems at once:

- the PSO loop stays small and testable
- benchmark functions are independent of execution strategy
- new strategies can be added without rewriting the optimizer
- persistence and visualization remain reusable from scripts and tests

It also matches the project requirements closely: `core/`, `objectives/`,
`parallel/`, `experiment/`, `io` functionality through `utils/io.py`, and
`viz/` are all present and clearly separated.

## 5. Core Algorithm Decisions

### 5.1 Canonical PSO update rule

The project uses the standard PSO velocity update with inertia, cognitive, and
social components:

- inertia term `w * velocity`
- cognitive term using the particle best
- social term using the topology best

This keeps the algorithm canonical and easy to compare against textbook PSO.

The decision here was to prefer a standard and well-understood formulation over
more aggressive variants. That makes the behavior easier to test and easier to
defend academically.

### 5.2 Shared swarm state

The swarm stores:

- current particles
- swarm-wide global best position
- swarm-wide global best fitness

Particles store:

- current position
- current velocity
- personal best position
- personal best fitness
- their RNG

One important refinement added later was support for rerunning the same `PSO`
instance from its original initial state. The implementation snapshots the
swarm state before the first run and restores it before each subsequent run.

Why this decision matters:

- it makes repeated runs deterministic
- it avoids silent state carry-over bugs
- it improves testability

Without that restoration logic, calling `run()` twice on the same object would
have produced misleading results because the optimizer would have continued from
the already-optimized swarm.

### 5.3 Monotonic swarm best

The swarm best is designed to be monotonic: once a better fitness is found, it
becomes the new shared reference and the swarm best never worsens.

This was not just a mathematical property to assume. It was treated as a design
contract and explicitly tested because it is central to PSO correctness and to
all convergence plots.

## 6. Strategy Abstractions

The project depends on three main abstractions.

### 6.1 `FitnessEvaluator`

`PSO` does not know whether fitness is computed:

- sequentially
- with threads
- with processes
- in future, asynchronously or vectorized

It only knows that it can call:

```python
evaluate(positions) -> list[float]
```

This is the most important architectural decision in the repository.

Why it was chosen:

- it keeps the mathematical core fixed
- it makes comparisons across `V0/V1/V2` fair
- it prevents strategy-specific logic from leaking into the main loop

### 6.2 `BoundsPolicy`

Boundary handling was separated into its own abstraction because bounds are part
of the optimization problem definition, not just a low-level implementation
detail.

Current implementation:

- `ClampBounds`

Possible future implementations:

- reflective bounds
- penalty-based bounds

Keeping this separate makes the chosen policy explicit and easy to document.

### 6.3 `Topology`

The project currently implements:

- `GlobalBestTopology`

The topology interface was still introduced because neighborhood structure is a
real algorithmic choice, and the project specification mentioned local-best as a
possible extension.

This makes future work such as ring or von Neumann topology possible without
rewriting the PSO loop.

## 7. Boundary-Handling Decision

The explicit choice for this project is clamp bounds with velocity reset on
colliding axes.

### Chosen behavior

When a particle leaves the search box:

- position is clipped back into the valid range
- the velocity component on the colliding axis is set to zero

### Why this was chosen

This strategy was selected because it is:

- simple to reason about
- easy to test
- stable in higher dimensions

### Alternatives considered conceptually

- **Reflect**: more physically intuitive, but can create rebounds and oscillation
  near the walls
- **Penalty**: keeps motion unconstrained, but moves constraint handling into the
  objective layer and complicates interpretation

### Trade-off

Clamp plus velocity reset introduces a mild bias near the borders, but in
practice the cognitive and social terms pull particles back toward relevant
regions of the space quickly enough. For this project, that was a worthwhile
trade for stability and clarity.

## 8. Initialization and Search Stability

Two design choices were especially important for stable behavior.

### 8.1 Reproducible RNG

The project uses `np.random.default_rng(seed)` and passes seeded generators down
to the swarm and particles.

Why this decision was important:

- reproducibility by seed was a requirement
- V0, V1, and V2 had to start from identical swarm states
- tests rely on exact trajectory reproducibility

The PySwarm baseline was also updated to honor the same seed through NumPy and
Python random seeding, so that the baseline comparison would not silently ignore
reproducibility.

### 8.2 Velocity cap via `vmax_ratio`

Initial velocities are not drawn across the full search range. Instead, each
particle uses:

```text
vmax = vmax_ratio * (upper - lower)
```

This was a deliberate improvement over naive full-range initialization.

Why it was chosen:

- full-range velocities produce unstable early jumps
- in higher dimensions, they trigger many boundary corrections immediately
- moderate initial velocities improve early search stability

The repository exposes `vmax_ratio` so this choice stays configurable.

## 9. Stopping Criteria Decision

The project uses a combination of:

- maximum iterations
- tolerance
- patience

The loop tracks improvement in the global best. If the improvement remains below
`tol` for `patience` consecutive iterations, the run stops early.

Why this combination was chosen:

- max iterations alone is wasteful when convergence is already clear
- tolerance alone is too fragile because small numerical oscillations can trigger
  premature stop conditions
- patience makes the stop criterion more robust

This is one of the key decisions that improved runtime discipline without making
the convergence behavior brittle.

## 10. Why the Common Core Matters Experimentally

The strongest requirement for the comparison was:

`V0`, `V1`, and `V2` must behave identically as optimizers for the same seed and
configuration.

The final design enforces that by sharing:

- same swarm initialization
- same seed
- same objective
- same topology
- same bounds policy
- same update equations
- same stopping criteria

Only the evaluator changes.

This is why many of the benchmark tables show the same final fitness and the
same convergence iteration for `V0`, `V1`, and `V2`. That is not a bug or a
coincidence. It is the intended consequence of the architecture.

## 11. Execution-Strategy Decisions

### 11.1 V0 — Sequential baseline

The sequential evaluator is intentionally simple:

- no worker management
- no serialization
- no concurrency overhead

It serves both as the correctness baseline and the time baseline.

This decision is important because all speedups are defined relative to `V0`.

### 11.2 V1 — Threading

`V1` evaluates particle fitness concurrently using `ThreadPoolExecutor`.

Why it was included:

- the project specification required a threading-based version
- it provides a clean concurrency baseline
- it allows discussion of the GIL with real measurements instead of abstract theory

Expected limitation:

- for Python-heavy objective functions, the GIL limits parallel CPU execution

Observed consequence in the project:

- `V1` is sometimes competitive
- often it is slower than `V0`
- it can still be useful when overhead is low or when the objective spends time
  in native code

### 11.3 V2 — Multiprocessing

`V2` evaluates particle fitness using `ProcessPoolExecutor`.

Why it was included:

- to obtain true parallel execution beyond the GIL
- to satisfy the explicit `V2` requirement in the assignment

Main challenge:

- processes require pickling and IPC

This created two important design decisions:

### Objective functions must be picklable

The repository now validates this explicitly and fails early with a clear error
if a non-picklable objective is passed to `V2`.

Why this was added:

- without it, multiprocessing errors appeared late and opaquely
- early failure is much easier to debug and explain

### Batching

Instead of sending one particle per process task, `V2` groups particles into
batches.

Why this was chosen:

- one-particle tasks create too many small IPC exchanges
- medium-size batches amortize serialization cost better
- order can still be preserved by flattening batch results

Trade-off:

- very small batches waste overhead
- very large batches reduce load balancing

The final implementation uses automatic batch sizing by default, with optional
manual override from the CLI.

## 12. Hyperparameter Strategy

The project supports two modes:

- fixed user-configured hyperparameters
- objective-aware profile defaults with optional grid search

### 12.1 Why profiles were introduced

Using one single default tuple for every objective and dimension produced weak
behavior on difficult functions and higher dimensions. The project therefore
evolved toward conservative objective-aware profiles.

Each profile can define:

- `w`
- `c1`
- `c2`
- swarm size
- max iterations
- patience
- tolerance
- `vmax_ratio`
- quick seeds for light grid search

This was a practical compromise between fully manual tuning and hard-coded
one-size-fits-all defaults.

### 12.2 Grid search design

The grid search module explores:

- `w`
- `c1`
- `c2`
- swarm size

It supports selection by:

- final fitness
- AUC
- convergence iteration
- time

Why grid search is part of the design and not an add-on:

- some functions, especially Rastrigin, are highly sensitive to hyperparameters
- the project needed a principled way to compare search-quality trade-offs
- the assignment explicitly required configurable grid search

The light grid search used in `run_single.py` was also a design decision: it
gives better defaults without turning every single run into a full experimental
campaign.

## 13. Baseline Decision: PySwarm

PySwarm was included as an external baseline for reference, but the project does
not treat it as an internal method equivalent to `V0`, `V1`, or `V2`.

Why this distinction matters:

- PySwarm has its own implementation details
- its internal history is not available in the same form
- direct timing comparisons are useful, but not every metric is equally comparable

Two explicit choices follow from that:

1. PySwarm is included in result tables and speedup plots
2. metrics requiring per-iteration history are not fabricated when unavailable

This keeps the reporting more honest and avoids overstating comparability.

## 14. Persistence and Output Structure

The on-disk structure follows:

```text
results/runs/<objective>_d<dim>_s<seed>/
    summary.json
    history_v0.csv
    history_v1.csv
    history_v2.csv
```

### Why this layout was chosen

- objective, dimension, and seed are visible directly in the path
- multiple runs can coexist without overwriting each other
- the directory is easy to parse programmatically
- the structure is human-readable for manual inspection

### Format choices

- `JSON` for run summaries and configuration
- `CSV` for per-iteration histories

Why:

- JSON is convenient for nested structured metadata
- CSV is lightweight and easy to read for time series

This combination was simpler and more transparent than introducing a database or
binary storage format for the scope of this project.

## 15. Logging and Observability

The logger writes structured lines containing:

- timestamp
- level
- objective context
- method context
- event payload

Why this was chosen:

- experiments need traceability
- per-iteration logging helps diagnose stagnation and timing behavior
- method-specific context makes mixed run logs readable

The decision to log to file instead of cluttering the terminal also improved the
usability of the CLI scripts during long benchmark runs.

## 16. Visualization Decisions

The project includes two visualization layers.

### 16.1 Convergence plots

Convergence plots compare:

- `V0`
- `V1`
- `V2`
- a horizontal reference line for PySwarm final fitness

Why the baseline is a horizontal line:

- it is a real scalar result
- it is not a real internal convergence curve

This was made explicit in the labeling to avoid misreading the baseline as if it
were derived from the same iteration history as the internal runs.

### 16.2 Swarm animation

The animation module supports:

- `d=2`: contour background plus swarm motion
- `d=3`: particle motion plus best-point tracking

Why not a full 3D objective surface for `d=3`:

- once the decision space itself is three-dimensional, a full objective surface
  is not generally meaningful in the same way as a `d=2` contour plot

The recorder callback was also a deliberate design choice: it records the actual
optimizer trajectory without duplicating or shadowing the PSO loop.

## 17. Analysis Pipeline Decisions

The analysis step was later strengthened in several ways.

### 17.1 Backward-compatible summary loading

The loader accepts older summaries and backfills missing fields from history CSVs
where possible.

Why this mattered:

- the result format evolved during development
- older runs still had value for analysis
- strict schema rejection would have thrown away useful data

### 17.2 More honest summary metrics

For methods with no true history, especially the external baseline:

- AUC remains empty
- convergence iteration remains empty

This avoids fake completeness in the summary tables.

### 17.3 Better boxplots

The boxplot logic was refined because some early plots looked empty or
misleading.

Final design choices:

- overlay individual points on top of each boxplot
- show `n` in the x-axis labels
- switch to `symlog` when methods differ by several orders of magnitude

Why:

- with only two seeds, a boxplot alone can collapse visually
- baseline values can dwarf the internal methods on linear scale
- the plot should help interpretation, not hide it

## 18. Testing Decisions

The test suite is not just smoke testing. It encodes several project-level
guarantees:

- reproducibility by seed
- bounds enforcement
- monotonic global best
- convergence on Sphere
- equivalence between sequential and process evaluation for the same seed
- rejection of non-picklable multiprocessing objectives
- result-format and analysis-loader behavior

Why this test profile was chosen:

- it covers both algorithmic correctness and engineering contracts
- it verifies the fairness claim behind `V0/V1/V2`
- it catches regressions in persistence and reporting, not only in optimization

## 19. Known Limitations

The final design is strong for the required scope, but several limits remain.

- `V1` is still constrained by the GIL for Python-heavy objectives
- `V2` only helps when evaluation cost is large enough to amortize IPC overhead
- topology support is currently global-best only
- no vectorized `V4` implementation is present yet
- no `asyncio` `V3` path is implemented
- only two seeds are used in the current benchmark examples, so some plots are
  informative but statistically light

These are acceptable limitations for the declared delivery scope, but they are
important to mention in a serious design document.

## 20. Main Trade-offs and Final Rationale

The final design reflects a consistent set of trade-offs:

- simplicity over premature complexity in the core loop
- explicit abstractions over tightly coupled implementations
- reproducibility over ad-hoc experimentation
- honest metrics over cosmetically complete tables
- shared algorithmic behavior over strategy-specific forks

The most important design outcome is this:

the repository is not just a PSO implementation with some parallel code bolted
on. It is a controlled experimental framework where algorithmic behavior,
engineering quality, and result interpretation are aligned.

That is what makes the project maintainable and also what makes its benchmark
conclusions defensible.
