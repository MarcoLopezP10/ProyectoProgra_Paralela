# AUTHOR - MARCO LOPEZ PRIETO
# PSO — Particle Swarm Optimization

A complete, maintainable Python implementation of Particle Swarm Optimization (PSO) following software engineering best practices, used as a testbed for comparing different parallel and concurrent evaluation strategies.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Module Architecture](#3-module-architecture)
4. [Installation](#4-installation)
5. [Usage & Commands](#5-usage--commands)
6. [Parallelism Strategies](#6-parallelism-strategies)
7. [Experimental Results](#7-experimental-results)
8. [Design Decisions & Trade-offs](#8-design-decisions--trade-offs)
9. [Reproducibility](#9-reproducibility)

---

## 1. Project Overview

This project implements the canonical PSO algorithm for minimising continuous functions in R^d, and uses it to compare different fitness evaluation strategies:

| Version | Strategy | Description |
|---|---|---|
| V0 | Sequential | Baseline — one particle at a time |
| V1 | Threading | `ThreadPoolExecutor` — concurrent evaluation |
| V2 | Multiprocessing | `ProcessPoolExecutor` — true parallelism with batched evaluation |
| V3 | Asyncio | Cooperative concurrency for I/O-bound evaluation *(coming)* |
| V4 | NumPy vectorised | Implicit parallelism via matrix operations *(coming)* |

The core PSO algorithm never changes between versions. Only the fitness evaluator is swapped, which is possible because of the `FitnessEvaluator` abstraction.

---

## 2. Project Structure

```
PRACTICA-2.2/
├── core/                       # PSO algorithm
│   ├── particle.py             # Single particle (position, velocity, personal best)
│   ├── swarm.py                # Collection of particles + global best
│   └── pso.py                  # Main loop, timing instrumentation, logging
│
├── options/                    # Strategy abstractions
│   ├── bounds.py               # BoundsPolicy (ABC) + ClampBounds
│   ├── evaluator.py            # FitnessEvaluator (ABC) + SequentialEvaluator
│   └── topology.py             # Topology (ABC) + GlobalBestTopology
│
├── parallel/                   # Parallel/concurrent evaluators
│   └── evaluator.py            # ThreadPoolEvaluator (V1) + ProcessPoolEvaluator (V2)
│
├── objectives/                 # Benchmark functions
│   ├── sphere.py               # Sphere — unimodal, convex
│   ├── ackley.py               # Ackley — multimodal, origin trap
│   ├── rosenbrock.py           # Rosenbrock — narrow curved valley
│   └── rastrigin.py            # Rastrigin — highly multimodal
│
├── experiment/                 # Experiment orchestration
│   ├── grid_search.py          # Multi-seed grid search
│   └── run_single.py           # Single experiment runner (V0 + V1 + V2 + baseline)
│
├── baseline/                   # External reference
│   └── pswarm.py               # PySwarm library wrapper
│
├── viz/                        # Visualisation
│   ├── convergence.py          # Convergence curves (fitness vs iteration)
│   └── swarm_animation.py      # Swarm evolution animation for d=2 (GIF/MP4)
│
├── utils/                      # Cross-cutting utilities
│   ├── io.py                   # Structured persistence (JSON + CSV)
│   └── logger.py               # Structured logging with context
│
├── scripts/                    # Entry-point scripts
│   ├── run_pso.py              # Single reproducible run
│   ├── run_benchmarks.py       # Full benchmark suite
│   ├── run_grid_search.py      # Hyperparameter grid search
│   └── make_viz.py             # Swarm animations
│
├── tests/
│   └── test_pso.py             # Unit tests
│
├── results/                    # Auto-generated experiment results
└── logs/                       # Auto-generated logs and convergence plots
```

---

## 3. Module Architecture

### Dependency diagram

```
scripts/run_pso.py
scripts/run_benchmarks.py  ──► experiment/run_single.py ──► core/pso.py
scripts/run_grid_search.py ──► experiment/grid_search.py     │
scripts/make_viz.py ─────────────────────────────────────►  core/swarm.py
                                                              │
                                                ┌────────────┼────────────┐
                                                ▼            ▼            ▼
                                          options/      options/     options/
                                          evaluator     bounds       topology
                                              │
                                   ┌──────────┴──────────┐
                                   ▼                     ▼
                             SequentialEvaluator   ThreadPoolEvaluator / ProcessPoolEvaluator
                             (V0 — options/)       (V1 / V2 — parallel/)
                                   │
                                   ▼
                             objectives/
                             sphere, ackley, rosenbrock, rastrigin
```

### Key abstractions

**`FitnessEvaluator` (ABC)** — defined in `options/evaluator.py`. Single method: `evaluate(positions) -> List[float]`. PSO calls this once per iteration without knowing whether evaluation is sequential, threaded, process-based, or vectorised. This is what makes swapping V0/V1/V2/… completely transparent.

**`BoundsPolicy` (ABC)** — defined in `options/bounds.py`. Single method: `apply(position, velocity) -> (position, velocity)`. PSO applies this after every position update. Currently implemented by `ClampBounds`. `ReflectBounds` or `PenaltyBounds` can be added without modifying the core.

**`Topology` (ABC)** — defined in `options/topology.py`. Single method: `get_best_position(particle, swarm) -> ndarray`. Determines which position each particle uses as its social reference. Currently `GlobalBestTopology`. Ring or von Neumann topologies can be plugged in.

### How the PSO loop uses the abstractions

```python
# Every iteration of PSO.run():

positions = self.swarm.get_positions()

# 1. Evaluate fitness — delegated entirely to the evaluator
fitness = self.evaluator.evaluate(positions)      # FitnessEvaluator

# 2. Update personal and global bests
self.swarm.update_global_best(positions, fitness)

# 3. Move particles
for p in self.swarm.particles:
    best_pos = self.topology.get_best_position(p, self.swarm)  # Topology
    p.update_velocity(best_pos, self.w, self.c1, self.c2)
    p.update_position()
    p.position, p.velocity = self.bounds_handler.apply(        # BoundsPolicy
        p.position, p.velocity
    )
```

`PSO` never imports `SequentialEvaluator`, `ThreadPoolEvaluator`, `ClampBounds`, or `GlobalBestTopology` directly. It only knows the abstract interfaces.

### Why this architecture is useful

This structure was chosen to keep the implementation maintainable and to make
version-to-version comparisons fair:

- The PSO core loop is isolated in `core/pso.py`, so algorithmic behaviour does
  not change when switching from V0 to V1.
- Evaluation strategy is encapsulated behind `FitnessEvaluator`, which makes it
  possible to compare sequential and threaded execution under the same search
  conditions.
- Boundary handling is isolated in `BoundsPolicy`, which keeps the decision
  about how to deal with infeasible particles explicit and easy to document.
- Topology is isolated in `Topology`, which keeps the implementation extensible
  even though this project currently uses only the global-best variant.

This separation is important for software engineering quality, but also for
experimental validity: if V0, V1, and V2 use the same seed, same swarm size, same
coefficients, same bounds and same topology, then any observed difference can
be attributed to the evaluation strategy rather than to hidden algorithmic
changes.

---

## 4. Installation

```bash
# Clone the repository
git clone <repo-url>
cd PRACTICA-2.2

# Install dependencies (no virtual environment required)
pip install numpy matplotlib prettytable pyswarm pytest pillow
```

Python 3.10+ recommended.

If you want to run the test suite and all optional scripts, install every listed
dependency. Some modules such as `prettytable`, `matplotlib`, `pyswarm`,
`pillow`, or `pytest` are used only by specific parts of the project.

---

## 5. Usage & Commands

### Quick reference table

| Goal | Command |
|---|---|
| Single run, all functions, d=2 | `python -m scripts.run_pso` |
| Single function | `python -m scripts.run_pso --objective sphere` |
| Multiple dimensions and seeds | `python -m scripts.run_pso --dim 2 10 30 --seed 42 7` |
| With hyperparameter grid search | `python -m scripts.run_pso --dim 2 --grid-search` |
| Dry run (no files saved) | `python -m scripts.run_pso --no-save` |
| Full benchmark suite | `python -m scripts.run_benchmarks` |
| Benchmark custom dims/seeds | `python -m scripts.run_benchmarks --dims 2 10 30 --seeds 42 7 123` |
| Grid search (3×3×3, 5 seeds) | `python -m scripts.run_grid_search` |
| Grid search, single function | `python -m scripts.run_grid_search --objective sphere --dims 2` |
| Swarm animations (GIF) | `python -m scripts.make_viz` |
| Single animation | `python -m scripts.make_viz --objective sphere --seed 42` |
| Unit tests | `pytest tests/test_pso.py -v` |
| Help for any script | `python -m scripts.run_pso --help` |

### CLI arguments for `run_pso.py`

| Argument | Default | Description |
|---|---|---|
| `--objective` | all 4 | Function(s): `sphere` `ackley` `rosenbrock` `rastrigin` |
| `--dim` | `2` | Dimension(s), e.g. `--dim 2 10 30` |
| `--seed` | `42` | Seed(s), e.g. `--seed 42 7 123` |
| `--bounds-lo` | `-5.0` | Lower bound (all dimensions) |
| `--bounds-hi` | `5.0` | Upper bound (all dimensions) |
| `--w` | `0.7` | Inertia weight |
| `--c1` | `1.5` | Cognitive coefficient |
| `--c2` | `1.5` | Social coefficient |
| `--n-particles` | `80` | Swarm size |
| `--max-iters` | `500` | Maximum iterations |
| `--patience` | `30` | Early-stop patience |
| `--vmax-ratio` | auto | Velocity cap as a fraction of the search range |
| `--grid-search` | off | Enable hyperparameter grid search |
| `--workers` | auto | Max threads for V1 ThreadPoolExecutor |
| `--process-workers` | auto | Max processes for V2 ProcessPoolExecutor |
| `--batch-size` | auto | Particles per process task in V2 |
| `--no-save` | off | Skip saving files to disk |

If `--w`, `--c1`, `--c2`, `--n-particles`, `--max-iters`, `--patience`,
`--tol`, or `--vmax-ratio` are omitted, the project can resolve conservative
defaults automatically based on the objective function and dimension.

### CLI arguments for `run_benchmarks.py`

| Argument | Default | Description |
|---|---|---|
| `--objective` | all 4 | Function(s) to benchmark |
| `--dims` | `2 10 30` | Dimensions to evaluate |
| `--seeds` | `42 7` | Seeds to test |
| `--bounds-lo` | `-5.0` | Lower bound |
| `--bounds-hi` | `5.0` | Upper bound |
| `--n-particles` | auto | Swarm size override |
| `--max-iters` | auto | Iteration budget override |
| `--tol` | auto | Convergence tolerance override |
| `--patience` | auto | Early-stop patience override |
| `--vmax-ratio` | auto | Velocity cap override |
| `--workers` | auto | Max threads for V1 |
| `--process-workers` | auto | Max processes for V2 |
| `--batch-size` | auto | Particles per process task in V2 |
| `--grid-search` | off | Tune hyperparameters before each run |
| `--summary-csv` | `results/benchmark_summary.csv` | Aggregated CSV output |

### CLI arguments for `run_grid_search.py`

| Argument | Default | Description |
|---|---|---|
| `--objective` | all 4 | Function(s) to optimise during search |
| `--dims` | `2 10 30` | Dimensions to search |
| `--seeds` | `0 1 7 42 123` | Seeds averaged per combination |
| `--w` | auto | Inertia values to test |
| `--c1` | auto | Cognitive values to test |
| `--c2` | auto | Social values to test |
| `--n-particles` | auto | Swarm sizes to test |
| `--max-iters` | `150` | Iteration budget per combination |
| `--tol` | `1e-8` | Tolerance used inside search runs |
| `--patience` | `40` | Early-stop patience inside search runs |
| `--vmax-ratio` | auto | Velocity cap override |
| `--out-dir` | `results/grid_search` | CSV output directory |
| `--verbose` | off | Log every combination/seed |

When no explicit search grid is provided, the script uses a dimension-aware
search space chosen for each objective function.

### CLI arguments for `make_viz.py`

| Argument | Default | Description |
|---|---|---|
| `--objective` | all 4 | Function(s) to animate |
| `--seed` | `42` | Random seed |
| `--n-particles` | `40` | Swarm size for the animation run |
| `--max-iters` | `150` | Iteration budget |
| `--bounds-lo` | `-5.0` | Lower bound |
| `--bounds-hi` | `5.0` | Upper bound |
| `--w` | `0.7` | Inertia weight |
| `--c1` | `1.5` | Cognitive coefficient |
| `--c2` | `1.5` | Social coefficient |
| `--vmax-ratio` | `0.2` | Velocity cap as a fraction of range |
| `--fps` | `6` | Frames per second |
| `--format` | `gif` | Output format (`gif` or `mp4`) |
| `--resolution` | `120` | Contour-grid resolution |
| `--max-frames` | none | Optional frame cap |

### Recommended way to compare V0, V1, and V2

For a fair comparison between versions:

1. Use the same objective function, dimension, seed, bounds and hyperparameters.
2. Keep the topology and bounds policy fixed.
3. Compare both solution quality (`best_fitness`) and runtime (`time_s`).
4. Use multiple seeds when drawing conclusions about performance.

This project is explicitly designed so that V0, V1, and V2 share the same PSO
logic and differ only in the fitness evaluation strategy.

---

## 6. Parallelism Strategies

The PSO core loop is identical across all versions. Only the evaluator changes.

### V0 — Sequential (baseline)

```python
from options.evaluator import SequentialEvaluator
evaluator = SequentialEvaluator(sphere)
```

Simple Python list comprehension. One particle evaluated at a time. This is the baseline against which all other versions are measured.

### V1 — Threading (`ThreadPoolExecutor`)

V1 keeps the same swarm dynamics, coefficients, stopping criteria and bounds
policy as V0. The only change is that particle fitness values are evaluated via
`ThreadPoolExecutor`.

Important trade-off:

- For small NumPy-based objective functions, V1 may be slower than V0 because
  thread scheduling overhead and the Python GIL can dominate the runtime.
- For heavier or more latency-dominated objective functions, the same design can
  still be useful as a clean concurrent baseline.

The project therefore uses V1 as a concurrency comparison point, not as a
guaranteed speedup.

### V2 — Multiprocessing (`ProcessPoolExecutor`)

V2 also preserves exactly the same PSO dynamics as V0. The difference is that
fitness evaluation is delegated to separate OS processes, which bypass the GIL
and provide true CPU parallelism.

Important trade-offs:

- Spawning processes and sending particle positions across process boundaries
  introduces pickling and inter-process communication (IPC) overhead.
- For that reason, V2 uses batching: each submitted task evaluates a block of
  particles instead of a single particle.
- Batching reduces the number of IPC operations and usually improves V2
  compared to a one-particle-per-task design.

The project therefore uses V2 as the true-parallel baseline for CPU-bound
evaluation, while explicitly measuring whether that extra machinery pays off.

**When V2 is more likely to help:**
- CPU-bound objective functions whose per-particle evaluation is expensive
- Larger swarms and higher dimensions, where each submitted task does enough work
- Configurations where batching is tuned well enough to amortise IPC overhead

**When V2 may still lose against V0:**
- Very cheap objective functions such as small benchmark kernels
- Small swarms or very short runs
- Cases where the sequential particle-update step remains the dominant cost

---

## 6.1 Design Decisions And Trade-Offs

### Boundary strategy

The project uses `ClampBounds` as the explicit bounds-enforcement strategy.
Whenever a particle exits the box constraints, its position is clipped back into
the search space and the corresponding velocity component is reset to zero.

Why this was chosen:

- It is simple and predictable.
- It avoids unstable bouncing near the boundary.
- It works reliably across all tested dimensions.

Trade-off:

- It can introduce a mild bias near the borders of the search space.
- Alternative strategies such as reflection or penalty could be explored in
  future versions.

### Topology choice

The current implementation uses `GlobalBestTopology`.

Why this was chosen:

- It is the canonical PSO variant.
- It converges quickly on many standard benchmark functions.
- It keeps the implementation simple for the first stage of the project.

Trade-off:

- It may increase the risk of premature convergence on highly multimodal
  landscapes compared to local-best topologies.

### Hyperparameter strategy

The project supports both fixed hyperparameters and grid search. The quick
in-run tuner uses a smaller search space, while the dedicated grid-search script
 can be used for broader offline exploration.

Trade-off:

- Better hyperparameter selection usually improves final fitness.
- It also increases total experimental time, especially in higher dimensions.

### Concurrency strategy

V1 uses threads instead of changing the PSO logic itself.

Why this was chosen:

- It preserves fairness of comparison with V0.
- It demonstrates a clean separation between algorithm and evaluation policy.
- It keeps the system extensible for future versions.

Trade-off:

- It improves code extensibility more reliably than raw runtime in lightweight
  numerical benchmarks.

```python
from parallel.evaluator import ThreadPoolEvaluator
evaluator = ThreadPoolEvaluator(sphere, max_workers=4)
```

Distributes fitness evaluation across multiple threads. In theory multiple particles could be evaluated simultaneously. In practice, CPython's **GIL (Global Interpreter Lock)** prevents true parallel execution of Python bytecode — only one thread runs Python code at a time.

**Why V1 is slower than V0 for these benchmarks:**

Each benchmark function (Sphere, Ackley, Rosenbrock, Rastrigin) is a fast NumPy operation taking ~0.7ms per particle. Creating and managing a thread costs ~0.5–1ms in Python. The overhead exceeds the work:

```
V0: evaluate p1 → p2 → ... → p80     (direct, no overhead)
V1: create pool → dispatch 80 tasks → threads fight for GIL
    → collect results → destroy pool  (overhead > actual work)
```

This is confirmed by the timing breakdown: in V0, evaluation takes ~14–40% of total time. In V1 it jumps to ~65–72% — the ThreadPoolExecutor overhead is charged to the evaluation timer.

**When V1 would actually help:**
- I/O-bound evaluation (API calls, file reads, database queries) — the GIL is released during I/O, so threads can genuinely overlap
- Very expensive per-particle computation that releases the GIL (large NumPy operations on big arrays)

### V3 — Asyncio *(coming)*

Cooperative concurrency. Only makes sense when evaluation is I/O-bound — for example, each particle queries a local service with variable latency. Uses `asyncio.gather()` to overlap waiting times.

### V4 — NumPy vectorised *(coming)*

Eliminates all Python loops. Positions, velocities, and fitness evaluation operate on full matrices `(n_particles × dim)`. Expected to be the fastest version for pure mathematical objectives.

---

## 7. Experimental Results

The key evaluation criteria in this project are:

- Best fitness reached by each method.
- Number of iterations executed.
- Total runtime.
- Internal timing breakdown (`eval`, `update`, and residual overhead).

When interpreting results, lower fitness is better. Runtime should always be
interpreted together with solution quality: a faster method is not necessarily
better if it converges to a clearly worse solution.

All experiments run on Apple MacBook Air M2, macOS, Python 3.11/3.12.
Default hyperparameters: `w=0.7, c1=1.5, c2=1.5, n_particles=80, max_iters=500, seed=42`.

### Convergence quality

| Function | d | V0 fitness | V1 fitness | PySwarm fitness | Winner |
|---|---|---|---|---|---|
| Sphere | 2 | 4.20e-13 | 4.20e-13 | 3.67e-08 | V0 |
| Sphere | 10 | 2.63e-12 | 2.63e-12 | 2.70e-07 | V0 |
| Sphere | 30 | 1.16e-10 | 1.16e-10 | 1.03e-06 | V0 |
| Ackley | 2 | 4.48e-13 | 4.48e-13 | 1.56e-08 | V0 |
| Ackley | 10 | 6.03e-11 | 6.03e-11 | 1.69e-06 | V0 |
| Ackley | 30 | 9.31e-01 | 9.31e-01 | 7.88e-06 | PySwarm |
| Rosenbrock | 2 | 2.58e-12 | 2.58e-12 | 8.11e-10 | V0 |
| Rosenbrock | 10 | 1.19e+00 | 1.19e+00 | 1.84e+00 | V0 |
| Rosenbrock | 30 | 2.70e+01 | 2.70e+01 | 2.36e+01 | PySwarm |
| Rastrigin | 2 | 2.13e-14 | 2.13e-14 | 1.17e-07 | V0 |
| Rastrigin | 10 | 5.97e+00 | 5.97e+00 | 3.98e+00 | PySwarm |
| Rastrigin | 30 | 8.46e+01 | 8.46e+01 | 8.07e+01 | PySwarm |

**Key observation:** V0, V1, and V2 reach exactly the same fitness whenever the
same seed and hyperparameters are used. The evaluation strategy does not affect
solution quality — only execution time. This confirms the abstraction works
correctly: swapping the evaluator changes nothing algorithmically.

PySwarm wins on multimodal functions at high dimension (Ackley d=30, Rastrigin d=10/30, Rosenbrock d=30) because our PSO uses fixed hyperparameters not tuned for those cases. With grid search the gap closes significantly.

### Timing comparison (V0 vs V1 baseline study)

| Function | d | V0 time (s) | V1 time (s) | V1 speedup | V0 % eval | V1 % eval |
|---|---|---|---|---|---|---|
| Sphere | 2 | 0.41 | 0.98 | 0.42x | 13.7% | 67.0% |
| Sphere | 10 | 0.74 | 1.77 | 0.42x | 17.4% | 64.6% |
| Sphere | 30 | 1.63 | 4.61 | 0.35x | 16.0% | 64.3% |
| Ackley | 2 | 1.02 | 3.28 | 0.31x | 38.5% | 70.5% |
| Ackley | 10 | 2.27 | 4.28 | 0.53x | 40.6% | 72.1% |
| Ackley | 30 | 3.30 | 4.98 | 0.66x | 39.8% | 68.6% |
| Rosenbrock | 2 | 0.76 | 2.24 | 0.34x | 35.9% | 68.0% |
| Rosenbrock | 10 | 3.20 | 5.25 | 0.61x | 35.0% | 69.0% |
| Rosenbrock | 30 | 3.86 | 5.20 | 0.74x | 33.9% | 69.2% |
| Rastrigin | 2 | 0.51 | 0.99 | 0.52x | 31.4% | 66.9% |
| Rastrigin | 10 | 2.67 | 3.28 | 0.81x | 30.8% | 67.3% |
| Rastrigin | 30 | 3.40 | 4.21 | 0.81x | 37.1% | 66.6% |

**Key observations:**

**1. V1 is always slower than V0.** Speedup ranges from 0.31x to 0.81x. Threading adds overhead without benefit for CPU-bound small tasks. The GIL prevents true parallel execution.

**2. The `% eval` column explains why.** In V0 evaluation takes 14–40% of total time. In V1 it jumps to 65–72%. The ThreadPoolExecutor overhead inflates the evaluation timer 3–5×, making it the dominant cost where it previously was not.

**3. The gap narrows with dimension.** At d=2 V1 is ~0.31–0.52x. At d=30 it improves to ~0.66–0.81x. As dimension grows each particle evaluation becomes more expensive (more NumPy operations), so the fixed thread overhead becomes proportionally smaller.

**4. V0 `% update` dominates (~60–80%).** Particle position and velocity updates are the main bottleneck in the sequential version — not fitness evaluation. This suggests V4 (vectorised updates) could yield the biggest speedup.

### V2 validation and batching results

After implementing V2 with `ProcessPoolExecutor`, the first validation step was
to check correctness rather than speed. In every manual test performed, V0, V1,
and V2 reached exactly the same `best_fitness` with the same seed and
hyperparameters. This is the expected behaviour: V2 changes how the fitness is
computed, not the PSO algorithm itself.

Representative runs:

| Function | d | V0 fitness | V1 fitness | V2 fitness | V0 time (s) | V1 time (s) | V2 time (s) |
|---|---|---|---|---|---|---|---|
| Sphere | 2 | 7.63e-05 | 7.63e-05 | 7.63e-05 | 0.015 | 0.033 | 1.015 |
| Sphere | 30 | 4.97e-12 | 4.97e-12 | 4.97e-12 | 2.319 | 3.399 | 5.446 |
| Rastrigin | 30 | 3.38e+01 | 3.38e+01 | 3.38e+01 | 5.642 | 8.986 | 13.754 |
| Ackley | 30 | 8.42e-03 | 8.42e-03 | 8.42e-03 | 8.123 | 13.649 | 9.390 |

These runs show two useful conclusions:

1. V2 is correct: it preserves the exact same optimisation result as V0.
2. V2 is not automatically faster: for lightweight or moderately sized
   objective functions, process startup, pickling, and IPC can dominate the
   runtime.

To reduce that overhead, V2 uses batching. The following measurements were made
on `Ackley`, `d=30`, `n_particles=160`, `max_iters=400`, `process_workers=4`:

| Batch size | V2 time (s) | V2 speedup vs V0 |
|---|---|---|
| 8 | 10.586 | 0.569x |
| 16 | 9.146 | 0.660x |
| 32 | 7.261 | 0.893x |
| 64 | 7.835 | 0.753x |

This behaviour is exactly what the theory predicts:

- Small batches generate too many process tasks and too much IPC.
- Medium batches amortise overhead better.
- Very large batches start to reduce load balancing and can lose some benefit.

For the tested configuration, `batch_size=32` was the best V2 setting. Even
there, V2 did not outperform V0, but it clearly improved over worse
multiprocessing configurations. This validates batching as a real optimisation,
even when it is not enough to beat the sequential baseline.

### With grid search (d=2, optimised hyperparameters)

| Function | Default fitness | Grid search fitness | Hyperparams found |
|---|---|---|---|
| Sphere | 4.20e-13 | 2.31e-20 | w=0.4, c1=1.8, c2=1.2, n=60 |
| Ackley | 4.48e-13 | 4.44e-16 | w=0.4, c1=1.2, c2=1.2, n=90 |
| Rosenbrock | 2.58e-12 | 1.28e-16 | w=0.6, c1=1.2, c2=1.5, n=100 |
| Rastrigin | 2.13e-14 | 0.0 (exact) | w=0.5, c1=1.0, c2=1.4, n=80 |

Grid search: `w ∈ {0.4,0.6,0.8}`, `c1 ∈ {1.2,1.5,1.8}`, `c2 ∈ {1.2,1.5,1.8}`, 5 seeds per combination.

---

## 8. Design Decisions & Trade-offs

The most important engineering and experimental trade-offs of the project are:

- Strong modularity vs slightly more boilerplate: abstract interfaces make the
  code easier to extend and compare, but require more structure than a single
  monolithic script.
- Better fitness vs longer execution time: larger swarms, longer patience, and
  more robust grid search often improve the solution but increase cost.
- Concurrency vs actual speedup: V1 improves modularity and demonstrates a
  concurrent strategy, but does not necessarily outperform V0 on small
  CPU-bound benchmark functions.
- Multiprocessing vs IPC overhead: V2 bypasses the GIL and enables true CPU
  parallelism, but process startup, pickling, and inter-process communication
  can offset the gains.
- Simplicity vs feature coverage: this stage focuses on a solid V0/V1/V2
  implementation instead of prematurely adding many incomplete variants.

### Current limitations

This repository intentionally focuses on the first stage of the project. The
main current limitations are:

- V3 (`asyncio`) and V4 (vectorised NumPy) are still future work.
- The topology currently available is global-best only.
- Configuration is currently handled through CLI arguments rather than external
  YAML/JSON configuration files.
- The external baseline depends on `pyswarm` being installed.
- Threading and multiprocessing are primarily included as architectural and
  experimental comparisons, not as guaranteed performance improvements on all
  workloads.
- V2 only parallelises the fitness evaluation phase. Particle updates remain
  sequential and are still a major bottleneck in many runs.

These limitations are acceptable for the current stage because the project
already provides a complete, testable, instrumented and comparable PSO system.

### Bounds strategy: clamp + velocity zeroing

When a particle exits the search box its position is clipped to the boundary and the velocity component on the hit axis is set to zero.

Clamp was chosen over reflect because reflection can cause oscillation near boundaries in high dimensions. Zeroing the velocity prevents the particle from immediately trying to leave again in the next iteration — the cognitive/social terms then redirect it back into the interior.

**Trade-off:** zeroing introduces a mild bias toward the walls, but this is negligible in practice since the social attraction to the global best quickly dominates.

**Extensibility:** `ClampBounds` inherits from `BoundsPolicy` (ABC). Any alternative strategy can be added without modifying `PSO`.

### Topology: global best

Every particle knows the single best position found by the entire swarm.

Chosen for simplicity and fast convergence on unimodal functions. The trade-off is susceptibility to premature convergence on multimodal functions (Rastrigin, Ackley at d=30), which explains why PySwarm outperforms on those cases with fixed hyperparameters.

**Extensibility:** `Topology` ABC allows adding ring or von Neumann topologies without touching `PSO`.

### Velocity initialisation: `vmax = 0.2 × search_range`

Initial velocities sampled from `[-0.2×range, 0.2×range]` rather than the full range.

Full-range initialisation causes large initial jumps especially at d=30, where particles can overshoot the search box on the first iteration and trigger many bounds corrections. Moderate initialisation produces more stable early convergence.

### Persistence format: JSON + CSV

JSON for configuration and scalar metrics — human-readable, structured, easy to load with `json.load()` or `pandas.read_json()`, handles nested structures like `TimingBreakdown` naturally.

CSV for per-iteration convergence curves — compact, trivially loadable with `pandas.read_csv()`, directly plottable without parsing.

Result directories named `results/<function>_d<dim>_s<seed>/` so multiple experiments coexist without overwriting each other.

### Logging: file only

The logger writes exclusively to `logs/pso_summary.log`. Console output is handled separately by the PrettyTable summary. This keeps both readable and independent. The `ContextFormatter` adds `objective` and `method` to every line, enabling easy filtering:

```
2026-03-18 10:34:08 - --SPHERE-- - V0 - INFO - iter=10 | best=1.23e-05 | t_eval=0.42ms
```

The `method` field distinguishes at least `RUN`, `V0`, `V1`, and `V2`, which
allows the same experiment to be traced separately for sequential, threaded,
and multiprocessing evaluation.

### Early stopping: tolerance + patience

Stop if the global best improves by less than `tol=1e-10` for `patience=30` consecutive iterations. Most runs converge well before 500 iterations. Aggressive patience risks stopping prematurely; the defaults were chosen to balance speed and quality across all four functions.

---

## 9. Reproducibility

Reproducibility is treated as a core requirement of the project:

- Every run is parameterised by an explicit random seed.
- V0, V1, and V2 are launched with the same configuration and same seed so
  their results are directly comparable.
- Structured outputs are saved to disk in `results/` and `logs/`.
- The unit tests explicitly check reproducibility by seed.

This makes it possible to rerun experiments, compare versions fairly, and
inspect convergence behaviour after execution.

## 10. Logging, Timing And Observability

The implementation includes structured logging and timing instrumentation to
support debugging and analysis:

- Run configuration is logged at the start of each experiment.
- Per-iteration logs can include iteration number, best fitness, evaluation
  time, update time, total iteration time, and early-stop progress.
- Final logs summarise total runtime, evaluation time, update time and residual
  overhead.
- Results are also persisted as JSON summaries and CSV convergence histories.

This timing breakdown is especially important for interpreting V2. A correct
multiprocessing implementation can still be slower than V0 in total wall-clock
time if process overhead is high and the sequential particle-update phase
remains dominant.

This observability layer is useful both for software engineering quality and for
the later experimental report.

Every experiment is fully reproducible by seed.

`np.random.default_rng(seed)` controls all random operations — swarm initialisation, and `r1`/`r2` vectors in the velocity update. The same seed always produces the same trajectory and results.

The seed is recorded in three places: `summary.json` under `"seed"`, the log file (`PSO start | seed=42 ...`), and the result directory name (`sphere_d2_s42/`).

V0, V1, and V2 always use the same seed, so they start from identical swarm
states. In all experiments performed so far they reach exactly the same
fitness, confirming that the evaluator swap is algorithmically transparent.

To reproduce any experiment exactly:

```bash
python -m scripts.run_pso --objective sphere --dim 2 --seed 42
```
