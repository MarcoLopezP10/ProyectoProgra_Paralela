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
| V2 | Multiprocessing | `ProcessPoolExecutor` — true parallelism *(coming)* |
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
│   └── evaluator.py            # ThreadPoolEvaluator (V1)
│
├── objectives/                 # Benchmark functions
│   ├── sphere.py               # Sphere — unimodal, convex
│   ├── ackley.py               # Ackley — multimodal, origin trap
│   ├── rosenbrock.py           # Rosenbrock — narrow curved valley
│   └── rastrigin.py            # Rastrigin — highly multimodal
│
├── experiment/                 # Experiment orchestration
│   ├── grid_search.py          # Multi-seed grid search
│   └── run_single.py           # Single experiment runner (V0 + V1 + baseline)
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
                             SequentialEvaluator   ThreadPoolEvaluator
                             (V0 — options/)       (V1 — parallel/)
                                   │
                                   ▼
                             objectives/
                             sphere, ackley, rosenbrock, rastrigin
```

### Key abstractions

**`FitnessEvaluator` (ABC)** — defined in `options/evaluator.py`. Single method: `evaluate(positions) -> List[float]`. PSO calls this once per iteration without knowing whether evaluation is sequential, threaded, or vectorised. This is what makes swapping V0/V1/V2/… completely transparent.

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
| `--grid-search` | off | Enable hyperparameter grid search |
| `--workers` | auto | Max threads for V1 ThreadPoolExecutor |
| `--no-save` | off | Skip saving files to disk |

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

### V2 — Multiprocessing *(coming)*

`ProcessPoolExecutor` — separate OS processes bypass the GIL entirely, achieving true CPU parallelism. Cost: pickling overhead and inter-process communication (IPC). Will use batching (sending blocks of particles per task) to amortise that cost.

### V3 — Asyncio *(coming)*

Cooperative concurrency. Only makes sense when evaluation is I/O-bound — for example, each particle queries a local service with variable latency. Uses `asyncio.gather()` to overlap waiting times.

### V4 — NumPy vectorised *(coming)*

Eliminates all Python loops. Positions, velocities, and fitness evaluation operate on full matrices `(n_particles × dim)`. Expected to be the fastest version for pure mathematical objectives.

---

## 7. Experimental Results

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

**Key observation:** V0 and V1 always reach exactly the same fitness. The parallelism strategy does not affect solution quality — only execution time. This confirms the abstraction works correctly: swapping the evaluator changes nothing algorithmically.

PySwarm wins on multimodal functions at high dimension (Ackley d=30, Rastrigin d=10/30, Rosenbrock d=30) because our PSO uses fixed hyperparameters not tuned for those cases. With grid search the gap closes significantly.

### Timing comparison

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

### Early stopping: tolerance + patience

Stop if the global best improves by less than `tol=1e-10` for `patience=30` consecutive iterations. Most runs converge well before 500 iterations. Aggressive patience risks stopping prematurely; the defaults were chosen to balance speed and quality across all four functions.

---

## 9. Reproducibility

Every experiment is fully reproducible by seed.

`np.random.default_rng(seed)` controls all random operations — swarm initialisation, and `r1`/`r2` vectors in the velocity update. The same seed always produces the same trajectory and results.

The seed is recorded in three places: `summary.json` under `"seed"`, the log file (`PSO start | seed=42 ...`), and the result directory name (`sphere_d2_s42/`).

V0 and V1 always use the same seed, so they start from identical swarm states. In all experiments they reach exactly the same fitness, confirming that the evaluator swap is algorithmically transparent.

To reproduce any experiment exactly:

```bash
python -m scripts.run_pso --objective sphere --dim 2 --seed 42
```