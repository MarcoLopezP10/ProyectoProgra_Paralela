# AUTHOR - MARCO LOPEZ PRIETO
# PSO — Particle Swarm Optimization

A maintainable Python implementation of Particle Swarm Optimization (PSO) used as a controlled testbed for comparing multiple execution strategies under the same algorithmic conditions.

Current implemented scope:

- `V0`: sequential baseline
- `V1`: threading with `ThreadPoolExecutor`
- `V2`: multiprocessing with `ProcessPoolExecutor` and batching
- `V3`: `asyncio`-based cooperative concurrency
- `V4`: NumPy-vectorized evaluation and update for supported numerical objectives

Project documents:

- Design notes: `docs/design.md`
- Final report: `docs/final_report.md`
- Internal explanation of versions/results: `docs/v_comparison_and_results.md`
- EDL case ladder and technical notes: `cases/README.md`
- EDL execution protocol: `docs/edl_execution_protocol.md`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Version Summary](#2-version-summary)
3. [Project Structure](#3-project-structure)
4. [Architecture](#4-architecture)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Experimental Highlights](#7-experimental-highlights)
8. [EDL Workflow](#8-edl-workflow)
9. [Reproducibility](#9-reproducibility)

---

## 1. Project Overview

This project implements the canonical PSO algorithm for continuous optimization and compares several ways of executing the expensive parts of the workflow.

The important point is that the project does **not** compare unrelated optimizers. It compares different execution strategies for the same PSO family.

Shared across versions:

- same swarm structure
- same velocity/position update equations
- same topology
- same bounds policy
- same stopping criteria
- same seed-based initialization

Different across versions:

- how evaluations are scheduled or computed
- for `V4`, how the numerical update path is executed internally

This makes the benchmark results fair and interpretable.

### Applied use case: Economic Load Dispatch

Beyond synthetic benchmarks, the repository includes an applied engineering workflow for **Economic Load Dispatch (EDL)**.

In EDL, PSO searches for the best generation vector subject to:

- generator lower and upper bounds
- power-balance constraints
- optional transmission losses
- optional valve-point effects

The EDL workflow currently compares `V0`, `V1`, and `V2` on realistic constrained cases.

---

## 2. Version Summary

| Version | Strategy | What it is good for |
|---|---|---|
| `V0` | Sequential | correctness baseline, low-overhead reference |
| `V1` | Threading | concurrency baseline, sometimes useful when evaluations can overlap waiting |
| `V2` | Multiprocessing | true parallel baseline outside the GIL, but often limited by IPC overhead |
| `V3` | Asyncio | latency-aware or cooperative-concurrency objectives such as `latency_mix` |
| `V4` | NumPy vectorized | structured numerical workloads where Python loop overhead dominates |

Quick interpretation:

- `V3` is the best strategy for latency-shaped workloads.
- `V4` is the best strategy for large vectorizable numerical workloads.

---

## 3. Project Structure

```text
PRACTICA-2.2/
├── core/                 # Particle, swarm, PSO loop
├── options/              # Abstract strategy interfaces
├── parallel/             # V1/V2/V3/V4 evaluators
├── objectives/           # Benchmark and EDL objectives
├── experiment/           # Single runs, grid search, EDL orchestration
├── baseline/             # PySwarm wrapper
├── utils/                # IO, logging, metadata, method metadata
├── viz/                  # Convergence and EDL figures
├── scripts/              # CLI entry points
├── tests/                # Unit/integration tests
├── cases/                # EDL datasets
├── docs/                 # Design, report, internal explanations
├── results/              # Raw saved outputs
├── reports/              # Generated analysis/EDL figures
└── logs/                 # Logs and convergence plots
```

---

## 4. Architecture

The optimizer core is evaluator-agnostic.

Main abstractions:

- `FitnessEvaluator`
- `BoundsPolicy`
- `Topology`

At a high level:

```text
scripts/ -> experiment/ -> core/
                      -> options/
                      -> parallel/
                      -> objectives/
                      -> utils/ + viz/
```

### Why this matters

This design allows the project to compare `V0`-`V4` without rewriting the optimizer every time.

`V4` required only a minimal extension of the evaluator interface:

- classic path: `evaluate(positions)`
- optimized path: optional `step(...)` hook

That hook lets `V4` run a full vectorized PSO step while leaving the common core intact.

### Supported `V4` fast-path objectives

The vectorized `V4` path supports:

- `sphere`
- `ackley`
- `rastrigin`
- `rosenbrock`

If an objective is unsupported, `V4` falls back safely instead of pretending to vectorize everything.

---

## 5. Installation

```bash
git clone <repo-url>
cd PRACTICA-2.2
pip install -e ".[dev]"
```

Python `3.10+` is recommended.

If you only want the core runs, a lighter environment also works as long as the required runtime dependencies are installed.

---

## 6. Usage

All main workflows are exposed as scripts under `scripts/`.

### Single PSO run

```bash
python3 -m scripts.run_pso
python3 -m scripts.run_pso --objective sphere --dim 30 --seed 7
python3 -m scripts.run_pso --objective latency_mix --dim 10 --seed 7 --n-particles 40 --max-iters 60
python3 -m scripts.run_pso --objective sphere --dim 2 --seed 7 --n-particles 8 --max-iters 20 --no-save
```

### Benchmark suite

```bash
python3 -m scripts.run_benchmarks
python3 -m scripts.run_benchmarks --objective sphere --dims 2 10 30 --seeds 42 7
python3 -m scripts.run_benchmarks --objective sphere --dims 2 --seeds 7 --summary-csv results/v4_check.csv
```

### Grid search

```bash
python3 -m scripts.run_grid_search
python3 -m scripts.run_grid_search --objective sphere --dims 2 --strategy v4 --metric time_s
python3 -m scripts.run_grid_search --objective latency_mix --dims 2 --strategy v3 --metric time_s
```

### Analysis

```bash
python3 -m scripts.analyze_results
python3 -m scripts.analyze_results --results-dir results/runs --out-dir reports/analysis_v4_check
```

### Tests

```bash
python3 -m pytest -q
```

### Common `run_pso` objectives

Current objective choices include:

- `sphere`
- `ackley`
- `rosenbrock`
- `rastrigin`
- `latency_mix`

### Recommended validation commands

These are good smoke tests for the current project state:

```bash
python3 -m scripts.run_pso --objective sphere --dim 30 --seed 7 --n-particles 60 --max-iters 120 --no-save
python3 -m scripts.run_pso --objective ackley --dim 30 --seed 7 --n-particles 60 --max-iters 120 --no-save
python3 -m scripts.run_pso --objective rastrigin --dim 30 --seed 7 --n-particles 60 --max-iters 120 --no-save
python3 -m scripts.run_pso --objective latency_mix --dim 10 --seed 7 --n-particles 40 --max-iters 60 --no-save
```

---

## 7. Experimental Highlights

Representative outcomes from the final implementation:

### `sphere`, `d=30`

- `V0`: `0.3971s`
- `V4`: `0.0803s`
- `V4` speedup vs `V0`: `4.948x`
- same final fitness across `V0`-`V4`

Meaning:

- vectorization clearly wins on larger numerical workloads

### `ackley`, `d=30`

- `V0`: `0.6678s`
- `V4`: `0.0913s`
- `V4` speedup vs `V0`: `7.312x`
- same final fitness across `V0`-`V4`

Meaning:

- `V4` is not only good for trivial functions such as `sphere`

### `rastrigin`, `d=30`

- `V0`: `1.2815s`
- `V4`: `0.1064s`
- `V4` speedup vs `V0`: `12.040x`
- same final fitness across `V0`-`V4`

Meaning:

- even on a difficult multimodal landscape, the arithmetic structure still strongly favors `V4`

### `latency_mix`, `d=10`

- `V0`: `6.8641s`
- `V3`: `0.4815s`
- `V3` speedup vs `V0`: `14.255x`
- `V4`: `7.0432s`

Meaning:

- `V3` is the right answer for latency-aware objectives
- `V4` is not meant to dominate there, and it does not

### Main takeaway

- `V4` wins when the bottleneck is numerical computation in Python loops.
- `V3` wins when the bottleneck is waiting/latency.

That is the clearest summary of the project.

---

## 8. EDL Workflow

The EDL path is intentionally separate from the synthetic benchmark path.

### Main commands

```bash
python3 -m scripts.run_edl --case cases/edl_case_3u.json --seed 42
python3 -m scripts.run_edl --case cases/edl_case_6u.json --variant edl_3 edl_4 --seed 42
python3 -m scripts.run_edl --case cases/edl_case_40u.json --variant edl_1 edl_2 --seed 7 --n-particles 100 --max-iters 300 --no-save --no-plots
python3 -m scripts.analyze_edl --case edl_3u_demo --seed 42
```

### Important compatibility note

Not every EDL case supports every EDL variant.

Example:

- `edl_case_40u.json` supports `edl_1` and `edl_2`
- it does **not** support `edl_4`, because the case does not include a transmission-loss model

If you run an incompatible case/variant combination, the project marks it as `unavailable` instead of producing invalid results.

### What EDL demonstrates

EDL is useful because it proves that the PSO framework also works on:

- constrained objectives
- applied optimization cases
- domain-aware workflows where data compatibility matters

Representative `40u` observations:

- `edl_1`: same best fitness and same dispatch across `V0`, `V1`, `V2`
- `edl_2`: same best fitness and same dispatch across `V0`, `V1`, `V2`
- `V2` is again typically the slowest because of process overhead

---

## 9. Reproducibility

The project is reproducible by design:

- runs take explicit seeds
- summaries persist configuration and timing
- the optimizer can rerun from the same initial swarm state
- test coverage verifies equivalence across strategies

Typical reproducibility checks:

```bash
python3 -m pytest -q
python3 -m scripts.run_pso --objective sphere --dim 2 --seed 7 --n-particles 8 --max-iters 20 --no-save
python3 -m scripts.run_benchmarks --objective sphere --dims 2 --seeds 7 --summary-csv results/check.csv
```

For a deeper explanation of the versions and why the results behave the way they do, see:

- `docs/v_comparison_and_results.md`
- `docs/design.md`
- `docs/final_report.md`
