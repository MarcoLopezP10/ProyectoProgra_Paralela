# AUTHOR - MARCO LOPEZ PRIETO
# PSO — Particle Swarm Optimization

Maintainable Python implementation of Particle Swarm Optimization (PSO) used as
a controlled testbed for comparing multiple execution strategies under the same
algorithmic conditions.

Implemented strategies:

- `V0`: sequential baseline
- `V1`: threading with `ThreadPoolExecutor`
- `V2`: multiprocessing with `ProcessPoolExecutor` and batching
- `V3`: `asyncio`-based cooperative concurrency
- `V4`: NumPy-vectorized evaluation and update for supported numerical objectives

The core idea of the repository is simple: there is one PSO algorithm and
several interchangeable execution strategies. The optimizer logic, seed,
topology, bounds policy, and stopping criteria stay fixed. Only the execution
path changes.

## 1. Repository Layout

```text
PRACTICA-2.2/
├── core/                 # Particle, swarm, PSO loop
├── options/              # Bounds, topology, evaluator interfaces
├── parallel/             # V1-V4 evaluator implementations
├── objectives/           # Benchmarks + Economic Load Dispatch objective
├── experiment/           # Single runs, benchmark suite, grid search, EDL runs
├── utils/                # Persistence, metadata, logging
├── viz/                  # Convergence and animation utilities
├── scripts/              # CLI entry points
├── tests/                # Unit and integration tests
├── docs/                 # Design notes, report, execution protocols
├── cases/                # EDL datasets
├── results/              # Saved experiment outputs
└── reports/              # Generated plots and summaries
```

Main documents:

- Design notes: `docs/design.md`
- Final report: `docs/final_report.md`
- EDL execution protocol: `docs/edl_execution_protocol.md`

## 2. Installation

```bash
git clone <repo-url>
cd PRACTICA-2.2
pip install -e ".[dev]"
```

Recommended Python version: `3.10+`

## 3. Main Features

- Canonical PSO for continuous minimization in `R^d`
- Arbitrary dimensionality
- Explicit box-constraint handling through `ClampBounds`
- Reproducibility through explicit seeds
- Shared timing instrumentation:
  - total time
  - evaluation time
  - update time
  - overhead
- Structured persistence:
  - `summary.json`
  - `history_v*.csv`
  - `trajectory_v*.npz`
- 2-D and 3-D visualizations from persisted trajectories
- Grid search over `(w, c1, c2, n_particles)`
- Applied case study: Economic Load Dispatch (EDL)

## 4. Objectives

Numerical benchmarks:

- `sphere`
- `ackley`
- `rosenbrock`
- `rastrigin`

Latency-shaped objective:

- `latency_mix`

Applied engineering objective:

- `economic_dispatch`

## 5. Usage

### Single run

```bash
python3 -m scripts.run_pso
python3 -m scripts.run_pso --objective sphere --dim 30 --seed 7
python3 -m scripts.run_pso --objective latency_mix --dim 10 --seed 7 --n-particles 40 --max-iters 60
```

### Benchmark suite

```bash
python3 -m scripts.run_benchmarks
python3 -m scripts.run_benchmarks --objective sphere ackley rosenbrock rastrigin --dims 2 10 30 --seeds 0 1 7 42 123 --report-ready --suite-name final_protocol_check
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
python3 -m scripts.analyze_results --results-dir results/benchmark_suites/final_protocol_check/runs --out-dir reports/analysis_final_protocol
```

### Visualizations

```bash
python3 -m scripts.make_viz --objective sphere --dim 2 --seed 7 --strategy v0
python3 -m scripts.make_viz --objective ackley --dim 3 --seed 7 --strategy v0
```

### EDL workflow

```bash
python3 -m scripts.run_edl
python3 -m scripts.analyze_edl
```

### Tests

```bash
python3 -m pytest -q
```

## 6. Persistence Format

Each benchmark run is stored under:

```text
results/runs/<objective>_d<dim>_s<seed>/
  summary.json
  history_v0.csv
  history_v1.csv
  history_v2.csv
  history_v3.csv
  history_v4.csv
  trajectory_v0.npz
  trajectory_v1.npz
  trajectory_v2.npz
  trajectory_v3.npz
  trajectory_v4.npz
```

Important saved fields:

- `winner_internal`: best method among `V0`-`V4`
- `winner_overall`: best method if the external PySwarm baseline is included
- `baseline_reference`: textual comparison between the external baseline and the
  best internal result

The animation tool now reads `trajectory_v*.npz` directly, so the visualized
swarm path corresponds to the real stored execution instead of a fresh rerun.

## 7. Final Benchmark Protocol

Final benchmark suite used for the report:

- objectives: `sphere`, `ackley`, `rosenbrock`, `rastrigin`
- dimensions: `2`, `10`, `30`
- seeds: `0`, `1`, `7`, `42`, `123`
- total runs: `4 x 3 x 5 = 60`

The complete suite was executed with:

```bash
python3 -m scripts.run_benchmarks \
  --objective sphere ackley rosenbrock rastrigin \
  --dims 2 10 30 \
  --seeds 0 1 7 42 123 \
  --report-ready \
  --suite-name final_protocol_check
```

Saved artifacts:

- raw summaries: `results/benchmark_suites/final_protocol_check/runs/`
- benchmark CSV: `results/benchmark_suites/final_protocol_check/benchmark_summary.csv`
- suite manifest: `results/benchmark_suites/final_protocol_check/suite_manifest.json`
- aggregated plots: `reports/analysis_final_protocol/`

## 8. Experimental Highlights

The aggregate summary in `reports/analysis_final_protocol/analysis_summary.csv`
shows a very consistent pattern.

### Numerical workloads

`V4` is the strongest strategy on medium and large numerical benchmarks.

Representative mean speedups vs `V0`:

- `sphere d=30`: `3.86x`
- `ackley d=30`: `7.85x`
- `rosenbrock d=10`: `6.31x`
- `rosenbrock d=30`: `5.47x`
- `rastrigin d=10`: `6.01x`
- `rastrigin d=30`: `5.99x`

Interpretation:

- vectorization consistently reduces Python-loop overhead
- the gain grows with arithmetic intensity and swarm size
- the same final fitness is preserved across `V0`-`V4`

### Threads and asyncio on numerical workloads

`V1` and `V3` are not consistent winners for numerical benchmarks.

Typical aggregate behavior:

- `V1` stays near `V0` or slightly worse
- `V3` is sometimes slightly faster than `V0`
- `V2` is unstable because process overhead is expensive unless the workload is
  sufficiently heavy

This is expected and supports the main discussion of GIL, scheduling overhead,
and IPC cost.

### External PySwarm baseline

The project still records PySwarm as an external reference, but the internal
winner is tracked separately. This avoids mixing an external implementation with
the internal PSO family when discussing fairness.

Examples from the final suite:

- on `sphere`, internal `V4` clearly wins and PySwarm usually lags behind
- on `ackley d=30`, PySwarm sometimes reaches lower fitness than the internal
  family, even though the internal family is methodologically comparable among
  itself
- on `rastrigin d=10`, PySwarm can occasionally obtain a lower final fitness,
  which is reported as an external reference rather than an internal winner

### Latency-shaped workloads

For `latency_mix`, concurrency becomes meaningful:

- `V1` and `V3` can outperform `V0`
- `V4` does not dominate, because the bottleneck is not arithmetic structure

That distinction is important: the repository does not claim that one technique
wins everywhere. It shows that the best execution strategy depends on workload
type.

## 9. Economic Load Dispatch

The repository also includes an applied PSO workflow for Economic Load Dispatch.

Validated default run:

```bash
python3 -m scripts.run_edl
python3 -m scripts.analyze_edl
```

Observed on the default `3U` case with seed `42`:

- all three methods (`V0`, `V1`, `V2`) converged to the same dispatch vector in
  the valid variants
- the winner changed only by wall-clock time, not by solution quality
- `V1` won the base and valve-point variants
- `V0` won the variants with losses in the default checked execution

This is good evidence that the shared core remains behaviorally consistent in a
real constrained engineering problem.

## 10. Current Conclusions

The repository supports the following claims:

- one shared PSO core can support multiple execution strategies without changing
  optimization behavior
- `V4` is the recommended strategy for structured numerical workloads
- `V1` and `V3` only become compelling when there is waiting or latency to
  overlap
- `V2` is scientifically useful as a multiprocessing baseline, but often pays
  too much overhead on these benchmark sizes
- the EDL case study confirms that the framework is not limited to toy
  benchmarks
