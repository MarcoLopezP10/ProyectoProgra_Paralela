# Final Report — PSO `V0` to `V4`

## 1. Objective

This project implements a maintainable Particle Swarm Optimization (PSO)
framework in Python and uses it as a controlled experimental platform for
comparing several execution strategies under identical optimization conditions.

Implemented strategies:

- `V0`: sequential baseline
- `V1`: threaded evaluation
- `V2`: multiprocessing evaluation with batching
- `V3`: `asyncio`-based cooperative evaluation
- `V4`: NumPy-vectorized evaluation and update

The methodological rule of the project is strict:

- the optimizer logic stays fixed
- the seed stays fixed
- the starting swarm stays fixed
- only the execution strategy changes

That rule is what makes the results interpretable.

## 2. Methodology

### 2.1 Benchmarks

The final numerical benchmark suite used:

- `sphere`
- `ackley`
- `rosenbrock`
- `rastrigin`

Dimensions:

- `2`
- `10`
- `30`

Seeds:

- `0`
- `1`
- `7`
- `42`
- `123`

Total controlled runs:

- `4 objectives x 3 dimensions x 5 seeds = 60 runs`

Command used:

```bash
python3 -m scripts.run_benchmarks \
  --objective sphere ackley rosenbrock rastrigin \
  --dims 2 10 30 \
  --seeds 0 1 7 42 123 \
  --report-ready \
  --suite-name final_protocol_check
```

Saved outputs:

- raw runs: `results/benchmark_suites/final_protocol_check/runs/`
- flat summary: `results/benchmark_suites/final_protocol_check/benchmark_summary.csv`
- analysis: `reports/analysis_final_protocol/`

### 2.2 Metrics

For each run, the project records:

- final best fitness
- convergence iteration
- convergence AUC
- total wall-clock time
- evaluation time
- update time
- residual overhead

### 2.3 Baseline handling

The repository also records PySwarm as an external reference implementation.
However, the final persistence format now separates:

- `winner_internal`: best method among `V0`-`V4`
- `winner_overall`: best method if PySwarm is also considered

This is important because PySwarm is not part of the controlled internal PSO
family.

## 3. Final Aggregate Results

The aggregate summary is stored in:

- `reports/analysis_final_protocol/analysis_summary.csv`

The main conclusion is very clear:

- `V4` is the dominant strategy for structured numerical workloads
- `V1`, `V2`, and `V3` do not consistently outperform `V0` on pure numerical
  benchmarks
- the internal methods preserve the same final fitness under the same seed and
  configuration

## 4. Objective-by-Objective Interpretation

### 4.1 Sphere

Aggregate mean speedups vs `V0`:

- `d=2`: `V4 = 3.37x`
- `d=10`: `V4 = 3.70x`
- `d=30`: `V4 = 3.86x`

Interpretation:

- even on the simplest convex problem, vectorization is consistently effective
- concurrency strategies add overhead without enough useful work to amortize it
- `V3` stays close to `V0` on `d=30`, but does not surpass the vectorized path

This benchmark shows the cleanest baseline story: if the objective is regular
and arithmetic-heavy, vectorization is the right optimization route.

### 4.2 Ackley

Aggregate mean speedups vs `V0`:

- `d=2`: `V4 = 3.29x`
- `d=10`: `V4 = 5.76x`
- `d=30`: `V4 = 7.85x`

Interesting secondary pattern:

- `V3` reaches `1.32x` on `d=30`
- `V2` reaches `1.17x` on `d=30`

Interpretation:

- as the arithmetic structure becomes heavier, `V4` becomes much stronger
- `V3` and `V2` can occasionally improve over `V0`, but they are not the main
  story
- on some `d=30` seeds, the external PySwarm baseline reaches a lower final
  fitness than the internal family, which is why the baseline is reported
  separately instead of being mixed into the internal comparison

### 4.3 Rosenbrock

Aggregate mean speedups vs `V0`:

- `d=2`: `V4 = 4.07x`
- `d=10`: `V4 = 6.31x`
- `d=30`: `V4 = 5.47x`

Other strategies:

- `V3`: `1.22x` on `d=10`
- `V3`: `1.10x` on `d=30`
- `V1` and `V2`: usually near `V0` or worse

Interpretation:

- Rosenbrock is a strong test because it is not a trivial bowl, yet it still
  remains numerically structured enough for vectorization
- `V4` stays dominant while preserving the same fitness
- mild `V3` gains appear in some cases, but they are not strong enough to
  challenge the vectorized path

### 4.4 Rastrigin

Aggregate mean speedups vs `V0`:

- `d=2`: `V4 = 3.68x`
- `d=10`: `V4 = 6.01x`
- `d=30`: `V4 = 5.99x`

Other strategies:

- `V3`: `1.33x` on `d=30`
- `V2`: `0.94x` on `d=30`

Interpretation:

- the landscape is highly multimodal, but the arithmetic is still regular
- this confirms that `V4` is exploiting computation structure rather than
  convexity or landscape simplicity
- some seeds show interesting exceptions, such as `V2` or `V3` occasionally
  beating `V0`, but these are not stable enough to replace the global
  conclusion

## 5. What the Speedups Mean

The results support the following interpretation:

### `V1`

- good reference for lightweight concurrency
- not consistently useful for numerical benchmarks
- potentially useful when evaluations contain waiting

### `V2`

- scientifically useful as a multiprocessing baseline
- pays high overhead from process management, IPC, and serialization
- only occasionally becomes competitive on heavier workloads

### `V3`

- not designed as a numeric acceleration technique
- still sometimes slightly improves over `V0`
- conceptually strongest when latency or cooperative waiting exists

### `V4`

- strongest internal strategy in the final numerical suite
- benefits grow with dimension and arithmetic intensity
- preserves the same optimizer behavior while significantly reducing runtime

## 6. Why the Internal Comparison Is Valid

Across the final suite, the internal strategies consistently reached the same
final fitness per run:

- same seed
- same objective
- same hyperparameters
- same stopping criteria

This strongly suggests that:

- the optimizer core is stable
- the abstractions are correct
- the differences in runtime are truly execution-strategy differences

This is the central scientific value of the project.

## 7. Visualization and Persistence

The repository stores:

- `summary.json`
- `history_v*.csv`
- `trajectory_v*.npz`

The trajectory archives are important because they allow:

- 2-D and 3-D swarm animations
- reuse of real stored trajectories
- avoidance of misleading reruns for visualization only

The analysis pipeline produces:

- mean convergence plots
- final-fitness boxplots
- mean speedup plots
- timing-breakdown plots

These figures are available under `reports/analysis_final_protocol/`.

## 8. Applied Case Study: Economic Load Dispatch

The project also includes an Economic Load Dispatch case study, executed with:

```bash
python3 -m scripts.run_edl
python3 -m scripts.analyze_edl
```

Checked default execution:

- case: `edl_3u_demo`
- seed: `42`
- variants: `edl_1`, `edl_2`, `edl_3`, `edl_4`

Observed behavior:

- `V0`, `V1`, and `V2` converged to the same dispatch vector on the valid
  variants
- the winner changed only by runtime
- `V1` won `edl_1` and `edl_2`
- `V0` won `edl_3` and `edl_4`

Example:

- `edl_1` best powers: `[492.0401, 211.8510, 146.1089]`
- all three methods reached the same fitness: `8241.606706`

Interpretation:

- the framework is not limited to toy benchmarks
- the architecture stays valid in a constrained engineering scenario
- EDL strengthens the maintainability argument of the project

## 9. Main Discussion

The final results support a nuanced conclusion:

1. There is no universally best execution strategy for every possible workload.
2. For regular numerical PSO workloads, vectorization is the most effective
   optimization path.
3. Threads and asyncio are meaningful primarily when there is concurrency to
   exploit, not just arithmetic to compute.
4. Multiprocessing is important to evaluate, but its overhead often prevents it
   from being the best option at these benchmark scales.

This is a better result than a simplistic "`X` always wins" story, because it
matches the actual engineering trade-offs.

## 10. Final Conclusion

The project successfully meets its core objective:

- one maintainable PSO framework
- one shared optimizer core
- multiple interchangeable execution strategies
- reproducible experiments
- meaningful persistence and reporting
- an applied engineering case study

The strongest final recommendation is:

- use `V4` for structured numerical workloads
- use `V1` or `V3` only when the workload has waiting or latency to overlap
- treat `V2` as a valuable comparison point, but not as the default choice

In short, the repository demonstrates both sound software engineering and a
clear experimental conclusion: for this PSO workload family, vectorization is
the most robust path to better performance without changing optimizer behavior.
