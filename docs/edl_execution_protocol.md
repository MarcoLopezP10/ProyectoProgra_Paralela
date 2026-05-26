# EDL Final Execution Protocol

This document explains how the Economic Load Dispatch (EDL) case study is
executed, what outputs are generated, and how to interpret the checked results
already obtained in the repository.

## 1. Scope

The EDL workflow uses the same shared PSO core as the benchmark suite and
compares three execution strategies:

- `V0` — sequential
- `V1` — threading
- `V2` — multiprocessing

This is intentional: the EDL protocol is meant to validate the shared PSO core
on an engineering problem using the three directly relevant execution modes that
were checked for this objective family.

Available EDL variants:

- `edl_1` — base
- `edl_2` — valve-point
- `edl_3` — losses
- `edl_4` — valve-point + losses

Available datasets:

- `cases/edl_case_3u.json`
- `cases/edl_case_6u.json`
- `cases/edl_case_13u.json`
- `cases/edl_case_40u.json`

## 2. Main Goal of the EDL Study

The EDL case study is not mainly about achieving the largest speedup. Its role
in the project is to show that:

- the PSO framework works on a constrained engineering problem
- the same architecture used for the synthetic benchmarks can be reused without
  changing the optimizer core
- `V0`, `V1`, and `V2` remain behaviorally consistent on realistic inputs

So the central success criterion is:

- same or equivalent solution quality across strategies
- runtime differences explained by execution overhead

## 3. Recommended Commands

Default checked execution:

```bash
python3 -m scripts.run_edl
python3 -m scripts.analyze_edl
```

This produces:

- raw summaries in `results/edl/`
- aggregate CSVs in `results/edl/<case_name>/comparison.csv`
- report figures in `reports/edl/`

Examples for explicit cases:

```bash
python3 -m scripts.run_edl --case cases/edl_case_3u.json
python3 -m scripts.run_edl --case cases/edl_case_6u.json
python3 -m scripts.run_edl --case cases/edl_case_13u.json --variant edl_1 edl_2
python3 -m scripts.run_edl --case cases/edl_case_40u.json --variant edl_1 edl_2
```

## 4. Output Structure

For each run:

```text
results/edl/<case_name>/<variant>_s<seed>/
  summary.json
  history_v0.csv
  history_v1.csv
  history_v2.csv
```

Note:

- the deliverable EDL protocol is defined in terms of `V0`, `V1`, and `V2`
- if an exploratory run directory contains extra files from local experiments,
  those files are not part of the checked EDL protocol and can be ignored for
  the final interpretation

For each case:

```text
results/edl/<case_name>/comparison.csv
reports/edl/<case_name>/
```

Main saved information:

- best fitness
- fuel cost
- transmission losses
- balance error
- iterations
- runtime breakdown
- final dispatch vector

## 5. Checked Default Results

The default execution that was run and verified in the repository is:

- case: `edl_3u_demo`
- seed: `42`
- variants: `edl_1`, `edl_2`, `edl_3`, `edl_4`

### `edl_1` — Base

All three strategies converged to the same dispatch:

- `[492.0401, 211.8510, 146.1089]`

Final fitness:

- `8241.606706`

Times:

- `V0`: `4.4774s`
- `V1`: `2.2130s`
- `V2`: `4.9827s`

Winner:

- `V1 Threading`

Interpretation:

- same engineering solution
- the difference is execution overhead only

### `edl_2` — Valve-point

Same dispatch across methods:

- `[492.8413, 207.8316, 149.3271]`

Final fitness:

- `8502.280355`

Times:

- `V0`: `6.9076s`
- `V1`: `5.4923s`
- `V2`: `9.5009s`

Winner:

- `V1 Threading`

Interpretation:

- valve-point effects increase difficulty and runtime
- `V2` pays the largest overhead

### `edl_3` — Losses

Same dispatch across methods:

- `[474.9884, 285.9617, 134.2429]`

Final fitness:

- `8623.535676`

Transmission losses:

- `45.192955`

Times:

- `V0`: `3.8272s`
- `V1`: `3.9381s`
- `V2`: `4.8263s`

Winner:

- `V0 Sequential`

Interpretation:

- once losses are included, sequential execution can remain competitive
- no strategy changed the final engineering solution

### `edl_4` — Valve-point + Losses

Same dispatch across methods:

- `[486.6277, 272.0889, 137.5926]`

Final fitness:

- `9019.668565`

Transmission losses:

- `46.309288`

Times:

- `V0`: `3.8603s`
- `V1`: `4.2946s`
- `V2`: `4.0734s`

Winner:

- `V0 Sequential`

Interpretation:

- this is the most constrained checked `3U` variant
- the shared core still remains consistent across execution modes

## 6. What the Checked EDL Results Mean

The default checked execution supports three important claims:

1. The architecture is reusable.
   The same optimizer core works for synthetic benchmarks and for EDL.

2. The strategies remain comparable.
   In the checked runs, `V0`, `V1`, and `V2` produced the same dispatch vector
   and the same final fitness per variant.

3. Speedup is not the only metric that matters.
   The main scientific value here is consistency of the final engineering
   result, not just raw runtime.

## 7. Presentation Guidance

For the report or demo, the EDL section should emphasize:

- generator bounds are respected
- power-balance error is very small
- losses change the dispatch and the final objective
- valve-point effects increase difficulty
- execution strategies do not change the optimal dispatch in the checked runs

That is a strong argument for the maintainability of the architecture.

## 8. Recommended Figures

After running:

```bash
python3 -m scripts.analyze_edl
```

Use:

- `convergence_overview.png` to discuss optimizer behavior
- `dispatch_overview.png` to discuss final power allocation
- `summary_dashboard.png` to discuss fitness, losses, and balance error

These figures are enough for a short presentation and also fit naturally inside
the written report.

## 9. Validity Notes

Not every dataset supports every EDL variant.

Examples:

- cases without a loss model should not run `edl_3` or `edl_4`
- this is expected behavior, not a failure

The implementation correctly marks unsupported combinations as unavailable.

## 10. Final Conclusion

The EDL workflow validates the project beyond toy benchmarks:

- it reuses the same PSO core
- it preserves strategy comparability
- it produces stable engineering outputs
- it demonstrates that the framework is maintainable and extensible

For this project, EDL is the strongest practical evidence that the software
design decisions were correct.
