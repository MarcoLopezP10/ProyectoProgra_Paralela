# EDL Final Execution Protocol

This document defines the recommended workflow for running the **Economic Load
Dispatch (EDL)** case study in its final form.

The goal is to make the execution easy for a reviewer or professor, especially
from **VS Code Run and Debug**, while keeping the outputs consistent across all
delivery cases.

## 1. What This Protocol Covers

The protocol is based on the four EDL datasets:

- `3U` — validation case
- `6U` — medium constrained case with losses
- `13U` — large valve-point case
- `40U` — large scalability case

It uses the same PSO implementations:

- `V0` — sequential
- `V1` — threading
- `V2` — multiprocessing
- `V3` — asyncio

and also reports the external reference:

- `baseline` — PySwarm

and the same EDL variants:

- `edl_1` — base
- `edl_2` — valve-point
- `edl_3` — losses
- `edl_4` — valve-point + losses

## 2. Recommended Run and Debug Profiles

The simplest way to execute the final workflow is through the predefined
profiles in `.vscode/launch.json`.

### Main profiles for the final delivery

| Purpose | Run and Debug profile |
|---|---|
| Validate all four EDL variants on the small case | `EDL: 3U Final Run` |
| Show the full constrained workflow with losses | `EDL: 6U Final Run` |
| Show large-case valve-point behaviour | `EDL: 13U Final Run` |
| Show large-scale behaviour | `EDL: 40U Final Run` |

### Support profiles

| Purpose | Run and Debug profile |
|---|---|
| Quick validation run on `3U` | `EDL: 3U Validation (quick)` |
| Quick validation run on `6U` | `EDL: 6U Validation (quick)` |
| Quick validation run on `13U` | `EDL: 13U Validation (quick)` |
| Quick validation run on `40U` | `EDL: 40U Validation (quick)` |
| Choose any case and run all valid variants | `EDL: Custom Case (all valid variants)` |
| Choose any case and a specific variant | `EDL: Custom Case (single variant)` |
| Regenerate report figures from saved results | `EDL: Regenerate Reports` |

## 3. Final Parameters

The following settings are the recommended final configurations for the
delivery figures.

| Case | Variants | Particles | Max iterations | Main objective |
|---|---|---:|---:|---|
| `3U` | `edl_1` `edl_2` `edl_3` `edl_4` | `60` | `300` | Validate formulas and all EDL modes |
| `6U` | `edl_1` `edl_2` `edl_3` `edl_4` | `80` | `400` | Show the full EDL case with losses |
| `13U` | `edl_1` `edl_2` | `100` | `500` | Show scaling on a large valve-point problem |
| `40U` | `edl_1` `edl_2` | `120` | `600` | Show high-dimensional scalability |

These settings are already reflected in the predefined `Full + Reports`
profiles.

## 4. Expected Outputs

Each execution generates:

### Raw outputs

- `results/edl/<case_name>/comparison.csv`
- `results/edl/<case_name>/<variant>_s<seed>/summary.json`
- `results/edl/<case_name>/<variant>_s<seed>/history_v*.csv`

### Final report figures

- `reports/edl/<case_name>/seed_<seed>/convergence_overview.png`
- `reports/edl/<case_name>/seed_<seed>/dispatch_overview.png`
- `reports/edl/<case_name>/seed_<seed>/summary_dashboard.png`

These three figures are the only figures intended for final presentation.

## 5. What Each Figure Shows

### `convergence_overview.png`

Shows the evolution of the best fitness over the iterations for each EDL
variant. This is the main figure for explaining **PSO behaviour**.

Use it to show:

- that the PSO converges
- that the convergence pattern changes with valve-point and/or losses
- that `V0`, `V1`, `V2`, and `V3` stay behaviourally consistent

### `dispatch_overview.png`

Shows the final generator outputs selected by the winning method for each
variant. This is the main figure for explaining the **final solution**.

Use it to show:

- how the optimal dispatch changes between variants
- how the presence of losses shifts the total generated power upward
- how the case difficulty changes the resulting power allocation

### `summary_dashboard.png`

Summarises the winning result per variant:

- best fitness
- time
- losses
- balance error

Use it to show:

- the final numerical comparison
- whether balance error is acceptably small
- which variant is most expensive or most constrained

## 6. Recommended Presentation Order

If the reviewer runs everything:

1. Run `EDL: 3U Final Run`
2. Run `EDL: 6U Final Run`
3. Run `EDL: 13U Final Run`
4. Run `EDL: 40U Final Run`

If time is limited:

1. Run `EDL: 3U Final Run`
2. Run `EDL: 6U Final Run`
3. Run `EDL: 40U Final Run`

This shorter sequence still shows:

- validation
- full constrained behaviour
- scalability

## 7. Suggested Figure Selection

For a short explanation or live demo:

- `3U`: `dispatch_overview.png` and `summary_dashboard.png`
- `6U`: `dispatch_overview.png` and `summary_dashboard.png`
- `40U`: `convergence_overview.png` and `dispatch_overview.png`

For a written report:

- include all three figures for `3U`
- include all three figures for `6U`
- include `convergence_overview.png` and `summary_dashboard.png` for `13U`
- include `convergence_overview.png` and `dispatch_overview.png` for `40U`

## 8. Terminal Equivalents

If needed, the `Run and Debug` profiles correspond to these commands:

```bash
python3 -m scripts.run_edl --case cases/edl_case_3u.json --seed 42 --n-particles 60 --max-iters 300
python3 -m scripts.run_edl --case cases/edl_case_6u.json --seed 42 --n-particles 80 --max-iters 400
python3 -m scripts.run_edl --case cases/edl_case_13u.json --variant edl_1 edl_2 --seed 42 --n-particles 100 --max-iters 500
python3 -m scripts.run_edl --case cases/edl_case_40u.json --variant edl_1 edl_2 --seed 42 --n-particles 120 --max-iters 600
```

To regenerate figures from already saved results:

```bash
python3 -m scripts.analyze_edl --case edl_3u_demo --seed 42
python3 -m scripts.analyze_edl --case edl_6u_vpe_losses --seed 42
python3 -m scripts.analyze_edl --case edl_13u_vpe --seed 42
python3 -m scripts.analyze_edl --case edl_40u_vpe --seed 42
```

## 9. Acceptance Criteria

The execution can be considered correct when:

- the EDL table is produced for every valid variant
- `comparison.csv` is saved under `results/edl/<case_name>/`
- exactly three report figures appear under `reports/edl/<case_name>/seed_<seed>/`
- balance error is small in the saved summaries
- the `13U` and `40U` cases only run `edl_1` and `edl_2`
- the saved summaries include `baseline`, `v0`, `v1`, `v2`, and `v3`

This protocol is the intended final workflow for validating and presenting the
EDL case study.
