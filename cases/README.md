# EDL Cases, Formulation, and Data Schema

This directory contains the datasets used by the **Economic Load Dispatch
(EDL)** workflow added to the PSO project.

The purpose of the EDL workflow is to test the same `V0`, `V1`, and `V2` PSO
implementations on a realistic constrained optimisation problem instead of only
on synthetic benchmark functions.

## 1. Mathematical Formulation

The decision vector is the generator dispatch:

\[
P = [P_1, P_2, \dots, P_n]
\]

where \(P_i\) is the output power of generator \(i\).

### Base fuel-cost model

\[
F(P) = \sum_{i=1}^{n} (a_i P_i^2 + b_i P_i + c_i)
\]

### Valve-point model

When valve-point loading is enabled, the fuel cost becomes:

\[
F(P) = \sum_{i=1}^{n}
\left(
a_i P_i^2 + b_i P_i + c_i +
\left|e_i \sin(f_i(P_i^{\min} - P_i))\right|
\right)
\]

This introduces ripples and multiple local minima.

### Transmission losses

When losses are enabled, they are computed as:

\[
P_L = \frac{P^T B P}{\text{quadratic\_base\_mva}} + B_0^T P + B_{00}
\]

The default base is `1.0`. The `6U` case uses `100.0` because the published
loss coefficients are expressed on the IEEE 100-MVA base.

### Power-balance constraint

- Without losses:

\[
\sum_{i=1}^{n} P_i = P_D
\]

- With losses:

\[
\sum_{i=1}^{n} P_i = P_D + P_L
\]

### Box constraints

\[
P_i^{\min} \le P_i \le P_i^{\max}
\]

### Fitness used by PSO

The PSO minimises a penalised objective:

\[
\text{fitness}(P) = F(P) + \lambda \cdot \left|\sum_i P_i - (P_D + P_L)\right|^2
\]

where:

- `F(P)` is the fuel cost, with or without valve-point
- `P_L` is `0` when losses are disabled
- `\lambda` is the power-balance penalty

This fits the existing PSO architecture because only the objective function
changes; the core PSO loop remains identical across `V0`, `V1`, and `V2`.

## 2. Implemented EDL Variants

Each JSON case stores a full generator definition. The four EDL variants are
activated by flags at runtime rather than by duplicating the case files.

| Variant | Valve-point | Losses | Meaning |
|---|---:|---:|---|
| `edl_1` | no | no | Base quadratic dispatch |
| `edl_2` | yes | no | Non-smooth valve-point dispatch |
| `edl_3` | no | yes | Dispatch with transmission losses |
| `edl_4` | yes | yes | Full EDL problem |

## 3. Case Ladder

The available cases are intentionally organised as a progression from validation
to scalability.

| File | Units | Loss model available | Recommended variants | Main purpose |
|---|---:|---:|---|---|
| `edl_case_3u.json` | 3 | yes | `edl_1` `edl_2` `edl_3` `edl_4` | Small validation case |
| `edl_case_6u.json` | 6 | yes | `edl_1` `edl_2` `edl_3` `edl_4` | Medium constrained case |
| `edl_case_13u.json` | 13 | no | `edl_1` `edl_2` | Large valve-point case |
| `edl_case_40u.json` | 40 | no | `edl_1` `edl_2` | Very large scalability case |

Recommended progression:

1. Validate formulas and outputs with `3U`
2. Show losses and valve-point together with `6U`
3. Show scaling on `13U`
4. Show large-case behaviour on `40U`

## 4. JSON Schema

The loader expects a structure like this:

```json
{
  "case_name": "edl_3u_demo",
  "demand": 850.0,
  "generators": [
    {
      "name": "G1",
      "p_min": 100.0,
      "p_max": 600.0,
      "a": 0.001562,
      "b": 7.92,
      "c": 561.0,
      "e": 300.0,
      "f": 0.0315
    }
  ],
  "losses": {
    "B": [[0.00014]],
    "B0": [0.0],
    "B00": 0.0,
    "quadratic_base_mva": 1.0
  }
}
```

### Required top-level fields

- `case_name`
- `demand`
- `generators`

### Required generator fields

- `name`
- `p_min`
- `p_max`
- `a`
- `b`
- `c`

### Optional generator fields

- `e`
- `f`

If `e` and `f` are omitted, the case can still be used for `edl_1` and `edl_3`.

### Optional loss-model fields

- `losses.B`
- `losses.B0`
- `losses.B00`
- `losses.quadratic_base_mva`

If the loss model is missing, the runner marks `edl_3` and `edl_4` as
unavailable for that case.

### Provenance fields

Some JSON files may include source metadata for traceability. These fields are
ignored by the loader and do not affect the optimisation.

## 5. Technical Notes by Case

### `3U`

- Full validation case
- Supports all four variants
- Best choice for checking balance, losses, and report generation quickly

### `6U`

- Medium-size case with losses and valve-point
- Uses `quadratic_base_mva: 100.0`
- This is the most representative case for showing the full EDL workflow

### `13U`

- Large valve-point benchmark
- Standard source data does not include a transmission-loss model
- Use `edl_1` and `edl_2`

### `40U`

- Very large valve-point benchmark
- Intended to show scalability and runtime behaviour
- Use `edl_1` and `edl_2`

## 6. Execution

### Terminal examples

```bash
python3 -m scripts.run_edl --case cases/edl_case_3u.json --seed 42
python3 -m scripts.run_edl --case cases/edl_case_6u.json --seed 42
python3 -m scripts.run_edl --case cases/edl_case_13u.json --variant edl_1 edl_2 --seed 42
python3 -m scripts.run_edl --case cases/edl_case_40u.json --variant edl_1 edl_2 --seed 42
```

### VS Code Run and Debug

The project includes predefined profiles in `.vscode/launch.json`:

- `EDL: 3U Validation (quick)`
- `EDL: 3U Final Run`
- `EDL: 6U Validation (quick)`
- `EDL: 6U Final Run`
- `EDL: 13U Validation (quick)`
- `EDL: 13U Final Run`
- `EDL: 40U Validation (quick)`
- `EDL: 40U Final Run`
- `EDL: Custom Case (all valid variants)`
- `EDL: Custom Case (single variant)`
- `EDL: Regenerate Reports`

For the final delivery workflow, use the protocol documented in
[`docs/edl_execution_protocol.md`](../docs/edl_execution_protocol.md).

## 7. Outputs

Each EDL run saves:

- `results/edl/<case_name>/comparison.csv`
- `results/edl/<case_name>/<variant>_s<seed>/summary.json`
- `results/edl/<case_name>/<variant>_s<seed>/history_v0.csv`
- `results/edl/<case_name>/<variant>_s<seed>/history_v1.csv`
- `results/edl/<case_name>/<variant>_s<seed>/history_v2.csv` when available

The compact report figures are generated in:

- `reports/edl/<case_name>/seed_<seed>/convergence_overview.png`
- `reports/edl/<case_name>/seed_<seed>/dispatch_overview.png`
- `reports/edl/<case_name>/seed_<seed>/summary_dashboard.png`

These three figures are the intended final outputs for presentation and
comparison.
