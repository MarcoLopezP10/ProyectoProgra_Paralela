# EDL Case Ladder

These JSON files give you a size ladder for the new `EDL` workflow:

- `edl_case_3u.json`
  Small validation case. Supports valve-point and losses.
- `edl_case_6u.json`
  Medium case. Supports valve-point and losses.
  Uses the standard IEEE 100-MVA base on the quadratic loss term, so the JSON
  keeps the published `B` coefficients and declares `quadratic_base_mva: 100.0`.
- `edl_case_13u.json`
  Large valve-point case. Standard benchmark without transmission-loss data.
- `edl_case_40u.json`
  Very large valve-point case. Standard benchmark without transmission-loss data.

Recommended progression:

1. Validate formulas and reproducibility with `3U`.
2. Check constrained behavior with `6U`.
3. Stress the optimizer on `13U`.
4. Show scalability on `40U`.

Example commands:

```bash
python3 -m scripts.run_edl --case cases/edl_case_3u.json --seed 42
python3 -m scripts.run_edl --case cases/edl_case_6u.json --seed 42
python3 -m scripts.run_edl --case cases/edl_case_13u.json --variant edl_1 edl_2 --seed 42
python3 -m scripts.run_edl --case cases/edl_case_40u.json --variant edl_1 edl_2 --seed 42
```

Notes:

- `13U` and `40U` do not include transmission-loss coefficients in the standard
  source data used here, so `edl_3` and `edl_4` are marked unavailable by the
  runner unless you provide a custom loss model.
- Extra fields like `source` are included only for provenance; the loader uses
  the same core schema as before.
