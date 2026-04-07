"""scripts.analyze_results

Load persisted benchmark results and generate summary tables/plots.

Outputs:
- CSV summary by objective/dimension/method
- Mean convergence plots
- Final fitness boxplots
- Mean speedup bar charts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Allow direct execution as `python scripts/analyze_results.py` from editors
# like VS Code while keeping `python -m scripts.analyze_results` working.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


CONVERGENCE_METHODS = [
    ("v0", "V0 Sequential"),
    ("v1", "V1 Threading"),
    ("v2", "V2 Multiprocessing"),
]

SUMMARY_METHODS = CONVERGENCE_METHODS + [
    ("baseline", "PySwarm baseline"),
]

TIMING_COMPONENTS = [
    ("eval_s", "Evaluation", "#4c78a8"),
    ("update_s", "Update", "#f58518"),
    ("overhead_s", "Overhead", "#54a24b"),
]


def _trapezoid_area(values: List[float]) -> float:
    """Compute AUC without tripping NumPy 2.x deprecation warnings."""
    arr = np.asarray(values, dtype=float)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(arr))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze PSO benchmark results saved on disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default="results/runs", help="Directory containing summary.json files.")
    p.add_argument("--out-dir", default="reports/analysis", help="Directory for generated analysis artifacts.")
    p.add_argument("--objective", nargs="+", default=None, help="Optional subset of objective names.")
    p.add_argument("--dim", nargs="+", type=int, default=None, help="Optional subset of dimensions.")
    p.add_argument("--keep-old", action="store_true", help="Do not remove previously generated analysis files.")
    return p.parse_args(argv)


def _compute_auc(history: List[float]) -> Optional[float]:
    if not history:
        return None
    if len(history) == 1:
        return float(history[0])
    return _trapezoid_area(history)


def _compute_convergence_iteration(
    history: List[float],
    tol: Optional[float],
    rel_tol: float = 0.01,
) -> Optional[int]:
    if not history:
        return None
    final_best = float(history[-1])
    margin = max(float(tol or 1e-8), abs(final_best) * rel_tol)
    for idx, value in enumerate(history):
        if float(value) <= final_best + margin:
            return idx
    return len(history) - 1


def _history_path(results_dir: str, objective: str, dim: int, seed: int, method_key: str) -> str:
    return os.path.join(results_dir, f"{objective}_d{dim}_s{seed}", f"history_{method_key}.csv")


def _load_history(path: str) -> List[float]:
    if not os.path.exists(path):
        return []
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "best_fitness" in row and row["best_fitness"] != "":
                values.append(float(row["best_fitness"]))
    return values


def _normalize_method_summary(
    summary: Dict[str, Any],
    method_key: str,
    results_dir: str,
    label: str,
) -> Optional[Dict[str, Any]]:
    raw_method = summary.get(method_key)
    if not isinstance(raw_method, dict):
        return None

    method = dict(raw_method)
    history = _load_history(
        _history_path(
            results_dir,
            summary["objective"],
            int(summary["dim"]),
            int(summary["seed"]),
            method_key,
        )
    )

    # Backfill convergence metrics from the persisted history when a run was
    # produced by an older schema or when we are rebuilding reports from CSVs.
    if "best_fitness" not in method and history:
        method["best_fitness"] = float(history[-1])
    if "iterations" not in method:
        method["iterations"] = len(history)
    if method_key == "baseline" and not history:
        # PySwarm is kept as an external timing/fitness reference only; forcing
        # fake AUC or convergence values here would make the summary misleading.
        method["auc"] = None
        method["convergence_iteration"] = None
    elif method.get("auc") is None:
        method["auc"] = _compute_auc(history)
    if method_key != "baseline" and method.get("convergence_iteration") is None:
        method["convergence_iteration"] = _compute_convergence_iteration(history, summary.get("tol"))

    timing = method.get("timing")
    if not isinstance(timing, dict):
        timing = {}
    for key in ("total_s", "eval_s", "update_s", "overhead_s", "pct_eval", "pct_update"):
        timing.setdefault(key, 0.0)
    method["timing"] = timing
    method.setdefault("strategy", label)

    if "best_fitness" not in method or "iterations" not in method:
        return None
    return method


def _load_summaries(results_dir: str) -> Tuple[List[Dict], Dict[str, int]]:
    summaries: List[Dict] = []
    stats = {
        "found": 0,
        "loaded": 0,
        "skipped_missing_core": 0,
        "skipped_invalid_method": 0,
    }
    for root, _, files in os.walk(results_dir):
        if "summary.json" not in files:
            continue
        stats["found"] += 1
        path = os.path.join(root, "summary.json")
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        if not {"objective", "dim", "seed", "v0"}.issubset(summary.keys()):
            stats["skipped_missing_core"] += 1
            continue

        normalized = dict(summary)
        normalized["execution"] = summary.get("execution") or {}
        valid = True
        for method_key, label in SUMMARY_METHODS:
            method = _normalize_method_summary(normalized, method_key, results_dir, label)
            if method_key == "v0" and method is None:
                valid = False
                break
            normalized[method_key] = method

        if not valid:
            stats["skipped_invalid_method"] += 1
            continue

        summaries.append(normalized)
        stats["loaded"] += 1
    return summaries, stats


def _filter_summaries(summaries: Iterable[Dict], objectives, dims) -> List[Dict]:
    filtered = []
    for summary in summaries:
        if objectives and summary["objective"] not in objectives:
            continue
        if dims and summary["dim"] not in dims:
            continue
        filtered.append(summary)
    return filtered

def _curve_mean_std(curves: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not curves:
        return [], []
    max_len = max(len(curve) for curve in curves)
    padded = []
    for curve in curves:
        if len(curve) < max_len:
            padded.append(curve + [curve[-1]] * (max_len - len(curve)))
        else:
            padded.append(curve)
    matrix = np.asarray(padded, dtype=float)
    return matrix.mean(axis=0).tolist(), matrix.std(axis=0).tolist()


def _safe_mean(values: List[Optional[float]]) -> Any:
    finite = [float(value) for value in values if value is not None]
    return mean(finite) if finite else ""


def _boxplot_scale_hint(box_data: List[List[float]]) -> tuple[bool, Optional[float]]:
    """Decide whether a symlog y-scale will make the boxplot more readable."""
    positive = sorted(
        float(value)
        for values in box_data
        for value in values
        if np.isfinite(value) and value > 0
    )
    if len(positive) < 2:
        return False, None
    spread = positive[-1] / positive[0]
    if spread < 1e4:
        return False, None
    return True, max(positive[0] * 10.0, 1e-12)


def _clear_analysis_outputs(out_dir: str) -> None:
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) and (
            name.endswith(".png")
            or name == "analysis_summary.csv"
        ):
            os.remove(path)


def _write_summary_csv(rows: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = [
        "objective",
        "dim",
        "method",
        "runs",
        "mean_fitness",
        "std_fitness",
        "mean_auc",
        "mean_time_s",
        "std_time_s",
        "mean_convergence_iter",
        "mean_speedup_vs_v0",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_group(summary_group: List[Dict], results_dir: str, out_dir: str, objective: str, dim: int) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Mean convergence
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method_key, label in CONVERGENCE_METHODS:
        curves = [
            _load_history(_history_path(results_dir, objective, dim, summary["seed"], method_key))
            for summary in summary_group
        ]
        curves = [curve for curve in curves if curve]
        mean_values, std_values = _curve_mean_std(curves)
        if mean_values:
            xs = np.arange(len(mean_values))
            mean_arr = np.asarray(mean_values, dtype=float)
            std_arr = np.asarray(std_values, dtype=float)
            ax.plot(xs, mean_arr, linewidth=2.2, label=label)
            ax.fill_between(xs, mean_arr - std_arr, mean_arr + std_arr, alpha=0.15)
    ax.set_title(f"Mean convergence - {objective} d={dim}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.lines and all(min(line.get_ydata()) > 0 for line in ax.lines if len(line.get_ydata()) > 0):
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{objective}_d{dim}_mean_convergence.png"), dpi=220)
    plt.close(fig)

    # Final fitness boxplot
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    box_data = []
    box_labels = []
    for method_key, label in SUMMARY_METHODS:
        values = [summary[method_key]["best_fitness"] for summary in summary_group if summary.get(method_key)]
        if values:
            box_data.append(values)
            box_labels.append(label)
    if box_data:
        try:
            ax.boxplot(
                box_data,
                tick_labels=[f"{label}\n(n={len(values)})" for label, values in zip(box_labels, box_data)],
                showmeans=True,
                widths=0.55,
            )
        except TypeError:
            ax.boxplot(
                box_data,
                labels=[f"{label}\n(n={len(values)})" for label, values in zip(box_labels, box_data)],
                showmeans=True,
                widths=0.55,
            )
        for idx, values in enumerate(box_data, start=1):
            xs = np.linspace(idx - 0.07, idx + 0.07, num=len(values))
            ax.scatter(xs, values, color="#1f77b4", alpha=0.9, s=28, zorder=3)
        flat_values = [value for values in box_data for value in values]
        if flat_values:
            y_min = min(flat_values)
            y_max = max(flat_values)
            if abs(y_max - y_min) < 1e-15:
                margin = max(abs(y_min) * 0.2, 1e-6)
                ax.set_ylim(y_min - margin, y_max + margin)
        use_symlog, linthresh = _boxplot_scale_hint(box_data)
        if use_symlog and linthresh is not None:
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.text(
                0.01,
                0.98,
                "symlog y-scale",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#555555",
            )
    ax.set_title(f"Final fitness - {objective} d={dim}")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{objective}_d{dim}_final_fitness_boxplot.png"), dpi=220)
    plt.close(fig)

    # Mean speedup
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    labels = []
    values = []
    colors = []
    baseline_mean = mean(summary["v0"]["timing"]["total_s"] for summary in summary_group)
    for method_key, label in SUMMARY_METHODS[1:]:
        method_times = [summary[method_key]["timing"]["total_s"] for summary in summary_group if summary.get(method_key)]
        if method_times:
            labels.append(label)
            values.append(baseline_mean / mean(method_times))
            colors.append("#7f7f7f" if method_key == "baseline" else {"v1": "#ff7f0e", "v2": "#2ca02c"}[method_key])
    if values:
        ax.bar(labels, values, color=colors)
    ax.set_title(f"Mean wall-clock speedup vs V0 - {objective} d={dim}")
    ax.set_ylabel("Speedup (higher is better)")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{objective}_d{dim}_speedup.png"), dpi=220)
    plt.close(fig)

    # Timing breakdown for V0/V1/V2
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    available_methods = [
        (method_key, label)
        for method_key, label in CONVERGENCE_METHODS
        if any(summary.get(method_key) for summary in summary_group)
    ]
    if available_methods:
        bottoms = np.zeros(len(available_methods), dtype=float)
        labels = [label for _, label in available_methods]
        component_values = {
            component_key: [
                mean(
                    summary[method_key]["timing"].get(component_key, 0.0)
                    for summary in summary_group
                    if summary.get(method_key)
                )
                for method_key, _ in available_methods
            ]
            for component_key, _, _ in TIMING_COMPONENTS
        }
        for component_key, component_label, color in TIMING_COMPONENTS:
            values = np.asarray(component_values[component_key], dtype=float)
            ax.bar(labels, values, bottom=bottoms, label=component_label, color=color)
            bottoms += values
        ax.set_title(f"Mean timing breakdown - {objective} d={dim}")
        ax.set_ylabel("Seconds")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{objective}_d{dim}_timing_breakdown.png"), dpi=220)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.keep_old:
        _clear_analysis_outputs(args.out_dir)

    loaded_summaries, load_stats = _load_summaries(args.results_dir)
    summaries = _filter_summaries(
        loaded_summaries,
        set(args.objective) if args.objective else None,
        set(args.dim) if args.dim else None,
    )

    if not summaries:
        print("No summaries found for the selected filters.")
        print(
            f"Found {load_stats['found']} summaries, loaded {load_stats['loaded']}, "
            f"skipped {load_stats['skipped_missing_core'] + load_stats['skipped_invalid_method']}."
        )
        return

    grouped: Dict[tuple[str, int], List[Dict]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary["objective"], summary["dim"])].append(summary)

    table_rows: List[Dict] = []
    for (objective, dim), group in sorted(grouped.items()):
        baseline_mean = mean(item["v0"]["timing"]["total_s"] for item in group)
        for method_key, method_label in SUMMARY_METHODS:
            method_values = [item[method_key] for item in group if item.get(method_key)]
            if not method_values:
                continue
            mean_time_s = mean(item["timing"]["total_s"] for item in method_values)
            table_rows.append({
                "objective": objective,
                "dim": dim,
                "method": method_label,
                "runs": len(method_values),
                "mean_fitness": mean(item["best_fitness"] for item in method_values),
                "std_fitness": float(np.std([item["best_fitness"] for item in method_values])),
                "mean_auc": _safe_mean([item.get("auc") for item in method_values]),
                "mean_time_s": mean_time_s,
                "std_time_s": float(np.std([item["timing"]["total_s"] for item in method_values])),
                "mean_convergence_iter": _safe_mean([item.get("convergence_iteration") for item in method_values]),
                "mean_speedup_vs_v0": baseline_mean / mean_time_s if mean_time_s > 0 else 0.0,
            })
        _plot_group(group, args.results_dir, args.out_dir, objective, dim)

    summary_csv_path = os.path.join(args.out_dir, "analysis_summary.csv")
    _write_summary_csv(table_rows, summary_csv_path)
    print(
        f"Loaded {load_stats['loaded']} / {load_stats['found']} summaries "
        f"(skipped_missing_core={load_stats['skipped_missing_core']}, "
        f"skipped_invalid_method={load_stats['skipped_invalid_method']})."
    )
    print(f"Saved analysis summary -> {summary_csv_path}")
    print(f"Saved plots -> {args.out_dir}")


if __name__ == "__main__":
    main()
