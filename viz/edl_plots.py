"""Minimal visual reports for Economic Load Dispatch runs."""

from __future__ import annotations

import os
import tempfile
import warnings
from math import ceil
from typing import Any, Optional

_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "edl-plot-cache")
os.makedirs(_CACHE_ROOT, exist_ok=True)
MPLCONFIGDIR = os.path.join(_CACHE_ROOT, "matplotlib")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from utils.methods import SUMMARY_METHOD_SPECS

METHOD_SPECS = [
    (spec.key, spec.label, spec.color, spec.has_history)
    for spec in SUMMARY_METHOD_SPECS
]

VARIANT_ORDER = {
    "edl_1": 0,
    "edl_2": 1,
    "edl_3": 2,
    "edl_4": 3,
}

VARIANT_COLORS = {
    "edl_1": "#4c78a8",
    "edl_2": "#f58518",
    "edl_3": "#54a24b",
    "edl_4": "#e45756",
}

METHOD_STYLES = {
    "v0": {"linestyle": "-", "marker": "o"},
    "v1": {"linestyle": "--", "marker": "s"},
    "v2": {"linestyle": ":", "marker": "^"},
    "v3": {"linestyle": "-.", "marker": "D"},
    "baseline": {"linestyle": "--", "marker": "x"},
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _history_path(run_dir: str, method_key: str) -> str:
    return os.path.join(run_dir, f"history_{method_key}.csv")


def _load_history(path: str) -> list[float]:
    if not os.path.exists(path):
        return []
    values: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if "best_fitness" not in header:
            return []
        best_idx = header.index("best_fitness")
        for line in f:
            cols = line.strip().split(",")
            if best_idx >= len(cols) or not cols[best_idx]:
                continue
            values.append(float(cols[best_idx]))
    return values


def _available_methods(summary: dict[str, Any]) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []
    for method_key, label, color, has_history in METHOD_SPECS:
        payload = summary.get(method_key)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            continue
        methods.append(
            {
                "method_key": method_key,
                "label": label,
                "color": color,
                "has_history": has_history,
                "payload": payload,
            }
        )
    return methods


def _winner_method(summary: dict[str, Any]) -> Optional[dict[str, Any]]:
    methods = _available_methods(summary)
    if not methods:
        return None

    winner_label = summary.get("winner")
    for method in methods:
        if method["payload"].get("strategy") == winner_label or method["label"] == winner_label:
            return method

    def sort_key(item: dict[str, Any]) -> tuple[float, float]:
        payload = item["payload"]
        return (
            float(payload.get("best_fitness", float("inf"))),
            float(payload.get("timing", {}).get("total_s", float("inf"))),
        )

    return min(methods, key=sort_key)


def _variant_sort_key(entry: dict[str, Any]) -> tuple[int, int]:
    summary = entry["summary"]
    return (
        VARIANT_ORDER.get(str(summary.get("variant", "")), 999),
        int(summary.get("seed", 0)),
    )


def _generator_tick_step(n_generators: int) -> int:
    return max(1, int(np.ceil(n_generators / 12)))


def _save_fig(fig: plt.Figure, out_path: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fbfbfc")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def _subplot_grid(n_items: int) -> tuple[int, int]:
    if n_items <= 1:
        return (1, 1)
    if n_items <= 4:
        return (2, 2)
    cols = 2
    rows = ceil(n_items / cols)
    return rows, cols


def _format_metric_value(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 10000:
        return f"{value:,.0f}"
    if abs_value >= 100:
        return f"{value:,.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    if abs_value >= 1e-2:
        return f"{value:.4f}"
    return f"{value:.2e}"


def plot_case_convergence_overview(entries: list[dict[str, Any]], out_dir: str) -> Optional[str]:
    ordered = sorted(entries, key=_variant_sort_key)
    if not ordered:
        return None

    rows, cols = _subplot_grid(len(ordered))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4.3), squeeze=False)
    axes_flat = list(axes.flat)
    any_curve = False

    for ax, entry in zip(axes_flat, ordered):
        summary = entry["summary"]
        run_dir = entry["run_dir"]
        methods = _available_methods(summary)
        curves_found = False
        final_values: list[float] = []
        for method in methods:
            if not method["has_history"]:
                continue
            history = _load_history(_history_path(run_dir, method["method_key"]))
            if not history:
                continue
            curves_found = True
            any_curve = True
            final_values.append(history[-1])
            xs = np.arange(1, len(history) + 1)
            style = METHOD_STYLES.get(method["method_key"], {})
            ax.plot(
                xs,
                history,
                linewidth=2.1,
                color=method["color"],
                label=method["label"],
                linestyle=style.get("linestyle", "-"),
            )
            ax.scatter(
                [xs[-1]],
                [history[-1]],
                color=method["color"],
                marker=style.get("marker", "o"),
                s=32,
                zorder=4,
            )

        ax.set_title(f"{summary['variant']} ({summary['variant_label']})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness")
        _style_axis(ax)
        if curves_found:
            if all(value > 0 for value in final_values):
                ax.set_yscale("log")
            ax.legend(fontsize=8)
            winner = _winner_method(summary)
            if winner is not None:
                payload = winner["payload"]
                ax.text(
                    0.03,
                    0.03,
                    f"Winner: {winner['label']}\nBest: {_format_metric_value(payload.get('best_fitness', 0.0) or 0.0)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
                )
        else:
            ax.text(0.5, 0.5, "No convergence history", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

        baseline = summary.get("baseline")
        if isinstance(baseline, dict) and baseline.get("status") == "ok":
            baseline_fitness = baseline.get("best_fitness")
            if baseline_fitness is not None:
                ax.axhline(
                    float(baseline_fitness),
                    linewidth=1.8,
                    linestyle="--",
                    color="#7f7f7f",
                    alpha=0.9,
                    label="PySwarm baseline",
                )
                if curves_found:
                    ax.legend(fontsize=8)

    for ax in axes_flat[len(ordered) :]:
        ax.remove()

    case_name = ordered[0]["summary"]["case_name"]
    seed = ordered[0]["summary"]["seed"]
    fig.suptitle(f"EDL convergence overview - {case_name} | seed {seed}", fontsize=14)
    if not any_curve:
        plt.close(fig)
        return None
    return _save_fig(fig, os.path.join(out_dir, "convergence_overview.png"))


def plot_case_dispatch_overview(entries: list[dict[str, Any]], out_dir: str) -> Optional[str]:
    ordered = sorted(entries, key=_variant_sort_key)
    winners = [(entry, _winner_method(entry["summary"])) for entry in ordered]
    winners = [(entry, winner) for entry, winner in winners if winner is not None]
    if not winners:
        return None

    rows, cols = _subplot_grid(len(winners))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.2, rows * 4.8), squeeze=False)
    axes_flat = list(axes.flat)
    y_max = max(
        float(np.max(np.asarray(winner["payload"].get("best_position") or [0.0], dtype=float)))
        for _, winner in winners
    )

    for ax, (entry, winner) in zip(axes_flat, winners):
        summary = entry["summary"]
        generator_names = list(summary.get("generator_names") or [])
        powers = np.asarray(winner["payload"].get("best_position") or [], dtype=float)
        if len(powers) == 0:
            ax.text(0.5, 0.5, "No dispatch available", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        x = np.arange(len(powers), dtype=float)
        color = VARIANT_COLORS.get(str(summary["variant"]), winner["color"])
        if len(powers) <= 12:
            bars = ax.bar(x, powers, color=color, width=0.75)
            labels = [_format_metric_value(value) for value in powers]
            ax.bar_label(bars, labels=labels, padding=2, fontsize=7, color="#333333")
        else:
            ax.plot(x, powers, color=color, linewidth=2.2, marker="o", markersize=3.2)
            ax.fill_between(x, powers, color=color, alpha=0.16)

        ax.set_title(f"{summary['variant']} ({summary['variant_label']})")
        ax.set_ylabel("Power (MW)")
        ax.set_ylim(0.0, y_max * 1.16 if y_max > 0 else 1.0)
        _style_axis(ax)
        step = _generator_tick_step(len(powers))
        tick_idx = np.arange(0, len(powers), step)
        labels = [generator_names[idx] if idx < len(generator_names) else f"G{idx + 1}" for idx in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.text(
            0.02,
            0.98,
            f"{winner['label']}\nTotal: {_format_metric_value(np.sum(powers))} MW",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
        )

    for ax in axes_flat[len(winners) :]:
        ax.remove()

    case_name = winners[0][0]["summary"]["case_name"]
    seed = winners[0][0]["summary"]["seed"]
    fig.suptitle(f"EDL dispatch overview - {case_name} | seed {seed}", fontsize=14)
    return _save_fig(fig, os.path.join(out_dir, "dispatch_overview.png"))


def plot_case_summary_dashboard(entries: list[dict[str, Any]], out_dir: str) -> Optional[str]:
    ordered = sorted(entries, key=_variant_sort_key)
    rows = []
    for entry in ordered:
        summary = entry["summary"]
        winner = _winner_method(summary)
        if winner is None:
            continue
        payload = winner["payload"]
        rows.append(
            {
                "variant": summary["variant"],
                "variant_label": summary["variant_label"],
                "winner": winner["label"],
                "fitness": float(payload.get("best_fitness", 0.0) or 0.0),
                "time_s": float(payload.get("timing", {}).get("total_s", 0.0)),
                "losses": float(payload.get("transmission_loss", 0.0) or 0.0),
                "balance_error": float(payload.get("power_balance_error", 0.0) or 0.0),
            }
        )
    if not rows:
        return None

    labels = [f"{row['variant']}\n{row['variant_label']}" for row in rows]
    x = np.arange(len(labels))
    colors = [VARIANT_COLORS.get(row["variant"], "#4c78a8") for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    specs = [
        ("fitness", "Winner fitness", False),
        ("time_s", "Winner time (s)", False),
        ("losses", "Winner losses", False),
        ("balance_error", "Winner |Balance|", True),
    ]
    for ax, (key, title, use_log) in zip(axes.flat, specs):
        values = np.asarray([row[key] for row in rows], dtype=float)
        bars = ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        _style_axis(ax)
        if use_log and np.all(values > 0):
            ax.set_yscale("log")
        ax.bar_label(
            bars,
            labels=[_format_metric_value(value) for value in values],
            padding=3,
            fontsize=8,
            color="#333333",
        )
        ax.margins(y=0.16)
        if not use_log and np.all(values < 1000):
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: _format_metric_value(value)))

    case_name = ordered[0]["summary"]["case_name"]
    seed = ordered[0]["summary"]["seed"]
    fig.suptitle(f"EDL summary dashboard - {case_name} | seed {seed}", fontsize=15, y=1.01)
    winners_text = " | ".join(f"{row['variant']} -> {row['winner']}" for row in rows)
    fig.text(
        0.5,
        0.952,
        f"Winners: {winners_text}",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    return _save_fig(fig, os.path.join(out_dir, "summary_dashboard.png"))


def generate_case_seed_report(entries: list[dict[str, Any]], out_dir: str) -> list[str]:
    _ensure_dir(out_dir)
    artifacts = [
        plot_case_convergence_overview(entries, out_dir),
        plot_case_dispatch_overview(entries, out_dir),
        plot_case_summary_dashboard(entries, out_dir),
    ]
    return [path for path in artifacts if path]


__all__ = ["generate_case_seed_report"]
