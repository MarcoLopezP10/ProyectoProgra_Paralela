"""Generate a minimal visual report for saved Economic Load Dispatch runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viz.edl_plots import generate_case_seed_report

VARIANT_ORDER = {
    "edl_1": 0,
    "edl_2": 1,
    "edl_3": 2,
    "edl_4": 3,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the essential visual report for saved EDL runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="results/edl",
        help="Directory containing saved EDL summaries and histories.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/edl",
        help="Directory for generated EDL reports.",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=None,
        metavar="CASE",
        help="Optional subset of case_name values to analyze.",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=None,
        metavar="EDL",
        help="Optional subset of variants to analyze.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=None,
        metavar="S",
        help="Optional subset of seeds to analyze.",
    )
    return parser.parse_args(argv)


def _load_entries(results_dir: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for root, _, files in os.walk(results_dir):
        if "summary.json" not in files:
            continue
        summary_path = os.path.join(root, "summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        entries.append(
            {
                "summary": summary,
                "run_dir": root,
                "summary_path": summary_path,
            }
        )
    return entries


def _filter_entries(
    entries: list[dict[str, Any]],
    case_names: Optional[set[str]],
    variants: Optional[set[str]],
    seeds: Optional[set[int]],
) -> list[dict[str, Any]]:
    filtered = []
    for entry in entries:
        summary = entry["summary"]
        if case_names and summary.get("case_name") not in case_names:
            continue
        if variants and summary.get("variant") not in variants:
            continue
        if seeds and int(summary.get("seed", -1)) not in seeds:
            continue
        filtered.append(entry)
    return filtered


def _sort_key(entry: dict[str, Any]) -> tuple[str, int, int]:
    summary = entry["summary"]
    return (
        str(summary.get("case_name", "")),
        int(summary.get("seed", 0)),
        VARIANT_ORDER.get(str(summary.get("variant", "")), 999),
    )


def analyze_edl_results(
    results_dir: str = "results/edl",
    out_dir: str = "reports/edl",
    case_names: Optional[list[str]] = None,
    variants: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    quiet: bool = False,
) -> dict[str, Any]:
    entries = _load_entries(results_dir)
    filtered = _filter_entries(
        entries,
        set(case_names) if case_names else None,
        set(variants) if variants else None,
        set(seeds) if seeds else None,
    )
    if not filtered:
        if not quiet:
            print("No EDL summaries found for the selected filters.")
        return {
            "entries": 0,
            "artifacts": [],
            "report_dirs": [],
        }

    index_path = os.path.join(out_dir, "analysis_index.csv")
    if os.path.exists(index_path):
        os.remove(index_path)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in filtered:
        summary = entry["summary"]
        grouped[(str(summary["case_name"]), int(summary["seed"]))].append(entry)

    artifacts: list[str] = []
    report_dirs: list[str] = []
    for (case_name, seed), group_entries in sorted(grouped.items()):
        case_seed_dir = os.path.join(out_dir, case_name, f"seed_{seed}")
        if os.path.exists(case_seed_dir):
            shutil.rmtree(case_seed_dir)
        ordered_entries = sorted(group_entries, key=_sort_key)
        os.makedirs(case_seed_dir, exist_ok=True)
        artifacts.extend(generate_case_seed_report(ordered_entries, case_seed_dir))
        report_dirs.append(case_seed_dir)

    if not quiet:
        print(f"Analyzed {len(filtered)} saved EDL run(s).")
        print(f"Saved minimal reports -> {out_dir}")

    return {
        "entries": len(filtered),
        "artifacts": artifacts,
        "report_dirs": report_dirs,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    analyze_edl_results(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        case_names=args.case,
        variants=args.variant,
        seeds=args.seed,
    )


if __name__ == "__main__":
    main()
