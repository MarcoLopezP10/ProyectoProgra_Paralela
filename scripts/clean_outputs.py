"""scripts.clean_outputs

Remove generated output directories while keeping logs by default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Allow direct execution as `python scripts/clean_outputs.py` from editors
# like VS Code while keeping `python -m scripts.clean_outputs` working.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean generated project outputs while keeping logs by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--remove-logs", action="store_true", help="Also remove generated logs.")
    return p.parse_args(argv)


def _safe_remove(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _remove_python_cache(root: str = ".") -> None:
    """Remove local Python bytecode caches before packaging the project."""
    for current_root, dirs, files in os.walk(root):
        for dirname in list(dirs):
            if dirname == "__pycache__":
                shutil.rmtree(os.path.join(current_root, dirname), ignore_errors=True)
        for filename in files:
            if filename.endswith(".pyc"):
                _safe_remove(os.path.join(current_root, filename))


def main(argv=None) -> None:
    args = parse_args(argv)

    targets = [
        "results/runs",
        "results/grid_search",
        "results/legacy",
        "results/benchmark_summary.csv",
        "reports/analysis",
        ".pytest_cache",
        "pso_v2.egg-info",
    ]
    if args.remove_logs:
        targets.extend(["logs/convergence", "logs/animations", "logs/pso_summary.log"])

    for target in targets:
        _safe_remove(target)

    _remove_python_cache(".")

    os.makedirs("results", exist_ok=True)
    print("Cleaned generated outputs.")


if __name__ == "__main__":
    main()
