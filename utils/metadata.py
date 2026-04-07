"""utils.metadata

Execution and environment metadata helpers used for reproducibility.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _run_git(args: list[str], repo_root: str) -> Optional[str]:
    """Return trimmed git output or None if git metadata is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = completed.stdout.strip()
    return value or None


def collect_execution_metadata(repo_root: str | os.PathLike[str] = ".") -> Dict[str, Any]:
    """Collect lightweight execution metadata for persisted experiment results."""
    repo_root = str(Path(repo_root).resolve())
    return {
        "repo_root": repo_root,
        "git_commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "git_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
    }
