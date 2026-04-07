from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.metadata import collect_execution_metadata


def test_collect_execution_metadata_contains_expected_keys():
    metadata = collect_execution_metadata(".")
    expected_keys = {
        "repo_root",
        "git_commit",
        "git_branch",
        "python_version",
        "python_implementation",
        "platform",
        "system",
        "release",
        "machine",
        "processor",
        "hostname",
        "cpu_count",
        "executable",
    }
    assert expected_keys.issubset(metadata.keys())
    assert metadata["repo_root"]
    assert metadata["python_version"]
