"""
Logging utility: append-only CSV metric logging, plus a thin wrapper
for creating per-run directories with reproducibility artifacts
(config snapshot + git commit hash).
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def _next_run_index(runs_root: str) -> int:
    """Return the next safe numeric run index: max(existing numeric suffixes) + 1.

    Fixes the previous `len(existing) + 1` logic, which used the COUNT of
    run_* directories rather than the highest number present. That is wrong
    whenever a run number is missing (deleted, renamed, or created out of
    band) or non-numeric: e.g. existing = [run_001, run_002, run_005] has
    len=3 -> next_idx=4 -> "run_004", which COLLIDES with an already-planned
    or manually created run_004, or silently reuses a number. Non-run
    directories (anything not matching ^run_(\\d+)$, including malformed
    names like "run_abc" or "run_01_backup") are ignored entirely rather
    than counted, so they can't perturb the index either.
    """
    max_idx = 0
    if os.path.isdir(runs_root):
        for d in os.listdir(runs_root):
            if not os.path.isdir(os.path.join(runs_root, d)):
                continue
            m = _RUN_DIR_RE.match(d)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


class CSVLogger:
    """Append dict rows to a CSV file, writing the header on first use."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._fieldnames: Optional[List[str]] = None
        if os.path.isfile(filepath):
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fieldnames = header

    def log(self, row: Dict[str, Any]) -> None:
        write_header = self._fieldnames is None
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "no-git-repo"


def create_run_dir(runs_root: str, run_name: Optional[str] = None) -> str:
    """Create runs/run_XXX/ with config/, checkpoints/, tensorboard/, plots/ subdirs."""
    os.makedirs(runs_root, exist_ok=True)
    if run_name is None:
        run_name = f"run_{_next_run_index(runs_root):03d}"

    run_dir = os.path.join(runs_root, run_name)
    # Guard against accidental overwrite: if the caller passed an explicit
    # run_name (or auto-numbering somehow raced) that already has content,
    # fail loudly instead of silently reusing/merging into it.
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise FileExistsError(
            f"Run directory '{run_dir}' already exists and is non-empty. "
            "Pass a different --run-name or remove the existing directory."
        )

    for sub in ("config", "checkpoints", "tensorboard", "plots"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    with open(os.path.join(run_dir, "git_commit_hash.txt"), "w") as f:
        f.write(get_git_commit_hash() + "\n")
    with open(os.path.join(run_dir, "created_at.txt"), "w") as f:
        f.write(datetime.now(timezone.utc).isoformat() + "\n")

    return run_dir
