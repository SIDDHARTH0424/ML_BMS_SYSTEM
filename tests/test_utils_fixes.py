"""
Regression tests for two audit fixes (see audit/ISSUES.md):

  ISSUE-001: utils/logger.create_run_dir used len(existing)+1 for run
             numbering, which breaks under gaps/non-numeric names.
  ISSUE-002: utils/metrics.aggregate_runs let a single NaN value poison
             mean/std for every run's value of that metric key.
"""
from __future__ import annotations

import math
import os

import pytest

from utils.logger import create_run_dir
from utils.metrics import aggregate_runs


# --------------------------------------------------------------------- #
# ISSUE-001: run-directory numbering
# --------------------------------------------------------------------- #
def test_no_runs_creates_run_001(tmp_path):
    run_dir = create_run_dir(str(tmp_path))
    assert os.path.basename(run_dir) == "run_001"


def test_single_existing_run_increments(tmp_path):
    create_run_dir(str(tmp_path))  # run_001
    run_dir = create_run_dir(str(tmp_path))
    assert os.path.basename(run_dir) == "run_002"


def test_gap_in_run_numbers_uses_max_plus_one(tmp_path):
    # run_001 and run_003 exist (run_002 missing) -> next should be run_004,
    # NOT run_003 (len=2+1) and NOT a collision with run_003.
    create_run_dir(str(tmp_path), run_name="run_001")
    create_run_dir(str(tmp_path), run_name="run_003")
    run_dir = create_run_dir(str(tmp_path))
    assert os.path.basename(run_dir) == "run_004"


def test_scattered_run_numbers_uses_max_plus_one(tmp_path):
    # run_001, run_002, run_005 exist -> next should be run_006 (max=5+1),
    # not run_004 (len=3+1, the original buggy behavior).
    for name in ("run_001", "run_002", "run_005"):
        create_run_dir(str(tmp_path), run_name=name)
    run_dir = create_run_dir(str(tmp_path))
    assert os.path.basename(run_dir) == "run_006"


def test_non_run_directories_are_ignored(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "not_a_run"))
    os.makedirs(os.path.join(str(tmp_path), "run_backup_old"))  # malformed, no trailing digits-only
    create_run_dir(str(tmp_path), run_name="run_002")
    run_dir = create_run_dir(str(tmp_path))
    assert os.path.basename(run_dir) == "run_003"


def test_malformed_run_names_do_not_crash_or_count(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "run_abc"))  # non-numeric suffix
    run_dir = create_run_dir(str(tmp_path))
    assert os.path.basename(run_dir) == "run_001"


def test_existing_nonempty_run_dir_refuses_overwrite(tmp_path):
    create_run_dir(str(tmp_path), run_name="run_001")
    with pytest.raises(FileExistsError):
        create_run_dir(str(tmp_path), run_name="run_001")


# --------------------------------------------------------------------- #
# ISSUE-002: NaN-safe aggregation
# --------------------------------------------------------------------- #
def test_aggregate_no_nans_matches_plain_mean_std():
    dicts = [{"time_to_target_s": 100.0}, {"time_to_target_s": 200.0}]
    result = aggregate_runs(dicts)
    assert result["time_to_target_s"]["mean"] == pytest.approx(150.0)
    assert result["time_to_target_s"]["valid_runs"] == 2
    assert result["time_to_target_s"]["failed_runs"] == 0


def test_single_nan_does_not_contaminate_other_runs():
    dicts = [
        {"time_to_target_s": 1800.0},
        {"time_to_target_s": 1900.0},
        {"time_to_target_s": float("nan")},  # did not reach target
    ]
    result = aggregate_runs(dicts)
    stats = result["time_to_target_s"]
    assert stats["mean"] == pytest.approx(1850.0)
    assert stats["valid_runs"] == 2
    assert stats["failed_runs"] == 1
    assert not math.isnan(stats["mean"])


def test_all_nan_reports_nan_explicitly_not_a_crash():
    dicts = [{"time_to_target_s": float("nan")}, {"time_to_target_s": float("nan")}]
    result = aggregate_runs(dicts)
    stats = result["time_to_target_s"]
    assert math.isnan(stats["mean"])
    assert math.isnan(stats["std"])
    assert stats["valid_runs"] == 0
    assert stats["failed_runs"] == 2


def test_empty_input_returns_empty_dict():
    assert aggregate_runs([]) == {}
