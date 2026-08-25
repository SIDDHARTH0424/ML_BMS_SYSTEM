"""
Regression tests for ISSUE-008 (audit/ISSUES.md): checkpoint selection
must not be decided by final_soc/mean_final_soc alone. Tests the pure
_select_best lexicographic logic directly (no PPO model / env needed).
"""
from __future__ import annotations

from training.select_best_checkpoint import _select_best


def _metrics(**overrides):
    base = {
        "target_reached_rate": 1.0,
        "safety_interventions": 5.0,
        "charging_time_s": 3000.0,
        "peak_temperature_c": 45.0,
        "energy_efficiency": 0.9,
        "final_soc": 0.95,
    }
    base.update(overrides)
    return base


def test_higher_final_soc_does_not_win_if_worse_on_everything_else():
    # Checkpoint C (96% SoC) is worse on safety/thermal/time than
    # checkpoint B (95% SoC) — B should win, unlike the old
    # mean_final_soc-only selector.
    candidates = [
        ("ckpt_B.zip", _metrics(final_soc=0.95, safety_interventions=2.0,
                                 charging_time_s=2200.0, peak_temperature_c=40.0)),
        ("ckpt_C.zip", _metrics(final_soc=0.96, safety_interventions=50.0,
                                 charging_time_s=3000.0, peak_temperature_c=48.0)),
    ]
    best_path, best_metrics = _select_best(candidates)
    assert best_path == "ckpt_B.zip"


def test_target_reached_rate_dominates_all_other_criteria():
    candidates = [
        ("ckpt_low_reach.zip", _metrics(target_reached_rate=0.5, charging_time_s=1000.0,
                                         peak_temperature_c=30.0)),
        ("ckpt_full_reach.zip", _metrics(target_reached_rate=1.0, charging_time_s=5000.0,
                                          peak_temperature_c=55.0)),
    ]
    best_path, _ = _select_best(candidates)
    assert best_path == "ckpt_full_reach.zip"


def test_fewer_safety_interventions_breaks_tie_on_target_reached():
    candidates = [
        ("ckpt_more_interventions.zip", _metrics(safety_interventions=20.0)),
        ("ckpt_fewer_interventions.zip", _metrics(safety_interventions=1.0)),
    ]
    best_path, _ = _select_best(candidates)
    assert best_path == "ckpt_fewer_interventions.zip"


def test_charging_time_breaks_tie_after_reach_and_safety():
    candidates = [
        ("ckpt_slow.zip", _metrics(charging_time_s=4000.0)),
        ("ckpt_fast.zip", _metrics(charging_time_s=2000.0)),
    ]
    best_path, _ = _select_best(candidates)
    assert best_path == "ckpt_fast.zip"


def test_single_candidate_returned_unchanged():
    candidates = [("only.zip", _metrics())]
    best_path, best_metrics = _select_best(candidates)
    assert best_path == "only.zip"
    assert best_metrics == candidates[0][1]


def test_fully_tied_candidates_fall_back_to_first():
    candidates = [("first.zip", _metrics()), ("second.zip", _metrics())]
    best_path, _ = _select_best(candidates)
    assert best_path == "first.zip"
