"""
Tests for diagnostic logging, callback extraction, and drive-cycle validation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor

from experiments.diagnostic_ab import ComponentAccumulator
from experiments.diagnostic_driving_ppo import RolloutEpisodeTracker
from experiments.validate_drive_cycles import validate_cycle
from training.evaluate_drive_ems import STANDARD_CYCLES


def test_rollout_episode_tracker_initial_empty():
    tracker = RolloutEpisodeTracker()
    metrics = tracker.get_chunk_metrics(None)
    assert math.isnan(metrics["rollout/ep_rew_mean"])
    assert math.isnan(metrics["rollout/ep_len_mean"])
    assert metrics["rollout/episodes_in_chunk"] == 0.0
    assert metrics["rollout/total_episodes"] == 0.0


def test_rollout_episode_tracker_accumulates_episodes():
    tracker = RolloutEpisodeTracker()
    tracker.locals = {
        "infos": [
            {"episode": {"r": 100.0, "l": 200}},
            {"episode": {"r": 150.0, "l": 300}},
        ]
    }
    tracker._on_step()
    
    metrics = tracker.get_chunk_metrics(None)
    assert metrics["rollout/ep_rew_mean"] == pytest.approx(125.0)
    assert metrics["rollout/ep_len_mean"] == pytest.approx(250.0)
    assert metrics["rollout/episodes_in_chunk"] == 2.0
    assert metrics["rollout/total_episodes"] == 2.0
    
    # Test reset chunk
    tracker.reset_chunk()
    metrics_after = tracker.get_chunk_metrics(None)
    assert math.isnan(metrics_after["rollout/ep_rew_mean"])
    assert metrics_after["rollout/total_episodes"] == 2.0


def test_component_accumulator_tracks_rewards_and_episodes():
    acc = ComponentAccumulator()
    acc.locals = {
        "infos": [
            {
                "reward_components": {"progress": 0.5, "thermal": 0.1},
                "episode": {"r": 250.0, "l": 500},
            }
        ]
    }
    acc._on_step()
    
    means = acc.means()
    assert means["progress"] == pytest.approx(0.5)
    assert means["thermal"] == pytest.approx(0.1)
    
    rollout = acc.get_rollout_metrics(None)
    assert rollout["rollout/ep_rew_mean"] == pytest.approx(250.0)
    assert rollout["rollout/ep_len_mean"] == pytest.approx(500.0)
    assert rollout["rollout/total_episodes"] == 1.0


def test_validate_standard_drive_cycles_exist_and_pass():
    for cycle_id, csv_path in STANDARD_CYCLES.items():
        meta_path = csv_path.replace("cycle.csv", "metadata.yaml")
        res = validate_cycle(csv_path, meta_path)
        assert bool(res["is_monotonic"]) is True
        assert bool(res["is_regular"]) is True
        assert bool(res["is_non_negative"]) is True
        assert bool(res["is_finite"]) is True
        assert res["duration_s"] > 0
        assert res["distance_km"] > 0.0
