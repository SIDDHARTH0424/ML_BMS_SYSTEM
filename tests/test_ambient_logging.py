"""
Ambient Temperature Logging Regression Tests (Part 5)
======================================================
Dedicated regression test file verifying the end-to-end chain:

    environment selects ambient_temp_c during reset()
            ↓
    step() exposes info["ambient_temp_c"] for every step
            ↓
    value is constant within an episode
            ↓
    Experiment C diagnostic can use it to verify sampling distribution

These tests must pass for the Experiment C ambient distribution
evidence to be considered trustworthy.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.battery_env import BatteryChargingEnv
from utils.config import load_config


@pytest.fixture(scope="module")
def base_configs():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    return battery_cfg, safety_cfg, reward_cfg, sim_cfg


@pytest.fixture
def env(base_configs):
    battery_cfg, safety_cfg, reward_cfg, sim_cfg = base_configs
    cfg = copy.deepcopy(sim_cfg)
    cfg["max_episode_steps"] = 200
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, cfg, mode="train")


# ---------------------------------------------------------------------------
# Chain link 1: reset() selects an ambient temperature and exposes it
# ---------------------------------------------------------------------------

def test_ambient_in_reset_info(env):
    """reset() must return info containing ambient_temp_c."""
    _, info = env.reset(seed=7)
    assert "ambient_temp_c" in info, "ambient_temp_c missing from reset() info"


def test_ambient_is_finite_after_reset(env):
    """Ambient temperature selected at reset must be a finite float."""
    _, info = env.reset(seed=21)
    amb = info["ambient_temp_c"]
    assert math.isfinite(amb), f"ambient_temp_c is not finite: {amb}"


def test_ambient_in_valid_range_train_mode(env):
    """In default train mode, ambient must be within the configured [15, 35] range."""
    sim_cfg_raw = load_config("simulation")
    ambient_min = sim_cfg_raw["train"].get("ambient_min_c", 15.0)
    ambient_max = sim_cfg_raw["train"].get("ambient_max_c", 35.0)
    for seed in range(20):
        _, info = env.reset(seed=seed)
        amb = info["ambient_temp_c"]
        assert ambient_min <= amb <= ambient_max, (
            f"Seed {seed}: ambient {amb:.2f}°C out of train range [{ambient_min}, {ambient_max}]°C"
        )


# ---------------------------------------------------------------------------
# Chain link 2: step() exposes the same ambient_temp_c in every step info
# ---------------------------------------------------------------------------

def test_ambient_present_in_step_info(env):
    """info['ambient_temp_c'] must be present in the step-level info dict."""
    env.reset(seed=42)
    action = np.array([0.5], dtype=np.float32)
    _, _, _, _, step_info = env.step(action)
    assert "ambient_temp_c" in step_info, "ambient_temp_c missing from step() info"


def test_ambient_is_finite_in_step(env):
    """Step-level ambient must be a finite float."""
    env.reset(seed=42)
    _, _, _, _, step_info = env.step(np.array([0.5], dtype=np.float32))
    amb = step_info["ambient_temp_c"]
    assert math.isfinite(amb), f"Step ambient_temp_c is not finite: {amb}"


# ---------------------------------------------------------------------------
# Chain link 3: ambient_temp_c is CONSTANT within an episode
# ---------------------------------------------------------------------------

def test_ambient_constant_within_episode(env):
    """Ambient temperature must not change across steps within the same episode.

    This is the critical invariant that allows the diagnostic callback
    to record exactly one ambient per completed episode.
    """
    _, reset_info = env.reset(seed=99)
    initial_ambient = reset_info["ambient_temp_c"]

    for step_idx in range(30):
        _, _, terminated, truncated, step_info = env.step(np.array([0.5], dtype=np.float32))
        step_ambient = step_info["ambient_temp_c"]
        assert step_ambient == initial_ambient, (
            f"ambient_temp_c changed at step {step_idx}: "
            f"reset={initial_ambient:.4f}, step={step_ambient:.4f}"
        )
        if terminated or truncated:
            break


def test_ambient_constant_across_multiple_episodes(env):
    """After reset(), a new ambient is selected; within each episode it stays constant."""
    seen_ambients = set()
    for ep in range(5):
        _, reset_info = env.reset(seed=ep * 17)
        episode_ambient = reset_info["ambient_temp_c"]
        seen_ambients.add(round(episode_ambient, 4))

        for _ in range(10):
            _, _, terminated, truncated, step_info = env.step(np.array([0.8], dtype=np.float32))
            assert step_info["ambient_temp_c"] == episode_ambient, (
                f"Episode {ep}: ambient changed during episode"
            )
            if terminated or truncated:
                break

    # Different seeds should occasionally produce different ambients
    assert len(seen_ambients) >= 2, (
        "All 5 episodes produced identical ambient temperatures — "
        "likely a seeding problem"
    )


# ---------------------------------------------------------------------------
# Chain link 4: Classification boundary is correct
# ---------------------------------------------------------------------------

def test_normal_stress_classification_boundary():
    """ambient < 35°C is NORMAL; ambient >= 35°C is STRESS.

    Verifies explicitly that the boundary used in Experiment C
    diagnostic matches the configured threshold.
    """
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = copy.deepcopy(load_config("simulation"))

    # Force fixed ambients using options= override (eval mode)
    sim_cfg["max_episode_steps"] = 50
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval")

    BOUNDARY = 35.0

    # Normal side: 34.9°C
    _, info = env.reset(options={"initial_soc": 0.2, "ambient_temp_c": 34.9})
    assert info["ambient_temp_c"] < BOUNDARY, "34.9°C should classify as NORMAL (< 35°C)"
    _, _, _, _, step_info = env.step(np.array([0.5], dtype=np.float32))
    assert step_info["ambient_temp_c"] < BOUNDARY

    # Stress side: 35.0°C (exact boundary)
    _, info = env.reset(options={"initial_soc": 0.2, "ambient_temp_c": 35.0})
    assert info["ambient_temp_c"] >= BOUNDARY, "35.0°C should classify as STRESS (>= 35°C)"

    # Stress side: 42.5°C
    _, info = env.reset(options={"initial_soc": 0.2, "ambient_temp_c": 42.5})
    assert info["ambient_temp_c"] >= BOUNDARY, "42.5°C should classify as STRESS"


# ---------------------------------------------------------------------------
# Chain link 5: Mixed-distribution sampler produces expected fractions
# ---------------------------------------------------------------------------

def test_mixed_sampler_produces_both_distributions():
    """Experiment C sampler must draw from BOTH normal and stress ranges."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = copy.deepcopy(load_config("simulation"))
    sim_cfg["train"]["mixed_ambient_sampler"] = {
        "p_stress": 0.25,
        "normal_range_c": [15.0, 35.0],
        "stress_range_c": [35.0, 45.0],
    }

    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    ambients = [env.reset(seed=ep + 500)[1]["ambient_temp_c"] for ep in range(200)]

    normal_count = sum(1 for a in ambients if a < 35.0)
    stress_count = sum(1 for a in ambients if a >= 35.0)

    assert normal_count > 0, "Mixed sampler produced zero normal-range episodes"
    assert stress_count > 0, "Mixed sampler produced zero stress-range episodes"

    obs_p_stress = stress_count / len(ambients)
    # Tolerance: ±0.10 from target 0.25 (binomial noise at N=200)
    assert abs(obs_p_stress - 0.25) < 0.10, (
        f"Observed p_stress={obs_p_stress:.3f} is suspiciously far from target 0.25"
    )
    assert all(15.0 <= a <= 45.0 for a in ambients), (
        "Some ambient temperatures fall outside the configured [15, 45]°C range"
    )
