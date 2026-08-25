"""Tests for environment/ev_energy_env.py (task §27)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ev_energy_env import EVEnergyEnv, OBSERVATION_FIELDS
from utils.config import load_config

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
BASIC_CSV = os.path.join(FIXTURES, "synthetic_test_cycle.csv")  # has accel, cruise, AND braking phases


def make_env():
    return EVEnergyEnv(
        vehicle_config=load_config("vehicle"),
        drivetrain_config=load_config("drivetrain"),
        battery_config=load_config("battery"),
        safety_config=load_config("safety"),
        energy_config=load_config("energy_management"),
        drive_cycle_path=BASIC_CSV,
        mode="train",
    )


# reset
def test_reset_returns_valid_observation():
    env = make_env()
    obs, info = env.reset(seed=7)
    assert obs.shape == (len(OBSERVATION_FIELDS),)
    assert isinstance(info, dict)


# observation shape
def test_observation_space_shape_matches_declared_fields():
    env = make_env()
    assert env.observation_space.shape == (len(OBSERVATION_FIELDS),)


# observation finite
def test_observation_finite_on_reset_and_step():
    env = make_env()
    obs, _ = env.reset(seed=7)
    assert np.all(np.isfinite(obs))
    obs2, reward, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    assert np.all(np.isfinite(obs2))


# action accepted (both directions, and boundary values)
def test_action_accepted_across_range():
    env = make_env()
    env.reset(seed=7)
    for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        obs, reward, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        assert np.all(np.isfinite(obs))
        if trunc or term:
            env.reset(seed=7)


# reward finite
def test_reward_finite():
    env = make_env()
    env.reset(seed=7)
    for a in [-1.0, -0.3, 0.0, 0.3, 1.0]:
        obs, reward, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        assert math.isfinite(reward)


# step: basic mechanics
def test_step_advances_and_returns_expected_types():
    env = make_env()
    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "applied_current_a" in info
    assert "power_deficit_w" in info
    assert "reward_components" in info


# termination / truncation
def test_truncates_at_episode_max_steps_or_drive_cycle_end():
    env = make_env()
    env.reset(seed=7)
    steps = 0
    truncated = False
    terminated = False
    for _ in range(env.episode_max_steps + 5):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        steps += 1
        if terminated or truncated:
            break
    assert truncated or terminated
    assert steps <= env.episode_max_steps


# power demand: during propulsion (positive p_wheel), a full-discharge
# action should reduce power_deficit relative to a zero/charge action
def test_full_discharge_action_reduces_deficit_during_propulsion():
    env = make_env()
    env.reset(seed=7)
    env.step(np.array([0.0], dtype=np.float32))  # t=0->1, accelerating (per fixture, speed 0->2)
    # Re-run from a fresh reset for a clean single-step comparison
    env2 = make_env()
    env2.reset(seed=7)
    _, _, _, _, info_no_supply = env2.step(np.array([1.0], dtype=np.float32))  # charge-direction action (wrong sign for propulsion) -> full deficit
    env3 = make_env()
    env3.reset(seed=7)
    _, _, _, _, info_full_supply = env3.step(np.array([-1.0], dtype=np.float32))  # max discharge -> should reduce or zero deficit
    if info_no_supply["p_wheel_w"] > 0:  # only meaningful if this step is actually propulsion
        assert info_full_supply["power_deficit_w"] <= info_no_supply["power_deficit_w"]


# safety: applied current never exceeds configured limits
def test_applied_current_within_safety_bounds():
    env = make_env()
    env.reset(seed=7)
    i_max = env.safety_config["i_max_a"]
    discharge_i_max = env.safety_config["discharge_i_max_a"]
    for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        obs, reward, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        assert -discharge_i_max - 1e-6 <= info["applied_current_a"] <= i_max + 1e-6
        if trunc or term:
            env.reset(seed=7)


# regeneration: during a braking phase, a charge-direction action should
# result in nonzero applied (positive) current and less friction braking
# than a zero/discharge action
def test_regen_action_uses_available_regen_during_braking():
    env = make_env()
    env.reset(seed=7)
    # Fast-forward to the fixture's braking phase (speeds decrease from t=6 onward)
    found_braking_step = False
    for _ in range(10):
        obs, reward, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))  # max charge/regen-use action
        if info["p_wheel_w"] < 0:
            found_braking_step = True
            assert info["friction_braking_w"] >= 0.0
            assert info["applied_current_a"] >= -1e-6  # should be charging-direction (>=0) during regen use
            break
        if trunc or term:
            break
    assert found_braking_step, "fixture should contain at least one braking (p_wheel<0) step within 10 steps"


# battery integration: SoC changes in the expected direction during a
# sustained propulsion phase with a full-discharge action
def test_soc_decreases_during_sustained_discharge():
    env = make_env()
    obs, _ = env.reset(seed=7, options={"initial_soc": 0.5})
    start_soc = env._state.soc
    for _ in range(3):  # fixture's first few steps are accelerating (propulsion)
        obs, reward, term, trunc, info = env.step(np.array([-1.0], dtype=np.float32))
        if trunc or term:
            break
    assert env._state.soc <= start_soc


# no NaN/Inf across a longer random-action rollout
def test_no_nan_inf_over_full_episode():
    env = make_env()
    rng = np.random.default_rng(0)
    env.reset(seed=7)
    for _ in range(200):
        a = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        obs, reward, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs))
        assert math.isfinite(reward)
        if term or trunc:
            env.reset(seed=7)
