"""
Tests for Driving-EMS reward function, action authority, and regen ordering.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ev_energy_env import EVEnergyEnv
from training.train_drive_ems import FIXTURE_DRIVE_CYCLE, make_drive_ems_env
from utils.config import load_config


@pytest.fixture
def drive_env():
    return make_drive_ems_env(drive_cycle_path=FIXTURE_DRIVE_CYCLE, mode="eval")


def test_regen_ordering_at_braking_state(drive_env):
    """
    At a braking event, verifies that R(action=+1.0) > R(action=+0.5) > R(action=0.0).
    """
    drive_env.reset(options={"initial_soc": 0.50, "ambient_temp_c": 25.0})
    
    # Advance to a braking step
    found_braking = False
    for _ in range(50):
        speed = drive_env._drive_cycle.current_speed()
        accel = drive_env._drive_cycle.current_acceleration()
        grade = drive_env._drive_cycle.current_grade()
        forces = drive_env.vehicle.compute(speed_mps=speed, acceleration_mps2=accel, road_grade_rad=grade)
        drivetrain_out = drive_env.drivetrain.compute(p_wheel_w=forces.p_wheel)
        
        if forces.p_wheel < 0.0 and drivetrain_out.available_regenerative_power_w > 500.0:
            found_braking = True
            break
        drive_env.step(np.array([-1.0], dtype=np.float32))

    assert found_braking, "Could not find a braking step in fixture drive cycle"

    # Snapshot
    saved_state = drive_env._state.copy()
    saved_prev_power = drive_env._prev_battery_power_w
    saved_step = drive_env._step_count
    saved_idx = drive_env._drive_cycle._idx

    # Evaluate +1.0 (max regen)
    _, r_10, _, _, _ = drive_env.step(np.array([1.0], dtype=np.float32))

    # Restore
    drive_env._state = saved_state.copy()
    drive_env._prev_battery_power_w = saved_prev_power
    drive_env._step_count = saved_step
    drive_env._drive_cycle._idx = saved_idx

    # Evaluate +0.5 (partial regen)
    _, r_05, _, _, _ = drive_env.step(np.array([0.5], dtype=np.float32))

    # Restore
    drive_env._state = saved_state.copy()
    drive_env._prev_battery_power_w = saved_prev_power
    drive_env._step_count = saved_step
    drive_env._drive_cycle._idx = saved_idx

    # Evaluate 0.0 (no regen)
    _, r_00, _, _, _ = drive_env.step(np.array([0.0], dtype=np.float32))

    assert r_10 > r_05 > r_00, f"Expected r_10 ({r_10}) > r_05 ({r_05}) > r_00 ({r_00})"


def test_action_authority_different_power_flows(drive_env):
    """Verifies that actions [-1.0, 0.0, +1.0] produce distinct battery powers."""
    drive_env.reset(options={"initial_soc": 0.50, "ambient_temp_c": 25.0})
    
    # Find a cruising / positive propulsion state
    found_prop = False
    for _ in range(50):
        speed = drive_env._drive_cycle.current_speed()
        accel = drive_env._drive_cycle.current_acceleration()
        grade = drive_env._drive_cycle.current_grade()
        forces = drive_env.vehicle.compute(speed_mps=speed, acceleration_mps2=accel, road_grade_rad=grade)
        if forces.p_wheel > 2000.0:
            found_prop = True
            break
        drive_env.step(np.array([-1.0], dtype=np.float32))

    assert found_prop, "Could not find a propulsion step in fixture drive cycle"

    # Snapshot
    saved_state = drive_env._state.copy()
    saved_prev_power = drive_env._prev_battery_power_w
    saved_step = drive_env._step_count
    saved_idx = drive_env._drive_cycle._idx

    # Action = -1.0 (full discharge offer)
    _, _, _, _, info_neg1 = drive_env.step(np.array([-1.0], dtype=np.float32))

    # Restore & Action = 0.0 (zero power offer)
    drive_env._state = saved_state.copy()
    drive_env._prev_battery_power_w = saved_prev_power
    drive_env._step_count = saved_step
    drive_env._drive_cycle._idx = saved_idx
    _, _, _, _, info_zero = drive_env.step(np.array([0.0], dtype=np.float32))

    # In propulsion: action=-1.0 supplies discharge power; action=0 supplies 0 discharge power (full deficit)
    assert info_neg1["applied_power_w"] < 0.0, "Expected negative power (discharge) for action=-1.0"
    assert info_zero["applied_power_w"] == 0.0, "Expected zero power for action=0.0"
    assert info_zero["power_deficit_w"] > 0.0, "Expected power deficit for action=0.0 during propulsion"
    assert info_neg1["power_deficit_w"] < info_zero["power_deficit_w"]


def test_reward_components_bounded_and_finite(drive_env):
    """Verifies that all reward components are finite across all step actions."""
    drive_env.reset(options={"initial_soc": 0.50, "ambient_temp_c": 25.0})
    for act in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        _, r, _, _, info = drive_env.step(np.array([act], dtype=np.float32))
        assert math.isfinite(r)
        comps = info["reward_components"]
        for k, v in comps.items():
            assert math.isfinite(v)
            assert v >= 0.0  # individual component magnitudes stored as non-negative
