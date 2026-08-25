"""
Tests for the Stable V3 state-aware thermal reward (reward.yaml
thermal_enabled / thermal_weight / thermal_reference_temp_c /
thermal_scale_c / thermal_q_reference_w).

Covers TEST 1-11 from the Stable V3 implementation task (TEST 12, PPO
config, is covered by experiments/ not here since it's a training-loop
property, not a reward/env property).
"""
import copy
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.battery_env import BatteryChargingEnv
from environment.ecm_model import BatteryState
from utils.config import load_config


def make_env(thermal_overrides=None):
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = copy.deepcopy(load_config("reward"))
    sim_cfg = load_config("simulation")
    if thermal_overrides:
        reward_cfg.update(thermal_overrides)
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")


def compute_reward_at(env, temperature_c, current_a, v_rc=0.0, soc=0.5):
    """Helper: build a prev/new state pair with a given temperature/current
    and return (total_reward, components) from _compute_reward directly,
    bypassing env.step() so we control temperature exactly."""
    prev_state = BatteryState(soc=soc, v_rc=v_rc, temperature_c=temperature_c)
    new_state = BatteryState(soc=soc + 1e-5, v_rc=v_rc, temperature_c=temperature_c)
    from safety.safety_layer import safety_layer
    applied_current, safety_info = safety_layer(
        current_a, prev_state, env.safety_config,
        estimated_voltage=env.ecm.terminal_voltage(prev_state, current_a),
    )
    terminal_voltage = env.ecm.terminal_voltage(new_state, applied_current)
    total, components = env._compute_reward(prev_state, current_a, new_state, applied_current, safety_info, terminal_voltage)
    return total, components


# TEST 1: thermal_enabled=False -> thermal_reward == 0
def test_thermal_disabled_gives_zero_reward():
    env = make_env({"thermal_enabled": False})
    _, comps = compute_reward_at(env, temperature_c=50.0, current_a=160.0)
    assert comps["thermal_reward"] == 0.0


# TEST 2: temperature < reference -> thermal penalty is zero/negligible
def test_temperature_below_reference_is_negligible():
    env = make_env({"thermal_enabled": True})
    t_ref = env.reward_config["thermal_reference_temp_c"]
    _, comps = compute_reward_at(env, temperature_c=t_ref - 5.0, current_a=160.0)
    assert comps["thermal_reward"] == 0.0


# TEST 3: temperature > reference -> thermal penalty < 0 (i.e. present/nonzero)
def test_temperature_above_reference_is_nonzero():
    env = make_env({"thermal_enabled": True})
    t_ref = env.reward_config["thermal_reference_temp_c"]
    _, comps = compute_reward_at(env, temperature_c=t_ref + 5.0, current_a=160.0)
    assert comps["thermal_reward"] > 0.0  # stored as a positive penalty magnitude, subtracted in total


# TEST 4: higher temperature -> thermal penalty magnitude increases
def test_thermal_penalty_increases_with_temperature():
    env = make_env({"thermal_enabled": True})
    t_ref = env.reward_config["thermal_reference_temp_c"]
    _, comps_low = compute_reward_at(env, temperature_c=t_ref + 2.0, current_a=160.0)
    _, comps_high = compute_reward_at(env, temperature_c=t_ref + 10.0, current_a=160.0)
    assert comps_high["thermal_reward"] > comps_low["thermal_reward"]


# TEST 5: same temperature but higher q_gen (via current) -> thermal penalty increases
def test_thermal_penalty_increases_with_qgen():
    env = make_env({"thermal_enabled": True})
    t_ref = env.reward_config["thermal_reference_temp_c"]
    _, comps_low_i = compute_reward_at(env, temperature_c=t_ref + 5.0, current_a=20.0)
    _, comps_high_i = compute_reward_at(env, temperature_c=t_ref + 5.0, current_a=160.0)
    assert comps_high_i["thermal_reward"] > comps_low_i["thermal_reward"]


# TEST 6 & 7: thermal penalty and total reward are finite across a temperature sweep
def test_thermal_reward_and_total_are_finite():
    env = make_env({"thermal_enabled": True})
    for t in [-10.0, 0.0, 39.9, 40.0, 45.0, 55.0, 60.0, 100.0]:
        total, comps = compute_reward_at(env, temperature_c=t, current_a=160.0)
        assert math.isfinite(comps["thermal_reward"])
        assert math.isfinite(total)


# TEST 8: q_gen (via ecm.heat_generation_w) is finite
def test_qgen_is_finite():
    battery_cfg = load_config("battery")
    from environment.ecm_model import BatteryECM
    ecm = BatteryECM(battery_cfg)
    for i in [0.0, 80.0, 160.0]:
        for v_rc in [0.0, 1.0, 2.0]:
            q = ecm.heat_generation_w(BatteryState(soc=0.5, v_rc=v_rc, temperature_c=25.0), i)
            assert math.isfinite(q)
            assert q >= 0.0


# TEST 9 & 10: safety layer output unchanged; applied_current <= safe_ceiling (and <= i_max)
def test_safety_layer_unaffected_by_thermal_flag():
    env_off = make_env({"thermal_enabled": False})
    env_on = make_env({"thermal_enabled": True})
    from safety.safety_layer import safety_layer
    state = BatteryState(soc=0.92, v_rc=0.0, temperature_c=50.0)
    applied_off, info_off = safety_layer(160.0, state, env_off.safety_config,
                                          estimated_voltage=env_off.ecm.terminal_voltage(state, 160.0))
    applied_on, info_on = safety_layer(160.0, state, env_on.safety_config,
                                        estimated_voltage=env_on.ecm.terminal_voltage(state, 160.0))
    assert applied_off == applied_on
    assert info_off.safe_current_ceiling == info_on.safe_current_ceiling
    assert applied_on <= info_on.safe_current_ceiling + 1e-9
    assert applied_on <= env_on.i_max + 1e-9


# TEST 11: SoH does not enter the reward, with the thermal reward on or off
def test_soh_not_in_reward_components():
    for enabled in (False, True):
        env = make_env({"thermal_enabled": enabled})
        _, comps = compute_reward_at(env, temperature_c=45.0, current_a=160.0)
        assert not any("soh" in k.lower() for k in comps.keys())


# Baseline reproducibility: thermal_enabled=False must reproduce the exact
# original (pre-Stable-V3) reward -- no thermal_reward contribution anywhere.
def test_baseline_mode_matches_original_reward_shape():
    env = make_env({"thermal_enabled": False})
    for t in [20.0, 40.0, 50.0, 60.0]:
        _, comps = compute_reward_at(env, temperature_c=t, current_a=160.0)
        assert comps["thermal_reward"] == 0.0
        # original components still present and computed as before
        for key in ["progress", "temp_penalty", "safety_penalty", "overrequest_penalty",
                    "smoothness_penalty", "time_penalty"]:
            assert key in comps


def test_thermal_derating_at_crossover_temperature():
    """At T=40C (empirically determined crossover), derating to 120A gives higher total reward than 160A."""
    env = make_env({"thermal_enabled": True})
    
    # Evaluate at T=40C
    # Using step with ECM dynamics
    prev_s = BatteryState(soc=0.30, v_rc=0.0, temperature_c=40.0)
    
    # 160A step
    new_s_160 = env.ecm.step(prev_s, 160.0, 40.0)
    vt_160 = env.ecm.terminal_voltage(new_s_160, 160.0)
    from safety.safety_layer import safety_layer
    app_160, sinfo_160 = safety_layer(160.0, prev_s, env.safety_config, estimated_voltage=vt_160)
    env._is_first_step = False
    env._prev_current_a = 160.0
    r_160, _ = env._compute_reward(prev_s, 160.0, new_s_160, app_160, sinfo_160, vt_160)
    
    # 120A step
    new_s_120 = env.ecm.step(prev_s, 120.0, 40.0)
    vt_120 = env.ecm.terminal_voltage(new_s_120, 120.0)
    app_120, sinfo_120 = safety_layer(120.0, prev_s, env.safety_config, estimated_voltage=vt_120)
    env._prev_current_a = 120.0
    r_120, _ = env._compute_reward(prev_s, 120.0, new_s_120, app_120, sinfo_120, vt_120)
    
    assert r_120 > r_160, f"Expected r_120 ({r_120}) > r_160 ({r_160}) at T=40C"


def test_thermal_derating_at_elevated_temperature():
    """At T=45C (high stress), lower current produces higher reward than 160A."""
    env = make_env({"thermal_enabled": True})
    prev_s = BatteryState(soc=0.30, v_rc=0.0, temperature_c=45.0)
    
    new_s_160 = env.ecm.step(prev_s, 160.0, 45.0)
    vt_160 = env.ecm.terminal_voltage(new_s_160, 160.0)
    from safety.safety_layer import safety_layer
    app_160, sinfo_160 = safety_layer(160.0, prev_s, env.safety_config, estimated_voltage=vt_160)
    env._is_first_step = False
    env._prev_current_a = 160.0
    r_160, _ = env._compute_reward(prev_s, 160.0, new_s_160, app_160, sinfo_160, vt_160)
    
    new_s_80 = env.ecm.step(prev_s, 80.0, 45.0)
    vt_80 = env.ecm.terminal_voltage(new_s_80, 80.0)
    app_80, sinfo_80 = safety_layer(80.0, prev_s, env.safety_config, estimated_voltage=vt_80)
    env._prev_current_a = 80.0
    r_80, _ = env._compute_reward(prev_s, 80.0, new_s_80, app_80, sinfo_80, vt_80)
    
    assert r_80 > r_160, f"Expected r_80 ({r_80}) > r_160 ({r_160}) at T=45C"

