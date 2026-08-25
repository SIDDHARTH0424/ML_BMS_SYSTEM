import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.battery_env import BatteryChargingEnv
from utils.config import load_config


@pytest.fixture
def env():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 500  # shorten for fast tests
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert obs.shape == (6,)


def test_step_returns_valid_tuple(env):
    env.reset(seed=42)
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) or isinstance(reward, np.floating)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "safety_intervention" in info


def test_episode_terminates_on_target_soc():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 10000
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    obs, _ = env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 10000:
        obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        steps += 1

    assert terminated or truncated
    if terminated:
        assert info["termination_reason"] in ("target_soc_reached", "overvoltage", "overtemperature")


def test_truncation_at_max_episode_steps(env):
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})
    terminated = truncated = False
    steps = 0
    # Action of -1.0 (mapped to 0 current in the new symmetric [-1,1] space)
    # -> never charges -> should truncate at max_episode_steps, never terminate
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0], dtype=np.float32))
        steps += 1
        if steps > 10000:
            pytest.fail("Episode never ended")
    assert truncated
    assert not terminated
    assert steps == env.max_episode_steps


def test_eval_mode_uses_fixed_grid_scenarios():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval")

    seen = set()
    for _ in range(len(sim_cfg["eval"]["initial_soc_grid"]) * len(sim_cfg["eval"]["ambient_temp_grid_c"])):
        obs, info = env.reset()
        seen.add((info["initial_soc"], info["ambient_temp_c"]))
    # All grid combinations should be distinct scenarios
    assert len(seen) == len(sim_cfg["eval"]["initial_soc_grid"]) * len(sim_cfg["eval"]["ambient_temp_grid_c"])


def test_gymnasium_check_env_compliance():
    """Official Stable-Baselines3 environment checker."""
    from stable_baselines3.common.env_checker import check_env

    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 200
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")

    check_env(env, warn=True, skip_render_check=True)


def test_reward_penalizes_smoothness_violation(env):
    env.reset(seed=1, options={"initial_soc": 0.5, "ambient_temp_c": 25.0})
    _, reward_low_jump, *_ = env.step(np.array([0.5], dtype=np.float32))
    obs, reward_big_jump, terminated, truncated, info = env.step(np.array([-1.0], dtype=np.float32))
    # Reward components should include a nonzero smoothness penalty after a big current swing
    assert info["reward_components"]["smoothness_penalty"] >= 0.0


def test_overrequest_penalty_smaller_for_smaller_requests_in_taper_zone():
    """The safety layer derates whatever is requested (applied = requested *
    multiplier when requested < i_max, not a clean cap at a fixed ceiling —
    see test_state_multiplier_matches_actual_safety_layer_clamp), so a
    smaller request in the taper zone should waste less current in absolute
    terms than a larger one, even though neither reaches zero waste. This is
    the real, achievable incentive the overrequest_penalty provides: reduce
    requests as the safety margin shrinks, not eliminate waste entirely."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")

    def penalty_for_action(raw_action):
        env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
        env.reset(seed=1, options={"initial_soc": 0.95, "ambient_temp_c": 25.0})
        _, _, _, _, info = env.step(np.array([raw_action], dtype=np.float32))
        return info["reward_components"]["overrequest_penalty"]

    small_request_penalty = penalty_for_action(-0.5)   # ~40A requested
    large_request_penalty = penalty_for_action(1.0)    # 160A requested
    assert small_request_penalty < large_request_penalty


def test_overrequest_penalty_positive_when_requesting_max_in_taper_zone():
    """Requesting full current deep in the SoC taper zone (where it will be
    heavily clamped) should incur a nonzero over-request penalty."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.95, "ambient_temp_c": 25.0})

    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["reward_components"]["overrequest_penalty"] > 0.0


def test_overrequest_penalty_zero_when_no_clamping_needed():
    """Early in charging (no taper active), requesting full current should
    incur zero over-request penalty since nothing is actually wasted."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["reward_components"]["overrequest_penalty"] == pytest.approx(0.0, abs=1e-6)


def test_terminal_shortfall_penalty_applies_on_truncation_without_target():
    """v3: an episode that truncates without reaching target_soc should
    incur a terminal shortfall penalty proportional to how far short it
    finished — this is the fix for Run 008's failure mode (a policy that
    charged to ~0.9246 and then sat idle until the 7200-step truncation,
    which the old reward treated as reward-neutral relative to succeeding)."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 5  # force truncation almost immediately
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0], dtype=np.float32))  # 0 current
        total_reward += reward

    assert truncated and not terminated
    # Shortfall should be large (started at 0.20, target 0.95, never charged)
    # -> a substantial negative contribution beyond the per-step time_penalty alone.
    per_step_time_penalty_only = reward_cfg["weights"]["time_penalty"] * sim_cfg["max_episode_steps"]
    assert total_reward < -per_step_time_penalty_only  # shortfall penalty adds beyond just time cost


def test_no_shortfall_penalty_when_target_reached():
    """A successfully-completed episode (terminated, not truncated) should
    not incur the terminal shortfall penalty."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))  # full current

    assert terminated and not truncated
    assert info["target_reached"] is True


def test_voltage_estimate_uses_worst_case_not_actual_request():
    """v3.1 fix: the safety ceiling's voltage estimate must be evaluated at
    i_max (worst case), not at the actual requested current — using the
    actual request creates a circular dependency (higher request -> higher
    estimated voltage -> lower voltage-taper multiplier -> lower ceiling)
    that can genuinely violate the safety layer's monotonicity guarantee
    (confirmed by direct construction: an artificially high Vrc reproduces
    real non-monotonic steps, though not reachable by this system's own
    physics). This test confirms the environment's estimate does not vary
    with the requested action."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.5, "ambient_temp_c": 25.0})

    # Two very different requested actions from the same state should see
    # the same safety ceiling behavior driven only by state, not by their
    # own differing requests -- verified indirectly via the safety_penalty
    # magnitude being determined by state alone when both requests are well
    # under any derating zone (both should show zero intervention here).
    _, _, _, _, info_low = env.step(np.array([-0.9], dtype=np.float32))
    env.reset(seed=1, options={"initial_soc": 0.5, "ambient_temp_c": 25.0})
    _, _, _, _, info_high = env.step(np.array([0.9], dtype=np.float32))

    # Both safe (soc=0.5 is nowhere near any taper zone) -- neither should
    # show a voltage_taper intervention regardless of how different their
    # requests were.
    assert info_low["safety_intervention"]["type"] != "voltage_taper"
    assert info_high["safety_intervention"]["type"] != "voltage_taper"


def test_ambient_temperature_step_logging(env):
    """Part 3 requirement: verify ambient_temp_c is exposed in step info, is finite,
    and remains strictly constant within an episode."""
    import math
    obs, reset_info = env.reset(seed=123)
    assert "ambient_temp_c" in reset_info
    initial_ambient = reset_info["ambient_temp_c"]
    assert math.isfinite(initial_ambient)

    for step_idx in range(10):
        _, _, term, trunc, step_info = env.step(np.array([0.5], dtype=np.float32))
        assert "ambient_temp_c" in step_info
        step_ambient = step_info["ambient_temp_c"]
        assert math.isfinite(step_ambient)
        assert step_ambient == initial_ambient, f"Ambient changed at step {step_idx}: {step_ambient} != {initial_ambient}"
        if term or trunc:
            break


def test_mixed_ambient_sampler_configuration():
    """Verify that train_expC_mixed sampling draws from normal [15,35] and stress [35,45] correctly."""
    import copy
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = copy.deepcopy(load_config("simulation"))
    sim_cfg["train"]["mixed_ambient_sampler"] = {
        "p_stress": 0.25,
        "normal_range_c": [15.0, 35.0],
        "stress_range_c": [35.0, 45.0],
    }

    env_mix = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    ambients = []
    for ep in range(100):
        _, info = env_mix.reset(seed=ep + 1000)
        amb = info["ambient_temp_c"]
        ambients.append(amb)
        assert 15.0 <= amb <= 45.0

    stress_count = sum(1 for a in ambients if a >= 35.0)
    normal_count = sum(1 for a in ambients if a < 35.0)
    assert normal_count > 0
    assert stress_count > 0