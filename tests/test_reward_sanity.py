"""
Permanent regression test for reward design sanity.

Earlier in this project, PPO repeatedly converged to "always request max
current" not because of a training bug, but because the reward (and,
separately, the safety layer's semantics) genuinely made that the optimal
strategy. This was only caught by manually simulating hand-designed
policies and comparing total reward — a check that's cheap enough to run
on every commit, so it's captured here as a permanent test rather than an
ad-hoc one-off script.

If this test starts failing, it means a reward or safety-layer change has
reintroduced a degenerate optimum (e.g. "always max" or "do nothing")
before spending any PPO training budget on it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.cc import ConstantCurrentController
from baselines.cccv import CCCVController
from environment.battery_env import BatteryChargingEnv
from safety.safety_layer import state_based_current_multiplier
from utils.config import load_config


@pytest.fixture
def env():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")


def _run_policy(env, action_fn, soc0=0.10, temp0=25.0, max_steps=7200):
    obs, _ = env.reset(options={"initial_soc": soc0, "ambient_temp_c": temp0})
    total = 0.0
    reached = False
    for _ in range(max_steps):
        action = action_fn(env, obs)
        obs, reward, term, trunc, info = env.step(np.array([action], dtype=np.float32))
        total += reward
        if term or trunc:
            reached = info.get("target_reached", False)
            break
    return total, reached


def _always_max(env, obs):
    return 1.0


def _do_nothing(env, obs):
    return -1.0


def _smart_taper(env, obs):
    mult = state_based_current_multiplier(env._state, env.safety_config)
    return 2.0 * mult - 1.0


def _controller_action_fn(controller):
    controller.reset()
    prev_current = [0.0]

    def action_fn(env, obs):
        v = env.ecm.terminal_voltage(env._state, prev_current[0])
        c_obs = {"soc": env._state.soc, "terminal_voltage": v, "temperature_c": env._state.temperature_c,
                 "previous_current_a": prev_current[0], "ambient_temp_c": env._ambient_temp_c}
        requested = controller.act(c_obs)
        prev_current[0] = requested
        return 2.0 * (requested / env.i_max) - 1.0

    return action_fn


def test_do_nothing_is_never_the_best_policy(env):
    """The degenerate 'sit idle forever' optimum must never be reward-optimal
    — this was a real failure mode before the time_penalty / terminal
    shortfall_penalty fix (v3)."""
    do_nothing_total, do_nothing_reached = _run_policy(env, _do_nothing)
    smart_total, smart_reached = _run_policy(env, _smart_taper)

    assert not do_nothing_reached
    assert smart_reached
    assert smart_total > do_nothing_total
    # Do-nothing should be substantially negative, not just "a bit worse"
    assert do_nothing_total < 0


def test_smart_taper_beats_always_max(env):
    """A policy that anticipates the safety ceiling (requests exactly what
    the safety layer allows) must score higher than blindly requesting max
    current always — this was a real failure mode before the v2 safety
    layer semantics fix (double-derating made 'always max' secretly
    optimal regardless of reward shaping)."""
    max_total, max_reached = _run_policy(env, _always_max)
    smart_total, smart_reached = _run_policy(env, _smart_taper)

    assert max_reached and smart_reached
    assert smart_total > max_total


def test_real_baseline_controllers_reach_target_and_beat_always_max(env):
    """Sanity check that real (non-synthetic) baseline controllers — which
    are supposed to represent reasonably good charging strategies — also
    outperform the naive always-max policy under the reward, and all
    successfully reach the target."""
    eval_cfg = load_config("evaluation")
    max_total, max_reached = _run_policy(env, _always_max)

    for name, controller in [
        ("CC", ConstantCurrentController(eval_cfg["cc"])),
        ("CCCV", CCCVController(eval_cfg["cccv"])),
    ]:
        total, reached = _run_policy(env, _controller_action_fn(controller))
        assert reached, f"{name} failed to reach target_soc"
        assert total > max_total, f"{name} (reward={total:.1f}) did not beat always_max (reward={max_total:.1f})"
