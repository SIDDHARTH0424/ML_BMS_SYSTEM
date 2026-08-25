"""
Environment-level invariant tests (audit ISSUES 014/015/016):

  ISSUE-014 (Issue 5 in the task): verify the safety layer is actually in
  the applied-current path -- I_applied that reaches the battery model must
  never exceed the safety ceiling or i_max, and requested_current (charging
  -only convention) must never go negative.

  ISSUE-015 (Issue 6): reward must stay finite (no NaN/Inf) across normal
  operation AND deliberately stressed states (high temperature, near
  voltage limit, near SoC target, terminal states).

  ISSUE-016 (Issue 7): observation must stay finite at every step under the
  same conditions.

These run full episodes with RANDOM actions (not a trained policy -- none
is required or assumed to exist), including actions sampled outside
[-1, 1] to also exercise the clipping path from ISSUE-013.
"""
from __future__ import annotations

import numpy as np
import pytest

from environment.env_factory import make_env


def _run_episode_and_collect(env, rng, max_steps=500, action_low=-1.0, action_high=1.0):
    """Runs one episode with random actions, returning per-step logs used
    to check the invariants below."""
    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    assert np.isfinite(obs).all(), "initial observation must be finite"

    records = []
    for _ in range(max_steps):
        action = rng.uniform(action_low, action_high, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "obs": obs,
                "reward": reward,
                "applied_current_a": info["applied_current_a"],
                "safety_intervention": info["safety_intervention"],
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated:
            break
    return records


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# --------------------------------------------------------------------- #
# ISSUE-014 / Issue 5: safety-layer order & invariants
# --------------------------------------------------------------------- #
def test_applied_current_never_exceeds_i_max_across_episode(rng):
    env = make_env(mode="eval")
    i_max = env.i_max
    for _ in range(5):
        records = _run_episode_and_collect(env, rng, action_low=-1.0, action_high=1.0)
        for r in records:
            assert r["applied_current_a"] <= i_max + 1e-6, (
                f"applied_current_a={r['applied_current_a']} exceeded i_max={i_max}"
            )
            assert r["applied_current_a"] >= 0.0, (
                "requested/applied current went negative -- this environment is "
                "charging-only per configs/safety.yaml (i_min_a=0.0)"
            )


def test_applied_current_never_exceeds_out_of_range_actions(rng):
    # Deliberately feed actions outside the declared [-1, 1] Box to exercise
    # the clipping path (ISSUE-013) together with the safety layer, end to
    # end -- a raw out-of-range action must still never reach the battery
    # model as an out-of-bound current.
    env = make_env(mode="eval")
    i_max = env.i_max
    records = _run_episode_and_collect(env, rng, action_low=-10.0, action_high=10.0)
    assert len(records) > 0
    for r in records:
        assert 0.0 <= r["applied_current_a"] <= i_max + 1e-6


def test_safety_intervention_flagged_when_requesting_max_in_derated_state(rng):
    # Force a high-temperature reset state (near t_hard_cutoff_c) and
    # request full current every step -- the safety layer must actually
    # intervene (applied < requested), proving the request does pass
    # through the safety layer rather than going straight to the battery
    # model.
    env = make_env(mode="eval")
    from utils.config import load_config
    safety_cfg = load_config("safety")
    hot_temp = safety_cfg["t_hard_cutoff_c"] - 2.0  # well into the derate band
    env.reset(seed=0, options={"initial_soc": 0.3, "ambient_temp_c": hot_temp})
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))  # request i_max
    assert info["safety_intervention"]["type"] != "none"
    assert info["applied_current_a"] < env.i_max - 1e-3


# --------------------------------------------------------------------- #
# ISSUE-015 / Issue 6: reward finiteness
# --------------------------------------------------------------------- #
def test_reward_finite_under_normal_conditions(rng):
    env = make_env(mode="eval")
    records = _run_episode_and_collect(env, rng)
    rewards = [r["reward"] for r in records]
    assert np.isfinite(rewards).all(), f"non-finite reward(s) found: {rewards}"


@pytest.mark.parametrize(
    "initial_soc,ambient_temp_c,label",
    [
        (0.3, 25.0, "normal"),
        (0.3, 50.0, "high_temperature"),   # near/above t_derate_start_c=45
        (0.98, 25.0, "near_soc_target"),   # target_soc default is high (see simulation.yaml)
        (0.5, 60.0, "extreme_temperature"),  # above t_hard_cutoff_c=55
    ],
)
def test_reward_finite_in_stressed_states(rng, initial_soc, ambient_temp_c, label):
    env = make_env(mode="eval")
    env.reset(seed=0, options={"initial_soc": initial_soc, "ambient_temp_c": ambient_temp_c})
    for _ in range(50):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward), f"non-finite reward in stressed state '{label}': {reward}"
        assert np.isfinite(obs).all(), f"non-finite observation in stressed state '{label}': {obs}"
        if terminated or truncated:
            # Terminal step: reward (incl. terminal bonus/penalty) must
            # still be finite -- checked above before this reset.
            env.reset(seed=0, options={"initial_soc": initial_soc, "ambient_temp_c": ambient_temp_c})


def test_reward_components_all_finite(rng):
    env = make_env(mode="eval")
    env.reset(seed=0)
    for _ in range(50):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        _, _, terminated, truncated, info = env.step(action)
        components = info["reward_components"]
        for k, v in components.items():
            assert np.isfinite(v), f"reward component '{k}'={v} is not finite"
        if terminated or truncated:
            env.reset(seed=0)


# --------------------------------------------------------------------- #
# ISSUE-016 / Issue 7: observation finiteness
# --------------------------------------------------------------------- #
def test_observation_finite_under_normal_conditions(rng):
    env = make_env(mode="eval")
    records = _run_episode_and_collect(env, rng)
    for r in records:
        assert np.isfinite(r["obs"]).all(), f"non-finite observation: {r['obs']}"


@pytest.mark.parametrize(
    "initial_soc,ambient_temp_c",
    [(0.01, 5.0), (0.99, 55.0), (0.5, -10.0), (0.5, 60.0)],
)
def test_observation_finite_at_extreme_initial_conditions(rng, initial_soc, ambient_temp_c):
    env = make_env(mode="eval")
    obs, _ = env.reset(seed=0, options={"initial_soc": initial_soc, "ambient_temp_c": ambient_temp_c})
    assert np.isfinite(obs).all()
    for _ in range(100):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all(), f"non-finite observation: {obs}"
        assert np.isfinite(reward)
        if terminated or truncated:
            break
