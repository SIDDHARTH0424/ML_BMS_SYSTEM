"""
Regression tests for action-space mapping and clipping (audit ISSUE-012,
ISSUE-013). Verifies action = -1/0/+1 map to 0 / 0.5*i_max / i_max amps
(and intermediate values in between), reading the environment's ACTUAL
configured i_max rather than hard-coding 160 A, so this test stays correct
even if configs/battery.yaml's i_max_a is ever legitimately changed.

Uses a benign reset state (mid SoC, moderate ambient temperature) chosen
from configs/safety.yaml's own thresholds (soc_taper_start=0.90,
t_derate_start_c=45.0) so no safety-layer derating is active on the first
step -- applied_current_a should exactly equal the raw action-mapping
formula's output, isolating the mapping itself from safety clamping
(which is tested separately in tests/test_safety.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from environment.env_factory import make_env
from utils.config import load_config


@pytest.fixture
def env():
    e = make_env(mode="eval")
    e.reset(seed=0, options={"initial_soc": 0.4, "ambient_temp_c": 25.0})
    return e


@pytest.fixture
def i_max():
    # Derived from the actual config, not hard-coded -- test remains
    # correct if i_max_a is legitimately changed later.
    return float(load_config("battery")["i_max_a"])


def _applied_current_for_action(env, action_val: float, i_max: float) -> float:
    """Reset to the same benign, non-derating state and take one step."""
    env.reset(seed=0, options={"initial_soc": 0.4, "ambient_temp_c": 25.0})
    _, _, _, _, info = env.step(np.array([action_val], dtype=np.float32))
    return info["applied_current_a"]


@pytest.mark.parametrize(
    "action_val,expected_fraction",
    [
        (-1.0, 0.0),
        (0.0, 0.5),
        (1.0, 1.0),
        (-0.5, 0.25),
        (0.5, 0.75),
    ],
)
def test_action_mapping_matches_formula(env, i_max, action_val, expected_fraction):
    applied = _applied_current_for_action(env, action_val, i_max)
    expected = expected_fraction * i_max
    assert applied == pytest.approx(expected, abs=1e-3)


def test_action_minus_one_is_zero_current(env, i_max):
    assert _applied_current_for_action(env, -1.0, i_max) == pytest.approx(0.0, abs=1e-6)


def test_action_zero_is_half_i_max(env, i_max):
    assert _applied_current_for_action(env, 0.0, i_max) == pytest.approx(0.5 * i_max, abs=1e-3)


def test_action_plus_one_is_i_max(env, i_max):
    assert _applied_current_for_action(env, 1.0, i_max) == pytest.approx(i_max, abs=1e-3)


# --------------------------------------------------------------------- #
# ISSUE-013: action clipping for out-of-declared-range inputs.
# environment/battery_env.py::step already clips via
# np.clip(np.asarray(action).flatten()[0], -1.0, 1.0) before mapping --
# these tests verify that behavior directly rather than assuming it.
# --------------------------------------------------------------------- #
def test_action_above_one_is_clipped_to_i_max(env, i_max):
    # 5.0 is well outside the declared [-1, 1] Box -- must behave exactly
    # like action=1.0 (full current), not overflow past i_max.
    applied = _applied_current_for_action(env, 5.0, i_max)
    assert applied == pytest.approx(i_max, abs=1e-3)


def test_action_below_minus_one_is_clipped_to_zero(env, i_max):
    applied = _applied_current_for_action(env, -5.0, i_max)
    assert applied == pytest.approx(0.0, abs=1e-6)


def test_action_clip_is_continuous_at_boundary():
    # No discontinuity: an action just past +1 should map to (approximately)
    # the same current as exactly +1, not jump/wrap.
    e = make_env(mode="eval")
    i_max_val = float(load_config("battery")["i_max_a"])
    at_boundary = _applied_current_for_action(e, 1.0, i_max_val)
    past_boundary = _applied_current_for_action(e, 1.2, i_max_val)
    assert at_boundary == pytest.approx(past_boundary, abs=1e-3)
