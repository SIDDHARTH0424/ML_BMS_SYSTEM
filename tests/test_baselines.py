"""
Regression tests for baseline controllers running through the full
safety-layer + ECM stack. These exist to catch unintended changes to the
battery model or safety layer (e.g. an accidental parameter edit) by
checking that charge time still falls in the physically-expected range —
distinct from tests/test_ecm.py, which validates the ECM in isolation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.cc import ConstantCurrentController
from environment.ecm_model import BatteryECM
from safety.safety_layer import safety_layer
from utils.config import load_config


@pytest.fixture
def configs():
    return {
        "battery": load_config("battery"),
        "safety": load_config("safety"),
        "evaluation": load_config("evaluation"),
    }


def _charge_from_to(controller, ecm, safety_cfg, from_soc, to_soc, ambient=25.0, max_steps=20000):
    state = ecm.reset_state(initial_soc=from_soc, ambient_temp_c=ambient)
    controller.reset()
    prev_current = 0.0
    steps = 0
    while state.soc < to_soc and steps < max_steps:
        v = ecm.terminal_voltage(state, prev_current)
        obs = {"soc": state.soc, "terminal_voltage": v, "temperature_c": state.temperature_c,
               "previous_current_a": prev_current, "ambient_temp_c": ambient}
        requested = controller.act(obs)
        applied, _ = safety_layer(requested, state, safety_cfg, estimated_voltage=v)
        state = ecm.step(state, applied, ambient)
        prev_current = applied
        steps += 1
    return steps, state.soc


def test_cc_charges_20_to_80_percent_within_expected_time(configs):
    """Regression: CC controller charging 20%->80% should take roughly
    (0.6 * capacity_ah / cc_current_a) hours, +/- safety-layer tapering
    slack near the top of the range. A large deviation signals an
    unintended change to R0/R1/capacity/safety thresholds."""
    battery_cfg, safety_cfg, eval_cfg = configs["battery"], configs["safety"], configs["evaluation"]
    ecm = BatteryECM(battery_cfg)
    controller = ConstantCurrentController(eval_cfg["cc"])

    steps, final_soc = _charge_from_to(controller, ecm, safety_cfg, from_soc=0.20, to_soc=0.80)
    elapsed_s = steps * ecm.dt

    expected_seconds = 0.60 * battery_cfg["nominal_capacity_ah"] * 3600.0 / eval_cfg["cc"]["current_a"]

    assert final_soc >= 0.80
    # Generous tolerance (safety-layer SoC tapering starts at 90%, so it
    # shouldn't affect the 20->80% window much, but current does ramp from 0).
    assert elapsed_s == pytest.approx(expected_seconds, rel=0.15)


def test_cc_charge_time_is_deterministic(configs):
    """Same scenario run twice must give identical results (no hidden randomness)."""
    battery_cfg, safety_cfg, eval_cfg = configs["battery"], configs["safety"], configs["evaluation"]
    ecm = BatteryECM(battery_cfg)

    steps1, soc1 = _charge_from_to(ConstantCurrentController(eval_cfg["cc"]), ecm, safety_cfg, 0.20, 0.80)
    steps2, soc2 = _charge_from_to(ConstantCurrentController(eval_cfg["cc"]), ecm, safety_cfg, 0.20, 0.80)

    assert steps1 == steps2
    assert soc1 == pytest.approx(soc2)
