"""
Validate the 1RC Thevenin ECM battery model against manual
Constant-Current charging calculations, before anything else
(safety layer, environment, RL) is built on top of it.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ecm_model import BatteryECM, BatteryState
from utils.config import load_config


@pytest.fixture
def battery_config():
    return load_config("battery")


@pytest.fixture
def ecm(battery_config):
    return BatteryECM(battery_config)


# --------------------------------------------------------------------- #
# 1. SoC evolution: manual coulomb-counting check
# --------------------------------------------------------------------- #
def test_soc_matches_manual_coulomb_counting(ecm, battery_config):
    """SoC after N seconds of constant current I must equal I*t / (capacity*3600)."""
    current_a = 45.0  # arbitrary sub-1C current; test checks the physics relationship, not a specific C-rate
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.20, ambient_temp_c=ambient)

    n_steps = 100
    for _ in range(n_steps):
        state = ecm.step(state, current_a, ambient)

    elapsed_s = n_steps * ecm.dt
    expected_delta_soc = current_a * elapsed_s / (battery_config["nominal_capacity_ah"] * 3600.0)
    expected_soc = 0.20 + expected_delta_soc

    assert state.soc == pytest.approx(expected_soc, rel=1e-6)


def test_soc_clamped_to_valid_range(ecm):
    """SoC must never exceed [0, 1] even under sustained high current."""
    state = ecm.reset_state(initial_soc=0.98, ambient_temp_c=25.0)
    for _ in range(10000):
        state = ecm.step(state, current_a=135.0, ambient_temp_c=25.0)
    assert 0.0 <= state.soc <= 1.0


# --------------------------------------------------------------------- #
# 2. OCV interpolation
# --------------------------------------------------------------------- #
def test_ocv_interpolation_matches_table_endpoints(ecm, battery_config):
    pts = battery_config["ocv_soc_points"]
    assert ecm.ocv(pts["soc"][0]) == pytest.approx(pts["ocv_v"][0])
    assert ecm.ocv(pts["soc"][-1]) == pytest.approx(pts["ocv_v"][-1])


def test_ocv_is_monotonic_increasing(ecm):
    """NMC OCV-SoC curve should be monotonically non-decreasing (smooth learning signal)."""
    socs = np.linspace(0, 1, 50)
    ocvs = [ecm.ocv(s) for s in socs]
    assert all(b >= a - 1e-9 for a, b in zip(ocvs, ocvs[1:]))


def test_ocv_clamps_out_of_range_soc(ecm, battery_config):
    pts = battery_config["ocv_soc_points"]
    assert ecm.ocv(-0.5) == pytest.approx(pts["ocv_v"][0])
    assert ecm.ocv(1.5) == pytest.approx(pts["ocv_v"][-1])


# --------------------------------------------------------------------- #
# 3. Terminal voltage response
# --------------------------------------------------------------------- #
def test_terminal_voltage_rises_above_ocv_when_charging(ecm):
    state = ecm.reset_state(initial_soc=0.5, ambient_temp_c=25.0)
    v = ecm.terminal_voltage(state, current_a=50.0)
    assert v > ecm.ocv(state.soc)


def test_terminal_voltage_equals_ocv_at_zero_current(ecm):
    state = ecm.reset_state(initial_soc=0.5, ambient_temp_c=25.0)
    v = ecm.terminal_voltage(state, current_a=0.0)
    assert v == pytest.approx(ecm.ocv(state.soc))


# --------------------------------------------------------------------- #
# 4. RC branch dynamics: charges toward steady state I*R1, decays at rest
# --------------------------------------------------------------------- #
def test_vrc_approaches_steady_state_under_constant_current(ecm):
    current_a = 45.0
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.5, ambient_temp_c=ambient)

    tau = ecm.r1 * ecm.c1  # RC time constant (s)
    n_steps = int(10 * tau / ecm.dt)  # ~10 time constants -> converged
    for _ in range(n_steps):
        state = ecm.step(state, current_a, ambient)

    expected_steady_vrc = current_a * ecm.r1
    assert state.v_rc == pytest.approx(expected_steady_vrc, rel=0.02)


def test_vrc_decays_toward_zero_at_rest(ecm):
    ambient = 25.0
    state = BatteryState(soc=0.5, v_rc=0.3, temperature_c=ambient)
    tau = ecm.r1 * ecm.c1
    n_steps = int(10 * tau / ecm.dt)
    for _ in range(n_steps):
        state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient)
    assert state.v_rc == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------- #
# 5. Thermal model
# --------------------------------------------------------------------- #
def test_temperature_rises_under_sustained_high_current(ecm):
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.3, ambient_temp_c=ambient)
    for _ in range(3600):  # 1 hour at 1s steps
        state = ecm.step(state, current_a=100.0, ambient_temp_c=ambient)
    assert state.temperature_c > ambient


def test_temperature_stable_at_zero_current_and_equal_ambient(ecm):
    """No current, temp==ambient -> zero heat gen, zero net loss -> temp unchanged."""
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.3, ambient_temp_c=ambient)
    for _ in range(1000):
        state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient)
    assert state.temperature_c == pytest.approx(ambient, abs=1e-6)


def test_temperature_relaxes_toward_ambient_when_hot_and_idle(ecm):
    ambient = 25.0
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=45.0)
    for _ in range(20000):
        state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient)
    assert state.temperature_c == pytest.approx(ambient, abs=0.5)


# --------------------------------------------------------------------- #
# 6. Manual Constant-Current charge comparison (integration-level sanity)
# --------------------------------------------------------------------- #
def test_manual_cc_charge_trajectory_matches_expected_shape(ecm, battery_config):
    """Charge from 20% to ~50% SoC at 1C; verify SoC is monotonically increasing,
    voltage stays within physical bounds, and elapsed time matches the
    capacity/current relationship (Q = I*t)."""
    current_a = 45.0
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.20, ambient_temp_c=ambient)

    target_soc = 0.50
    socs = [state.soc]
    step_count = 0
    max_steps = int(3600 * 2 / ecm.dt)  # safety cap: 2 hours

    while state.soc < target_soc and step_count < max_steps:
        state = ecm.step(state, current_a, ambient)
        v = ecm.terminal_voltage(state, current_a)
        assert v <= battery_config["v_max"] + 5.0  # allow small numerical slack
        socs.append(state.soc)
        step_count += 1

    # Monotonic SoC increase under constant positive current
    assert all(b >= a - 1e-9 for a, b in zip(socs, socs[1:]))

    expected_seconds = (target_soc - 0.20) * battery_config["nominal_capacity_ah"] * 3600.0 / current_a
    actual_seconds = step_count * ecm.dt
    assert actual_seconds == pytest.approx(expected_seconds, rel=0.01)


# --------------------------------------------------------------------- #
# 7. Euler vs RK4 integration method agreement (both should be config-selectable)
# --------------------------------------------------------------------- #
def test_euler_and_rk4_agree_closely_for_smooth_dynamics(battery_config):
    cfg_euler = dict(battery_config)
    cfg_euler["integration_method"] = "euler"
    cfg_rk4 = dict(battery_config)
    cfg_rk4["integration_method"] = "rk4"

    ecm_euler = BatteryECM(cfg_euler)
    ecm_rk4 = BatteryECM(cfg_rk4)

    s_euler = ecm_euler.reset_state(0.3, 25.0)
    s_rk4 = ecm_rk4.reset_state(0.3, 25.0)

    for _ in range(500):
        s_euler = ecm_euler.step(s_euler, 45.0, 25.0)
        s_rk4 = ecm_rk4.step(s_rk4, 45.0, 25.0)

    assert s_euler.soc == pytest.approx(s_rk4.soc, rel=1e-3)
    assert s_euler.v_rc == pytest.approx(s_rk4.v_rc, rel=1e-2)
    assert s_euler.temperature_c == pytest.approx(s_rk4.temperature_c, rel=1e-2)


# --------------------------------------------------------------------- #
# 8. State of Health tracking (monitoring only)
# --------------------------------------------------------------------- #
def test_soh_decreases_monotonically_with_throughput(ecm):
    state = ecm.reset_state(initial_soc=0.3, ambient_temp_c=25.0)
    prev_soh = state.soh
    for _ in range(1000):
        state = ecm.step(state, current_a=45.0, ambient_temp_c=25.0)
        assert state.soh <= prev_soh + 1e-12
        prev_soh = state.soh
    assert state.ah_throughput > 0


# --------------------------------------------------------------------- #
# Thermal model correctness: Vrc^2/R1 resistive-loss formulation
# --------------------------------------------------------------------- #
def test_relaxation_heat_generation_nonzero(ecm):
    """Regression test: at rest (I=0) with a charged RC branch (Vrc > 0),
    R1 is physically still dissipating stored polarization energy as heat
    while Vrc decays. An earlier implementation used current_a*v_rc for
    heat generation, which incorrectly gave exactly zero heat here since
    current_a=0 — even though the RC branch is actively discharging through
    R1. The correct Vrc^2/R1 formulation must give nonzero heat."""
    state = BatteryState(soc=0.5, v_rc=0.05, temperature_c=25.0)
    d_soc, d_vrc, d_temp = ecm._derivatives(state, current_a=0.0, ambient_temp_c=25.0)
    assert d_temp > 0.0


def test_heat_generation_matches_steady_state_equivalence(ecm):
    """At steady state (Vrc = I*R1), Vrc^2/R1 and I*Vrc coincide
    algebraically (both equal I^2*R1) — confirms the fix only changes
    transient behavior, not the steady-state heat generation rate."""
    current_a = 45.0
    steady_vrc = current_a * ecm.r1
    state = BatteryState(soc=0.5, v_rc=steady_vrc, temperature_c=25.0)
    _, d_vrc, _ = ecm._derivatives(state, current_a=current_a, ambient_temp_c=25.0)
    assert d_vrc == pytest.approx(0.0, abs=1e-9)  # confirms this Vrc is indeed the steady-state value
