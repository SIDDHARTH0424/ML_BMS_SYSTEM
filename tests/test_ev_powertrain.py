"""
End-to-end battery-integration tests (task §13, §26): confirms, through
the REAL ECM (not just the safety layer in isolation), that:
  - propulsion (discharge) -> SoC decreases
  - regeneration (charging-direction current from recovered power) -> SoC increases
  - the existing charging path's SoC behavior is completely unchanged

This is the direct test of §13's three required checks, one level up from
tests/test_safety_bidirectional.py (which tests the safety layer alone).
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ecm_model import BatteryECM, BatteryState
from environment.drivetrain_model import DrivetrainModel
from safety.safety_layer import safety_layer, safety_layer_bidirectional
from utils.config import load_config


def make_ecm():
    return BatteryECM(load_config("battery"))


# 1. Propulsion demand -> battery SOC decreases
def test_propulsion_decreases_soc():
    ecm = make_ecm()
    safety_cfg = load_config("safety")
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)

    # A propulsion power demand, converted through the drivetrain to a
    # negative (discharge) battery-current request.
    drivetrain = DrivetrainModel(load_config("drivetrain"))
    out = drivetrain.compute(p_wheel_w=20000.0)  # propulsion demand
    v_est = ecm.terminal_voltage(state, 0.0)
    requested_discharge_current = -(out.battery_power_w / v_est)  # P=IV -> I=P/V, negative = discharge

    applied, info = safety_layer_bidirectional(requested_discharge_current, state, safety_cfg, estimated_voltage=v_est)
    assert applied < 0.0

    new_state = ecm.step(state, applied, ambient_temp_c=25.0)
    assert new_state.soc < state.soc


# 2. Regenerative power -> battery SOC increases
def test_regeneration_increases_soc():
    ecm = make_ecm()
    safety_cfg = load_config("safety")
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)

    drivetrain = DrivetrainModel(load_config("drivetrain"))
    out = drivetrain.compute(p_wheel_w=-15000.0)  # braking -> regen opportunity
    assert out.available_regenerative_power_w > 0.0

    v_est = ecm.terminal_voltage(state, 0.0)
    requested_charge_current = out.available_regenerative_power_w / v_est  # positive = charging

    applied, info = safety_layer_bidirectional(requested_charge_current, state, safety_cfg, estimated_voltage=v_est)
    assert applied > 0.0  # regen current is positive/charging-direction -> takes the safety_layer() branch

    new_state = ecm.step(state, applied, ambient_temp_c=25.0)
    assert new_state.soc > state.soc


# 3. Existing charging behavior is completely unchanged
def test_charging_path_soc_behavior_unchanged():
    ecm = make_ecm()
    safety_cfg = load_config("safety")
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)

    requested = 100.0  # a normal charging request, exactly as battery_env.py would produce
    v_est = ecm.terminal_voltage(state, requested)

    applied_old, info_old = safety_layer(requested, state, safety_cfg, estimated_voltage=v_est)
    applied_new, info_new = safety_layer_bidirectional(requested, state, safety_cfg, estimated_voltage=v_est)
    assert applied_old == applied_new

    new_state_old = ecm.step(state, applied_old, ambient_temp_c=25.0)
    new_state_new = ecm.step(state, applied_new, ambient_temp_c=25.0)
    assert new_state_old.soc == new_state_new.soc
    assert new_state_old.soc > state.soc  # charging still increases SoC exactly as before


# Current sign consistency: discharge current negative, charge current
# positive, matching the project-wide convention documented in
# audit/vehicle_integration_plan.md
def test_current_sign_matches_documented_convention():
    ecm = make_ecm()
    safety_cfg = load_config("safety")
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    v_est = ecm.terminal_voltage(state, 0.0)

    applied_discharge, _ = safety_layer_bidirectional(-80.0, state, safety_cfg, estimated_voltage=v_est)
    applied_charge, _ = safety_layer_bidirectional(80.0, state, safety_cfg, estimated_voltage=v_est)
    assert applied_discharge < 0.0
    assert applied_charge > 0.0


# Voltage/temperature/safety-limit checks flow through correctly for the
# integrated path (not just the safety layer alone)
def test_integrated_path_respects_temperature_and_safety_limits():
    ecm = make_ecm()
    safety_cfg = load_config("safety")
    hot_state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=50.0)
    normal_state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    v_est_hot = ecm.terminal_voltage(hot_state, 0.0)
    v_est_normal = ecm.terminal_voltage(normal_state, 0.0)

    applied_hot, _ = safety_layer_bidirectional(-160.0, hot_state, safety_cfg, estimated_voltage=v_est_hot)
    applied_normal, _ = safety_layer_bidirectional(-160.0, normal_state, safety_cfg, estimated_voltage=v_est_normal)
    assert abs(applied_hot) < abs(applied_normal)

    new_hot = ecm.step(hot_state, applied_hot, ambient_temp_c=35.0)
    new_normal = ecm.step(normal_state, applied_normal, ambient_temp_c=25.0)
    assert math.isfinite(new_hot.soc) and math.isfinite(new_normal.soc)


def test_full_power_to_current_to_soc_pipeline_finite():
    ecm = make_ecm()
    safety_cfg = load_config("safety")
    drivetrain = DrivetrainModel(load_config("drivetrain"))
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)

    for p_wheel in [-50000.0, -5000.0, 0.0, 5000.0, 50000.0, 200000.0]:
        out = drivetrain.compute(p_wheel_w=p_wheel)
        v_est = ecm.terminal_voltage(state, 0.0)
        if out.available_regenerative_power_w > 0.0:
            req = out.available_regenerative_power_w / v_est
        elif out.battery_power_w > 0.0:
            req = -(out.battery_power_w / v_est)
        else:
            req = 0.0
        applied, info = safety_layer_bidirectional(req, state, safety_cfg, estimated_voltage=v_est)
        new_state = ecm.step(state, applied, ambient_temp_c=25.0)
        assert math.isfinite(applied)
        assert math.isfinite(new_state.soc)
        assert math.isfinite(new_state.temperature_c)
