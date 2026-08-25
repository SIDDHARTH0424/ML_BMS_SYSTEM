"""
Tests for the NEW safety_layer_bidirectional() / state_based_discharge_multiplier()
functions (task §13/§26). Confirms:
  1. Charging path is byte-for-byte identical to the existing safety_layer()
     (backward compatibility -- the whole point of adding a new function
     instead of modifying the existing one).
  2. Discharge/regeneration path behaves correctly and safely.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ecm_model import BatteryState
from safety.safety_layer import safety_layer, safety_layer_bidirectional
from utils.config import load_config


def cfg():
    return load_config("safety")


# --- 1. Backward compatibility: charging path unchanged ---------------- #

def test_positive_request_matches_existing_safety_layer_exactly():
    c = cfg()
    for soc in [0.1, 0.5, 0.92, 0.99]:
        for temp in [20.0, 44.0, 50.0]:
            for req in [0.0, 50.0, 160.0, 300.0]:
                state = BatteryState(soc=soc, v_rc=0.0, temperature_c=temp)
                applied_old, info_old = safety_layer(req, state, c, estimated_voltage=380.0)
                applied_new, info_new = safety_layer_bidirectional(req, state, c, estimated_voltage=380.0)
                assert applied_old == applied_new
                assert info_old.safe_current_ceiling == info_new.safe_current_ceiling
                assert info_old.intervention_type == info_new.intervention_type
                assert info_old.magnitude == info_new.magnitude


def test_zero_request_is_a_charging_request_not_discharge():
    # 0.0 >= 0.0 -> must take the charging branch (delegates to safety_layer),
    # not be treated as a degenerate discharge request.
    c = cfg()
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    applied, info = safety_layer_bidirectional(0.0, state, c)
    applied_orig, info_orig = safety_layer(0.0, state, c)
    assert applied == applied_orig == 0.0
    assert info.intervention_type == info_orig.intervention_type


# --- 2. Propulsion (discharge) direction -------------------------------- #

def test_discharge_request_returns_negative_applied_current():
    c = cfg()
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    applied, info = safety_layer_bidirectional(-50.0, state, c, estimated_voltage=360.0)
    assert applied < 0.0
    assert applied >= -c["discharge_i_max_a"] - 1e-9


def test_discharge_capped_at_discharge_i_max():
    c = cfg()
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    huge_request = -(c["discharge_i_max_a"] * 5.0)
    applied, info = safety_layer_bidirectional(huge_request, state, c, estimated_voltage=360.0)
    assert applied == pytest.approx(-c["discharge_i_max_a"])
    assert info.intervention_type == "discharge_current_limit"


# --- 3. Low-SoC discharge taper (mirrors the existing high-SoC charge taper) ---

def test_low_soc_derates_discharge():
    c = cfg()
    state_low = BatteryState(soc=0.05, v_rc=0.0, temperature_c=25.0)  # below soc_discharge_taper_start=0.10
    state_mid = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    applied_low, _ = safety_layer_bidirectional(-160.0, state_low, c, estimated_voltage=360.0)
    applied_mid, _ = safety_layer_bidirectional(-160.0, state_mid, c, estimated_voltage=360.0)
    assert abs(applied_low) < abs(applied_mid)


def test_empty_soc_blocks_discharge_entirely():
    c = cfg()
    state = BatteryState(soc=0.0, v_rc=0.0, temperature_c=25.0)  # == soc_discharge_empty
    applied, info = safety_layer_bidirectional(-100.0, state, c, estimated_voltage=360.0)
    assert applied == pytest.approx(0.0, abs=1e-6)


# --- 4. Undervoltage taper ---------------------------------------------- #

def test_low_voltage_derates_discharge():
    c = cfg()
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    applied_normal, _ = safety_layer_bidirectional(-100.0, state, c, estimated_voltage=360.0)
    applied_low_v, _ = safety_layer_bidirectional(-100.0, state, c, estimated_voltage=305.0)
    assert abs(applied_low_v) < abs(applied_normal)


def test_hard_min_voltage_blocks_discharge():
    c = cfg()
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    applied, info = safety_layer_bidirectional(-100.0, state, c, estimated_voltage=c["v_hard_min"])
    assert applied == pytest.approx(0.0, abs=1e-6)


# --- 5. Temperature derating reused correctly for discharge -------------- #

def test_high_temperature_derates_discharge_too():
    c = cfg()
    state_hot = BatteryState(soc=0.5, v_rc=0.0, temperature_c=50.0)  # between derate_start/cutoff
    state_normal = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    applied_hot, _ = safety_layer_bidirectional(-160.0, state_hot, c, estimated_voltage=360.0)
    applied_normal, _ = safety_layer_bidirectional(-160.0, state_normal, c, estimated_voltage=360.0)
    assert abs(applied_hot) < abs(applied_normal)


# --- 6. Monotonicity: applied magnitude never decreases as |request| decreases --
def test_discharge_monotonic_in_request_magnitude():
    c = cfg()
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=25.0)
    prev = 0.0
    for req in [-10.0, -30.0, -60.0, -100.0, -160.0, -200.0]:
        applied, _ = safety_layer_bidirectional(req, state, c, estimated_voltage=360.0)
        assert abs(applied) >= prev - 1e-9
        prev = abs(applied)


# --- 7. Finite outputs everywhere ---------------------------------------- #
def test_outputs_finite():
    c = cfg()
    for soc in [0.0, 0.05, 0.5, 0.95, 1.0]:
        for temp in [-10.0, 25.0, 50.0, 70.0]:
            for req in [-500.0, -1.0, 0.0, 1.0, 500.0]:
                state = BatteryState(soc=soc, v_rc=0.0, temperature_c=temp)
                applied, info = safety_layer_bidirectional(req, state, c, estimated_voltage=350.0)
                assert math.isfinite(applied)
                assert math.isfinite(info.safe_current_ceiling)
                assert math.isfinite(info.magnitude)
