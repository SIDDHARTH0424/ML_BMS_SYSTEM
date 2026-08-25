"""Tests for environment/drivetrain_model.py (task §24)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.drivetrain_model import DrivetrainModel
from utils.config import load_config


def make_model():
    return DrivetrainModel(load_config("drivetrain"))


# propulsion power: positive wheel power -> positive battery draw
def test_propulsion_power_basic():
    m = make_model()
    out = m.compute(p_wheel_w=10000.0)
    assert out.traction_power_w == 10000.0
    assert out.motor_power_w == 10000.0
    assert out.battery_power_w > out.motor_power_w  # efficiency < 1
    assert out.available_regenerative_power_w == 0.0


# efficiency: P_battery = P_wheel / eta_total exactly
def test_propulsion_efficiency_formula():
    cfg = dict(load_config("drivetrain"))
    cfg["propulsion_efficiency"] = 0.8
    m = DrivetrainModel(cfg)
    out = m.compute(p_wheel_w=8000.0)
    assert out.battery_power_w == pytest.approx(8000.0 / 0.8)


# losses: propulsion losses are battery_power - motor_power, non-negative
def test_propulsion_losses_nonnegative():
    m = make_model()
    out = m.compute(p_wheel_w=15000.0)
    assert out.drivetrain_losses_w == pytest.approx(out.battery_power_w - out.motor_power_w)
    assert out.drivetrain_losses_w >= 0.0


def test_zero_wheel_power_zero_everything():
    m = make_model()
    out = m.compute(p_wheel_w=0.0)
    assert out.battery_power_w == 0.0
    assert out.motor_power_w == 0.0
    assert out.drivetrain_losses_w == 0.0
    assert out.available_regenerative_power_w == 0.0


# power limits: propulsion demand above motor_max_power_w is capped
def test_propulsion_capped_at_motor_max_power():
    m = make_model()
    huge_demand = m.motor_max_power_w * 3.0
    out = m.compute(p_wheel_w=huge_demand)
    assert out.traction_power_w == m.motor_max_power_w
    assert out.motor_power_w == m.motor_max_power_w


# regeneration: negative wheel power -> positive available regen power
def test_regeneration_basic():
    m = make_model()
    out = m.compute(p_wheel_w=-5000.0)
    assert out.available_regenerative_power_w > 0.0
    assert out.battery_power_w == 0.0


# regenerative efficiency: recovered power < mechanical power available (no energy creation)
def test_regen_recovers_less_than_mechanical_available():
    m = make_model()
    mechanical_available = 5000.0
    out = m.compute(p_wheel_w=-mechanical_available)
    assert out.available_regenerative_power_w < mechanical_available
    assert out.available_regenerative_power_w == pytest.approx(
        min(mechanical_available, m.motor_max_power_w, m.max_regen_power_w) * m.regen_efficiency
    )


# regen power limits: capped by max_regen_power_w even if mechanically available power is higher
def test_regen_capped_by_max_regen_power():
    m = make_model()
    huge_braking = m.max_regen_power_w * 10.0
    out = m.compute(p_wheel_w=-huge_braking)
    expected_cap = min(m.max_regen_power_w, m.motor_max_power_w)
    assert out.motor_power_w == pytest.approx(expected_cap)
    assert out.available_regenerative_power_w == pytest.approx(expected_cap * m.regen_efficiency)


# regen power limits: capped by motor_max_power_w
def test_regen_capped_by_motor_max_power():
    cfg = dict(load_config("drivetrain"))
    cfg["max_regen_power_w"] = cfg["motor_max_power_w"] * 10.0  # regen limit above motor cap
    m = DrivetrainModel(cfg)
    out = m.compute(p_wheel_w=-cfg["motor_max_power_w"] * 5.0)
    assert out.motor_power_w == pytest.approx(m.motor_max_power_w)


# no energy creation: regen output is always strictly less than mechanical
# input whenever regen_efficiency < 1, across a range of braking magnitudes
def test_no_energy_creation_across_range():
    m = make_model()
    for mech in [1.0, 100.0, 5000.0, 50000.0, 500000.0]:
        out = m.compute(p_wheel_w=-mech)
        recovered = out.available_regenerative_power_w
        capped_mechanical_input = min(mech, m.motor_max_power_w, m.max_regen_power_w)
        assert recovered <= capped_mechanical_input + 1e-9
        assert recovered == pytest.approx(capped_mechanical_input * m.regen_efficiency)


# config validation: efficiencies must be in (0, 1]
def test_invalid_efficiency_rejected():
    cfg = dict(load_config("drivetrain"))
    bad = dict(cfg)
    bad["propulsion_efficiency"] = 1.5
    with pytest.raises(ValueError):
        DrivetrainModel(bad)
    bad2 = dict(cfg)
    bad2["regen_efficiency"] = 0.0
    with pytest.raises(ValueError):
        DrivetrainModel(bad2)


# finite outputs across a wide range of inputs, including extremes
def test_outputs_finite():
    m = make_model()
    for p in [-1e7, -50000.0, -1.0, 0.0, 1.0, 50000.0, 1e7]:
        out = m.compute(p_wheel_w=p)
        for val in (out.traction_power_w, out.motor_power_w, out.battery_power_w,
                    out.drivetrain_losses_w, out.available_regenerative_power_w):
            assert math.isfinite(val)
