"""Tests for environment/vehicle_dynamics.py (task §22, 10 required cases)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.vehicle_dynamics import VehicleDynamics
from utils.config import load_config


def make_model():
    return VehicleDynamics(load_config("vehicle"))


# 1. Zero speed
def test_zero_speed_zero_power():
    m = make_model()
    f = m.compute(speed_mps=0.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    assert f.p_wheel == 0.0
    assert f.f_roll == 0.0  # no rolling resistance at a standstill
    assert f.f_aero == 0.0  # v^2 term is zero


# 2. Constant speed (flat road): drag + rolling resistance dominate, no
# accel/grade force, power > 0 (still costs energy to maintain speed).
def test_constant_speed_flat_road():
    m = make_model()
    f = m.compute(speed_mps=20.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    assert f.f_accel == 0.0
    assert f.f_grade == 0.0
    assert f.f_roll > 0.0
    assert f.f_aero > 0.0
    assert f.f_tractive == f.f_roll + f.f_aero
    assert f.p_wheel > 0.0


# 3. Positive acceleration: required force/power increases vs. constant speed
def test_positive_acceleration_increases_demand():
    m = make_model()
    f_const = m.compute(speed_mps=20.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    f_accel = m.compute(speed_mps=20.0, acceleration_mps2=1.5, road_grade_rad=0.0)
    assert f_accel.f_tractive > f_const.f_tractive
    assert f_accel.p_wheel > f_const.p_wheel


# 4. Negative acceleration: braking/regeneration opportunity (tractive
# force/power can go negative)
def test_negative_acceleration_gives_braking_opportunity():
    m = make_model()
    f = m.compute(speed_mps=20.0, acceleration_mps2=-3.0, road_grade_rad=0.0)
    assert f.f_accel < 0.0
    assert f.f_tractive < 0.0
    assert f.p_wheel < 0.0


# 5. Uphill: power demand increases vs. flat road
def test_uphill_increases_power_demand():
    m = make_model()
    f_flat = m.compute(speed_mps=15.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    f_up = m.compute(speed_mps=15.0, acceleration_mps2=0.0, road_grade_rad=math.radians(5.0))
    assert f_up.f_grade > 0.0
    assert f_up.f_tractive > f_flat.f_tractive
    assert f_up.p_wheel > f_flat.p_wheel


# 6. Downhill: grade force changes sign appropriately (can offset/negate demand)
def test_downhill_grade_force_sign():
    m = make_model()
    f_down = m.compute(speed_mps=15.0, acceleration_mps2=0.0, road_grade_rad=math.radians(-5.0))
    f_flat = m.compute(speed_mps=15.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    assert f_down.f_grade < 0.0
    assert f_down.f_tractive < f_flat.f_tractive


# 7. Higher speed: aerodynamic drag increases (nonlinearly, v^2)
def test_aero_drag_increases_with_speed():
    m = make_model()
    f_low = m.compute(speed_mps=10.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    f_high = m.compute(speed_mps=30.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    assert f_high.f_aero > f_low.f_aero
    # v^2 scaling: tripling speed should roughly 9x the aero force
    assert f_high.f_aero / f_low.f_aero > 8.0


# 8. Increasing mass increases force demand (accel + roll + grade all scale with mass)
def test_increasing_mass_increases_force():
    cfg = dict(load_config("vehicle"))
    cfg_heavy = dict(cfg)
    cfg_heavy["mass_kg"] = cfg["mass_kg"] * 1.5
    m_light = VehicleDynamics(cfg)
    m_heavy = VehicleDynamics(cfg_heavy)
    f_light = m_light.compute(speed_mps=15.0, acceleration_mps2=1.0, road_grade_rad=math.radians(3.0))
    f_heavy = m_heavy.compute(speed_mps=15.0, acceleration_mps2=1.0, road_grade_rad=math.radians(3.0))
    assert f_heavy.f_tractive > f_light.f_tractive


# 9. Increasing drag coefficient increases aero force (and only aero force)
def test_increasing_drag_coefficient_increases_aero_only():
    cfg = dict(load_config("vehicle"))
    cfg_dirty = dict(cfg)
    cfg_dirty["drag_coefficient"] = cfg["drag_coefficient"] * 2.0
    m_clean = VehicleDynamics(cfg)
    m_dirty = VehicleDynamics(cfg_dirty)
    f_clean = m_clean.compute(speed_mps=25.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    f_dirty = m_dirty.compute(speed_mps=25.0, acceleration_mps2=0.0, road_grade_rad=0.0)
    assert f_dirty.f_aero > f_clean.f_aero
    assert f_dirty.f_roll == f_clean.f_roll
    assert f_dirty.f_accel == f_clean.f_accel


# 10. All outputs finite across a range of realistic and edge-case inputs
def test_outputs_finite():
    m = make_model()
    for v in [0.0, 0.01, 5.0, 33.3, 55.0]:
        for a in [-4.0, -1.0, 0.0, 1.0, 3.0]:
            for grade_deg in [-15.0, -5.0, 0.0, 5.0, 15.0]:
                f = m.compute(speed_mps=v, acceleration_mps2=a, road_grade_rad=math.radians(grade_deg))
                for val in (f.f_accel, f.f_roll, f.f_aero, f.f_grade, f.f_tractive, f.p_wheel):
                    assert math.isfinite(val)
