"""Tests for environment/drive_cycle.py (task §23)."""
import csv
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.drive_cycle import DriveCycle, DriveCycleValidationError

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
BASIC_CSV = os.path.join(FIXTURES, "synthetic_test_cycle.csv")
GRADE_CSV = os.path.join(FIXTURES, "synthetic_test_cycle_with_grade.csv")


def write_csv(rows, columns):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.writer(f)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(row)
    f.close()
    return f.name


# reset
def test_reset_returns_to_first_sample():
    dc = DriveCycle(BASIC_CSV)
    dc.step()
    dc.step()
    dc.reset()
    assert dc.current_time() == 0.0
    assert dc.current_speed() == 0.0


# indexing / timestep
def test_step_advances_one_sample_each_call():
    dc = DriveCycle(BASIC_CSV)
    t0 = dc.current_time()
    dc.step()
    t1 = dc.current_time()
    assert t1 - t0 == dc.dt_seconds
    assert dc.dt_seconds == 1.0


# "interpolation if used" -- this implementation deliberately does NOT
# interpolate; confirm step() lands exactly on recorded samples, not
# an interpolated value.
def test_no_interpolation_lands_on_exact_recorded_values():
    dc = DriveCycle(BASIC_CSV)
    dc.step()  # index 1 -> t=1, speed=2.0 exactly as recorded
    assert dc.current_speed() == 2.0


# speed validity: negative speed rejected
def test_negative_speed_rejected():
    path = write_csv([[0, 1.0], [1, -2.0]], ["time_s", "speed_mps"])
    with pytest.raises(DriveCycleValidationError):
        DriveCycle(path)


# acceleration derivation when column absent
def test_acceleration_derived_when_missing():
    dc = DriveCycle(BASIC_CSV)  # no acceleration_mps2 column in this fixture
    dc.reset()
    assert dc.current_acceleration() == 0.0  # first sample, no prior -> 0.0
    dc.step()  # t=1, speed 0.0->2.0, dt=1 -> a = 2.0
    assert dc.current_acceleration() == pytest.approx(2.0)
    dc.step()  # t=2, speed 2.0->5.0, dt=1 -> a = 3.0
    assert dc.current_acceleration() == pytest.approx(3.0)


# acceleration NOT re-derived when explicitly provided
def test_acceleration_not_overridden_when_provided():
    dc = DriveCycle(GRADE_CSV)  # acceleration_mps2 present, all zeros, speed constant
    for _ in range(len(dc)):
        assert dc.current_acceleration() == 0.0
        if not dc.is_done():
            dc.step()


# grade handling: absent -> 0.0 radians; present -> converted from degrees
def test_grade_defaults_to_zero_when_absent():
    dc = DriveCycle(BASIC_CSV)
    assert dc.current_grade() == 0.0


def test_grade_converted_from_degrees_to_radians():
    dc = DriveCycle(GRADE_CSV)
    dc.step()  # t=1, road_grade_deg=3.0
    assert dc.current_grade() == pytest.approx(math.radians(3.0))
    dc.step()  # t=2, road_grade_deg=3.0
    dc.step()  # t=3, road_grade_deg=-2.0
    assert dc.current_grade() == pytest.approx(math.radians(-2.0))


# end-of-cycle behavior
def test_is_done_at_last_sample_and_step_returns_false():
    dc = DriveCycle(BASIC_CSV)
    for _ in range(len(dc) - 1):
        assert not dc.is_done()
        assert dc.step() is True
    assert dc.is_done()
    assert dc.step() is False  # already at the end, step() is a no-op returning False
    assert dc.current_time() == 10.0  # stayed at the last sample, didn't run off the end


# NaN/Inf rejection
def test_nan_speed_rejected():
    path = write_csv([[0, 1.0], [1, float("nan")]], ["time_s", "speed_mps"])
    with pytest.raises(DriveCycleValidationError):
        DriveCycle(path)


def test_inf_time_rejected():
    path = write_csv([[0, 1.0], [float("inf"), 2.0]], ["time_s", "speed_mps"])
    with pytest.raises(DriveCycleValidationError):
        DriveCycle(path)


# time must strictly increase / positive timestep
def test_non_increasing_time_rejected():
    path = write_csv([[0, 1.0], [0, 2.0]], ["time_s", "speed_mps"])
    with pytest.raises(DriveCycleValidationError):
        DriveCycle(path)


def test_decreasing_time_rejected():
    path = write_csv([[1, 1.0], [0, 2.0]], ["time_s", "speed_mps"])
    with pytest.raises(DriveCycleValidationError):
        DriveCycle(path)


# missing required column
def test_missing_speed_column_rejected():
    path = write_csv([[0], [1]], ["time_s"])
    with pytest.raises(DriveCycleValidationError):
        DriveCycle(path)


# file not found
def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DriveCycle("/nonexistent/path/does_not_exist.csv")


# no future values exposed
def test_current_methods_never_expose_future_samples():
    dc = DriveCycle(BASIC_CSV)
    # At index 0, current_speed() must be the t=0 value (0.0), never a
    # later one -- there is no method on this class that accepts an
    # offset or lookahead argument at all (verified structurally: the
    # only accessors are the current_*() zero-arg methods).
    assert dc.current_speed() == 0.0
    import inspect
    sig_names = [n for n in dir(dc) if n.startswith("current_")]
    for name in sig_names:
        sig = inspect.signature(getattr(dc, name))
        assert len(sig.parameters) == 0, f"{name} should take no arguments (no lookahead)"
