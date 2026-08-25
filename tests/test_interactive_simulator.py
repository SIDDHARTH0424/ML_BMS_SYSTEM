"""
Unit/integration tests for app/interactive_ev_simulator.py.

These tests verify:
- helper parsing functions
- trace behavior
- validated model and drive-cycle resolution
- charging/driving environment initialization
- reset and step progression
- mode/controller switching
- temperature, speed, SOC, power, and regen extraction
- wrapped-environment drive-cycle lookup

The tests are designed for the project's real simulator and models; they do
not create fake environments or fake PPO models.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Headless pygame for CI / terminal execution
# -----------------------------------------------------------------------------

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from app.interactive_ev_simulator import (  # noqa: E402
    InteractiveSimulator,
    ROOT,
    Trace,
    UIState,
    first_float,
    safe_float,
)


@pytest.fixture
def simulator():
    """Create the real InteractiveSimulator in headless mode and clean it up."""
    sim = InteractiveSimulator()
    yield sim

    # Prevent pygame resources/windows from leaking between tests.
    try:
        import pygame

        pygame.quit()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Helper tests
# -----------------------------------------------------------------------------


def test_safe_float_and_first_float():
    """Malformed values should safely fall back while finite values are kept."""
    assert safe_float(12.34) == pytest.approx(12.34)
    assert safe_float("45.6") == pytest.approx(45.6)
    assert safe_float(float("nan"), default=5.0) == pytest.approx(5.0)
    assert safe_float(float("inf"), default=7.0) == pytest.approx(7.0)
    assert safe_float(None, default=0.0) == pytest.approx(0.0)

    data = {"a": "invalid", "b": 42.0, "c": 99.0}
    assert first_float(data, ["x", "a", "b", "c"]) == pytest.approx(42.0)
    assert first_float(data, ["nonexistent"], default=-1.0) == pytest.approx(-1.0)


def test_trace_operations():
    """Trace keeps only max_points and clear removes all values."""
    trace = Trace(max_points=5)

    for i in range(10):
        trace.add(float(i))

    assert len(trace.x) == 5
    assert len(trace.y) == 5
    assert trace.y[-1] == pytest.approx(9.0)
    assert trace.x[-1] == pytest.approx(9.0)

    trace.clear()
    assert trace.x == []
    assert trace.y == []


def test_ui_state_defaults():
    """UIState defaults should match the simulator's initial operating state."""
    state = UIState()

    assert state.mode == "charging"
    assert state.playing is False
    assert state.controller == "ppo"
    assert state.speed_multiplier == pytest.approx(2.0)
    assert state.ambient_c == pytest.approx(25.0)
    assert state.initial_soc == pytest.approx(0.50)
    assert state.cycle_index == 3


# -----------------------------------------------------------------------------
# Resource/model/cycle resolution
# -----------------------------------------------------------------------------


def test_project_root_exists():
    """The simulator ROOT must point to a real project directory."""
    assert isinstance(ROOT, Path)
    assert ROOT.exists()
    assert ROOT.is_dir()


def test_visualizer_model_resolution(simulator):
    """The frozen PPO model paths should resolve to existing files."""
    charging_path = simulator.charging_model_path
    driving_path = simulator.driving_model_path

    assert charging_path is not None, (
        "Charging PPO model was not found under final_models/"
    )
    assert charging_path.exists(), f"Charging model not found: {charging_path}"
    assert "charging_A1_50k_seed7" in str(charging_path)

    assert driving_path is not None, (
        "Driving PPO model was not found under final_models/"
    )
    assert driving_path.exists(), f"Driving model not found: {driving_path}"
    assert "driving_B3_100k_seed7" in str(driving_path)


def test_visualizer_cycle_resolution(simulator):
    """All four standard driving cycles should resolve to existing CSV files."""
    assert len(simulator.cycle_paths) == 4

    labels = [label for label, _ in simulator.cycle_paths]
    assert labels == ["UDDS", "HWFET", "US06", "WLTP"]

    for label, path in simulator.cycle_paths:
        assert isinstance(path, Path)
        assert path.exists(), f"Drive cycle for {label} not found: {path}"
        assert path.is_file(), f"Drive cycle path is not a file for {label}: {path}"
        assert path.suffix.lower() == ".csv"


# -----------------------------------------------------------------------------
# Environment lifecycle
# -----------------------------------------------------------------------------


def test_visualizer_initial_state(simulator):
    """The simulator should start in charging/PPO mode with a live environment."""
    assert simulator.ui.mode == "charging"
    assert simulator.ui.controller == "ppo"
    assert simulator.env is not None
    assert simulator.obs is not None
    assert simulator.done is False
    assert simulator.sim_time == pytest.approx(0.0)


def test_visualizer_reset(simulator):
    """Reset should initialize state and add an initial trace point."""
    simulator._reset_env()

    assert simulator.env is not None
    assert simulator.obs is not None
    assert isinstance(simulator.info, dict)
    assert simulator.done is False
    assert simulator.sim_time == pytest.approx(0.0)
    assert len(simulator.trace_soc.y) >= 1
    assert len(simulator.trace_temp.y) >= 1
    assert len(simulator.trace_power.y) >= 1
    assert len(simulator.trace_speed.y) >= 1
    assert len(simulator.trace_action.y) >= 1


def test_visualizer_step(simulator):
    """A successful simulator step should advance time and append one trace point."""
    simulator._reset_env()

    initial_time = simulator.sim_time
    initial_trace_len = len(simulator.trace_soc.y)

    simulator._step_once()

    # A normal real environment step must advance time. If the environment
    # terminates immediately, the test still requires a trace point.
    assert simulator.sim_time > initial_time
    assert len(simulator.trace_soc.y) == initial_trace_len + 1
    assert len(simulator.trace_temp.y) == initial_trace_len + 1
    assert len(simulator.trace_power.y) == initial_trace_len + 1
    assert len(simulator.trace_action.y) == initial_trace_len + 1


# -----------------------------------------------------------------------------
# Mode/controller switching
# -----------------------------------------------------------------------------


def test_visualizer_mode_switch(simulator):
    """Switching modes should rebuild the real environment and reset time."""
    assert simulator.ui.mode == "charging"
    assert simulator.env is not None

    simulator.toggle_mode()

    assert simulator.ui.mode == "driving"
    assert simulator.env is not None
    assert simulator.model is not None or simulator.baseline is not None
    assert simulator.sim_time == pytest.approx(0.0)
    assert simulator.ui.controller == "ppo"

    simulator.toggle_mode()

    assert simulator.ui.mode == "charging"
    assert simulator.env is not None
    assert simulator.model is not None or simulator.baseline is not None
    assert simulator.sim_time == pytest.approx(0.0)
    assert simulator.ui.controller == "ppo"


def test_visualizer_controller_switch(simulator):
    """Switch between PPO and the correct baseline controller in both modes."""
    # Charging mode: PPO -> MaxCurrentController -> PPO
    assert simulator.ui.mode == "charging"
    assert simulator.ui.controller == "ppo"
    assert simulator.model is not None

    simulator.toggle_controller()
    assert simulator.ui.controller == "baseline"
    assert simulator.model is None
    assert simulator.baseline is not None

    simulator.toggle_controller()
    assert simulator.ui.controller == "ppo"
    assert simulator.model is not None
    assert simulator.baseline is None

    # Driving mode: PPO -> RuleBasedEMS
    simulator.toggle_mode()
    assert simulator.ui.mode == "driving"
    assert simulator.ui.controller == "ppo"
    assert simulator.model is not None

    simulator.toggle_controller()
    assert simulator.ui.controller == "baseline"
    assert simulator.model is None
    assert simulator.baseline is not None

    simulator.toggle_controller()
    assert simulator.ui.controller == "ppo"
    assert simulator.model is not None
    assert simulator.baseline is None


# -----------------------------------------------------------------------------
# Environment-wrapper / drive-cycle access
# -----------------------------------------------------------------------------


def test_drive_cycle_access_in_driving_mode(simulator):
    """Driving mode must expose the real cycle even when Gym wrappers are present."""
    simulator.ui.mode = "driving"
    simulator.ui.controller = "ppo"
    simulator._load_mode()
    simulator._reset_env()

    cycle = simulator._get_drive_cycle()

    assert cycle is not None, (
        "Could not find _drive_cycle/drive_cycle through the environment wrappers"
    )


def test_env_attribute_lookup(simulator):
    """_get_env_attr should search the outer environment and nested wrappers."""
    # These attributes are used by the simulator itself and should be available
    # after constructing the charging environment.
    dt = simulator._get_env_attr("dt", default=None)
    assert dt is not None
    assert np.isfinite(float(dt))


# -----------------------------------------------------------------------------
# Metric extraction
# -----------------------------------------------------------------------------


def test_visualizer_temperature_extraction(simulator):
    """Temperature extraction should return a finite, physically reasonable value."""
    simulator._reset_env()
    temp = simulator._get_temperature()

    assert isinstance(temp, float)
    assert np.isfinite(temp)
    assert 0.0 < temp < 100.0


def test_visualizer_soc_extraction(simulator):
    """SOC extraction should return a finite fraction between 0 and 1."""
    simulator._reset_env()
    soc = simulator._get_soc()

    assert isinstance(soc, float)
    assert np.isfinite(soc)
    assert 0.0 <= soc <= 1.0


def test_visualizer_speed_extraction(simulator):
    """Charging speed is zero; driving speed must be finite and non-negative."""
    # Charging has no drive cycle.
    simulator.ui.mode = "charging"
    simulator._load_mode()
    simulator._reset_env()

    charging_speed = simulator._get_speed()
    assert isinstance(charging_speed, float)
    assert charging_speed == pytest.approx(0.0)

    # Driving should use the real drive cycle.
    simulator.ui.mode = "driving"
    simulator.ui.controller = "ppo"
    simulator._load_mode()
    simulator._reset_env()

    initial_speed = simulator._get_speed()
    assert isinstance(initial_speed, float)
    assert np.isfinite(initial_speed)
    assert initial_speed >= 0.0

    simulator._step_once()

    speed = simulator._get_speed()
    assert isinstance(speed, float)
    assert np.isfinite(speed)
    assert speed >= 0.0


def test_visualizer_power_extraction(simulator):
    """Power extraction should always return a finite kW value."""
    simulator._reset_env()
    simulator._step_once()

    power = simulator._get_power()

    assert isinstance(power, float)
    assert np.isfinite(power)


def test_visualizer_regen_extraction_in_driving(simulator):
    """Regen extraction should return a finite non-negative kW value."""
    simulator.ui.mode = "driving"
    simulator.ui.controller = "ppo"
    simulator._load_mode()
    simulator._reset_env()
    simulator._step_once()

    regen = simulator._get_regen()

    assert isinstance(regen, float)
    assert np.isfinite(regen)
    assert regen >= 0.0


def test_current_metrics_charging(simulator):
    """Charging mode metrics should expose the expected keys and finite values."""
    simulator.ui.mode = "charging"
    simulator.ui.controller = "ppo"
    simulator._load_mode()
    simulator._reset_env()

    metrics = simulator._current_metrics()

    expected = {
        "SOC",
        "Temperature",
        "Ambient",
        "Voltage",
        "Applied Current",
        "Target",
        "Step",
        "Safety",
    }

    assert expected.issubset(metrics.keys())
    assert np.isfinite(float(metrics["SOC"]))
    assert np.isfinite(float(metrics["Temperature"]))
    assert np.isfinite(float(metrics["Ambient"]))


def test_current_metrics_driving(simulator):
    """Driving mode metrics should expose speed, power, regen and deficit."""
    simulator.ui.mode = "driving"
    simulator.ui.controller = "ppo"
    simulator._load_mode()
    simulator._reset_env()
    simulator._step_once()

    metrics = simulator._current_metrics()

    expected = {
        "SOC",
        "Temperature",
        "Ambient",
        "Speed",
        "Battery Power",
        "Regen",
        "Deficit Wh",
        "Safety",
    }

    assert expected.issubset(metrics.keys())

    for key in ("SOC", "Temperature", "Ambient", "Speed", "Battery Power", "Regen", "Deficit Wh"):
        assert np.isfinite(float(metrics[key])), f"Non-finite metric: {key}"

    assert metrics["Speed"] >= 0.0
    assert metrics["Regen"] >= 0.0


# -----------------------------------------------------------------------------
# Cycle selection
# -----------------------------------------------------------------------------


def test_cycle_selection(simulator):
    """All four keyboard-equivalent cycle selections should map correctly."""
    simulator.ui.mode = "driving"

    expected = ["UDDS", "HWFET", "US06", "WLTP"]

    for index, label in enumerate(expected):
        simulator.set_cycle(index)
        assert simulator.ui.cycle_index == index
        assert simulator.cycle_paths[index][0] == label
        assert simulator.env is not None
        assert simulator.sim_time == pytest.approx(0.0)


# -----------------------------------------------------------------------------
# Safety checks for malformed data
# -----------------------------------------------------------------------------


def test_metric_helpers_are_resilient_to_missing_info(simulator):
    """Metric helpers should not crash when info is temporarily empty."""
    simulator.info = {}

    temperature = simulator._get_temperature()
    soc = simulator._get_soc()
    speed = simulator._get_speed()
    power = simulator._get_power()
    regen = simulator._get_regen()

    for value in (temperature, soc, speed, power, regen):
        assert isinstance(value, float)
        assert np.isfinite(value)

    assert speed >= 0.0
    assert regen >= 0.0
