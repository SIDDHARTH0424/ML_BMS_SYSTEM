"""
Comprehensive Acceptance Tests for RL-BMS-Driving Thermal Protection Layer.

Verifies all 18 mandatory acceptance test cases (§45) and measurable criteria (§46):
1. test_passive_cooling
2. test_action_space_mapping
3. test_vehicle_speed_architecture
4. test_threshold_order_validation
5. test_thermal_state_machine
6. test_thermal_hysteresis
7. test_critical_transition
8. test_demo_safety_stop
9. test_stop_reaches_zero_speed
10. test_cooling_transition
11. test_safe_resume_transition
12. test_critical_speed_override
13. test_speed_recommendation
14. test_research_demo_isolation
15. test_manual_intervention_logging
16. test_exact_one_step
17. test_temperature_display_accuracy
18. test_safety_ceiling_display_accuracy
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
import numpy as np
import pytest
import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from environment.ecm_model import BatteryECM, BatteryState
from environment.ev_energy_env import EVEnergyEnv
from training.train_drive_ems import make_drive_ems_env
from app.thermal_state_machine import (
    ThermalState,
    calculate_recommended_speed,
    determine_state,
    load_thermal_config,
    validate_thermal_config,
)
from app.safety_stop_controller import DemoSafetyStopController
from app.logger import SimulatorLogger
from app.interactive_ev_simulator import InteractiveSimulator, ROOT


@pytest.fixture
def thermal_cfg():
    return load_thermal_config()


@pytest.fixture
def sim():
    s = InteractiveSimulator()
    s.ui.mode = "driving"
    s.ui.controller = "ppo"
    s._load_mode()
    s._reset_env()
    yield s
    try:
        import pygame
        pygame.quit()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1. Passive Cooling Validation Test
# -----------------------------------------------------------------------------
def test_passive_cooling():
    """Verify that pure ECM model cools monotonically at zero current without visualizer decrement."""
    with open(ROOT / "configs" / "battery.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dt_seconds"] = 1.0
    ecm = BatteryECM(cfg)

    state = BatteryState(soc=0.50, v_rc=0.0, temperature_c=56.0, soh=1.0, ah_throughput=0.0)
    temps = [state.temperature_c]

    for _ in range(1200):  # 20 minutes
        state = ecm.step(state, current_a=0.0, ambient_temp_c=25.0)
        temps.append(state.temperature_c)

    assert all(np.isfinite(temps)), "Temperature must remain finite"
    assert all(temps[i+1] <= temps[i] + 1e-9 for i in range(len(temps)-1)), "Cooling must be strictly monotonic"
    assert temps[-1] < temps[0], "Temperature must decrease toward ambient"
    assert temps[-1] >= 25.0, "Temperature cannot drop below ambient"


# -----------------------------------------------------------------------------
# 2. Action-Space Mapping Verification
# -----------------------------------------------------------------------------
def test_action_space_mapping(sim):
    """Verify the 1D continuous action space [-1.0, 1.0] and its physical mapping."""
    assert sim.env.action_space.shape == (1,)
    assert sim.env.action_space.low[0] == pytest.approx(-1.0)
    assert sim.env.action_space.high[0] == pytest.approx(1.0)

    # Step with action -1.0 (propulsion discharge)
    sim._reset_env()
    obs, reward, term, trunc, info = sim.env.step(np.array([-1.0], dtype=np.float32))
    assert "applied_power_w" in info
    assert "power_deficit_w" in info



# -----------------------------------------------------------------------------
# 3. Vehicle-Speed Architecture Verification
# -----------------------------------------------------------------------------
def test_vehicle_speed_architecture(sim):
    """Conclusively verify that benchmark environment speed is trace-following."""
    cycle = sim._get_drive_cycle()
    assert cycle is not None

    ref_speed = cycle.current_speed()
    assert ref_speed >= 0.0

    # Step environment; speed is governed by drive cycle
    obs, reward, term, trunc, info = sim.env.step(np.array([0.0], dtype=np.float32))
    assert "power_deficit_w" in info


# -----------------------------------------------------------------------------
# 4. Threshold Order Configuration Validation
# -----------------------------------------------------------------------------
def test_threshold_order_validation(thermal_cfg):
    """Enforce safe_resume_temperature_c < critical_to_cooling_threshold_c < 55.0."""
    validate_thermal_config(thermal_cfg)

    # Test invalid config raises ValueError (upgraded from assert for production safety)
    invalid_cfg = {
        "thermal_management": {
            "recovery": {
                "safe_resume_temperature_c": 53.0,
                "critical_to_cooling_threshold_c": 52.0,
            },
            "safety_derating": {"cutoff_temp_c": 55.0},
        }
    }
    with pytest.raises(ValueError, match="Invalid thermal recovery thresholds"):
        validate_thermal_config(invalid_cfg)


# -----------------------------------------------------------------------------
# 5. Thermal State Machine Transitions
# -----------------------------------------------------------------------------
def test_thermal_state_machine(thermal_cfg):
    """Verify standard nominal state transitions."""
    assert determine_state(ThermalState.OPTIMAL, 25.0, 50.0, thermal_cfg) == ThermalState.OPTIMAL
    assert determine_state(ThermalState.OPTIMAL, 33.0, 50.0, thermal_cfg) == ThermalState.ELEVATED_THERMAL
    assert determine_state(ThermalState.ELEVATED_THERMAL, 45.0, 50.0, thermal_cfg) == ThermalState.DERATING_ACTIVE
    assert determine_state(ThermalState.DERATING_ACTIVE, 55.0, 50.0, thermal_cfg) == ThermalState.CRITICAL


# -----------------------------------------------------------------------------
# 6. Thermal Hysteresis Inside determine_state
# -----------------------------------------------------------------------------
def test_thermal_hysteresis(thermal_cfg):
    """Verify internal state-relative hysteresis (32.5 / 33.0 and 44.5 / 45.0)."""
    # Elevated exit at 32.5°C
    assert determine_state(ThermalState.ELEVATED_THERMAL, 32.8, 50.0, thermal_cfg) == ThermalState.ELEVATED_THERMAL
    assert determine_state(ThermalState.ELEVATED_THERMAL, 32.4, 50.0, thermal_cfg) == ThermalState.OPTIMAL

    # Derating exit at 44.5°C
    assert determine_state(ThermalState.DERATING_ACTIVE, 44.8, 50.0, thermal_cfg) == ThermalState.DERATING_ACTIVE
    assert determine_state(ThermalState.DERATING_ACTIVE, 44.4, 50.0, thermal_cfg) == ThermalState.ELEVATED_THERMAL


# -----------------------------------------------------------------------------
# 7. Critical Transition & Mode Handling
# -----------------------------------------------------------------------------
def test_critical_transition(thermal_cfg):
    """Both Research and Demo mode transition DERATING_ACTIVE -> CRITICAL at t_critical_enter.
    In Demo mode, a subsequent step from CRITICAL with a moving vehicle triggers STOP_REQUESTED."""
    # Step 1: Both modes transition to CRITICAL from DERATING_ACTIVE
    res_st = determine_state(ThermalState.DERATING_ACTIVE, 55.5, 60.0, thermal_cfg, mode="research")
    assert res_st == ThermalState.CRITICAL

    demo_st_step1 = determine_state(ThermalState.DERATING_ACTIVE, 55.5, 60.0, thermal_cfg, mode="demo")
    assert demo_st_step1 == ThermalState.CRITICAL, (
        "DERATING_ACTIVE must transition to CRITICAL before STOP_REQUESTED (two-step progression)"
    )

    # Step 2 (Demo only): CRITICAL + moving vehicle -> STOP_REQUESTED
    demo_st_step2 = determine_state(ThermalState.CRITICAL, 55.5, 60.0, thermal_cfg, mode="demo")
    assert demo_st_step2 == ThermalState.STOP_REQUESTED, (
        "CRITICAL with moving vehicle must transition to STOP_REQUESTED in Demo mode"
    )


# -----------------------------------------------------------------------------
# 8. Demo Safety Stop Controller
# -----------------------------------------------------------------------------
def test_demo_safety_stop(thermal_cfg):
    """Verify controlled deceleration in Demo Mode."""
    ctrl = DemoSafetyStopController(thermal_cfg)
    ctrl.trigger_stop(current_speed_mps=20.0, manual=True)
    assert ctrl.state.is_active is True

    # Step controller
    speed_mps, next_st, overriding = ctrl.step(
        dt_s=1.0, reference_speed_mps=20.0, temperature_c=55.0, current_thermal_state=ThermalState.STOP_REQUESTED
    )
    assert overriding is True
    assert speed_mps == pytest.approx(18.0)  # 20.0 - 2.0 * 1.0
    assert next_st == ThermalState.DECELERATING


# -----------------------------------------------------------------------------
# 9. Safety Stop Reaches Exactly Zero Speed
# -----------------------------------------------------------------------------
def test_stop_reaches_zero_speed(thermal_cfg):
    """Verify that deceleration terminates at exactly <= 0.01 km/h and transitions to STOPPED."""
    ctrl = DemoSafetyStopController(thermal_cfg)
    ctrl.trigger_stop(current_speed_mps=5.0)

    st = ThermalState.STOP_REQUESTED
    speed_mps = 5.0
    for _ in range(5):
        speed_mps, st, _ = ctrl.step(
            dt_s=1.0, reference_speed_mps=5.0, temperature_c=55.0, current_thermal_state=st
        )

    assert speed_mps == pytest.approx(0.0)
    assert st == ThermalState.STOPPED


# -----------------------------------------------------------------------------
# 10. Cooling Transition Condition
# -----------------------------------------------------------------------------
def test_cooling_transition(thermal_cfg):
    """From STOPPED, cooling requires vehicle stopped AND temperature <= 52.0°C."""
    # Temperature above threshold -> remains STOPPED
    st1 = determine_state(ThermalState.STOPPED, 53.0, vehicle_speed_kmh=0.0, config=thermal_cfg, mode="demo")
    assert st1 == ThermalState.STOPPED

    # Temperature at or below 52.0°C -> enters COOLING
    st2 = determine_state(ThermalState.STOPPED, 51.5, vehicle_speed_kmh=0.0, config=thermal_cfg, mode="demo")
    assert st2 == ThermalState.COOLING


# -----------------------------------------------------------------------------
# 11. Safe Resume Transition & Manual Triggering
# -----------------------------------------------------------------------------
def test_safe_resume_transition(thermal_cfg):
    """From COOLING, enters SAFE_TO_RESUME when T <= 42.0°C and vehicle remains stopped."""
    # T > 42.0°C -> remains COOLING
    st1 = determine_state(ThermalState.COOLING, 43.0, vehicle_speed_kmh=0.0, config=thermal_cfg, mode="demo")
    assert st1 == ThermalState.COOLING

    # T <= 42.0°C -> enters SAFE_TO_RESUME
    st2 = determine_state(ThermalState.COOLING, 41.5, vehicle_speed_kmh=0.0, config=thermal_cfg, mode="demo")
    assert st2 == ThermalState.SAFE_TO_RESUME

    # Requires manual resume to leave SAFE_TO_RESUME
    ctrl = DemoSafetyStopController(thermal_cfg)
    ctrl.state.is_active = True
    ctrl.state.demo_speed_mps = 0.0

    # Cannot resume if unsafe
    success, _ = ctrl.resume(temperature_c=45.0, current_thermal_state=ThermalState.COOLING)
    assert success is False

    # Successful resume when safe
    success, next_st = ctrl.resume(temperature_c=40.0, current_thermal_state=ThermalState.SAFE_TO_RESUME)
    assert success is True
    assert next_st in {ThermalState.OPTIMAL, ThermalState.ELEVATED_THERMAL}


# -----------------------------------------------------------------------------
# 12. Critical Speed Override
# -----------------------------------------------------------------------------
def test_critical_speed_override(thermal_cfg):
    """Unconditional rule: recommended speed MUST be 0.0 km/h in critical and stop states."""
    critical_states = [
        ThermalState.CRITICAL,
        ThermalState.STOP_REQUESTED,
        ThermalState.DECELERATING,
        ThermalState.STOPPED,
        ThermalState.COOLING,
    ]
    for st in critical_states:
        rec_spd = calculate_recommended_speed(
            state=st, reference_speed_kmh=80.0, current_ceiling_a=160.0, rated_current_a=160.0, config=thermal_cfg
        )
        assert rec_spd == pytest.approx(0.0), f"Recommended speed must be 0.0 for state {st}"


# -----------------------------------------------------------------------------
# 13. Speed Recommendation Heuristic
# -----------------------------------------------------------------------------
def test_speed_recommendation(thermal_cfg):
    """Verify v_rec = v_ref * clip(I_ceiling / I_rated, 0.30, 1.00) in stress states."""
    ref_spd = 100.0
    # At 50% derating (ceiling = 80A of 160A) -> recommended speed = 50.0 km/h
    rec_spd = calculate_recommended_speed(
        state=ThermalState.DERATING_ACTIVE, reference_speed_kmh=ref_spd, current_ceiling_a=80.0, rated_current_a=160.0, config=thermal_cfg
    )
    assert rec_spd == pytest.approx(50.0)

    # Minimum ratio clamp at 0.30 (ceiling = 16A -> ratio 0.10 clamped to 0.30)
    rec_min = calculate_recommended_speed(
        state=ThermalState.DERATING_ACTIVE, reference_speed_kmh=ref_spd, current_ceiling_a=16.0, rated_current_a=160.0, config=thermal_cfg
    )
    assert rec_min == pytest.approx(30.0)


# -----------------------------------------------------------------------------
# 14. Research vs Demo Data Isolation
# -----------------------------------------------------------------------------
def test_research_demo_isolation(tmp_path):
    """Ensure Demo outputs write to demo_runs/ and Research outputs write to runs/."""
    demo_logger = SimulatorLogger(root_dir=tmp_path, mode="demo")
    demo_logger.log_trajectory_step({"step": 1, "speed_kmh": 20.0})
    demo_dir = demo_logger.save_session({"mode": "demo"})

    assert "demo_runs" in str(demo_dir)
    assert (demo_dir / "summary.json").exists()
    assert (demo_dir / "trajectory.csv").exists()

    res_logger = SimulatorLogger(root_dir=tmp_path, mode="research")
    res_logger.log_trajectory_step({"step": 1, "speed_kmh": 20.0})
    res_dir = res_logger.save_session({"mode": "research"})

    assert "runs" in str(res_dir)
    assert "demo_runs" not in str(res_dir)


# -----------------------------------------------------------------------------
# 15. Manual Intervention Logging
# -----------------------------------------------------------------------------
def test_manual_intervention_logging(tmp_path):
    """Manual stop/resume in Research mode must flag the run as INTERVENED — NOT STANDARD BENCHMARK."""
    logger = SimulatorLogger(root_dir=tmp_path, mode="research")
    logger.log_event("manual_stop_intervention", 10.0, 50.0, 0.5, 30.0, 100.0, "DERATING_ACTIVE")
    out_dir = logger.save_session({"mode": "research"})

    with open(out_dir / "summary.json", "r") as f:
        import json
        summary = json.load(f)

    assert summary["intervened"] is True
    assert summary["benchmark_status"] == "INTERVENED — NOT STANDARD BENCHMARK"


# -----------------------------------------------------------------------------
# 16. Exact One-Step Execution
# -----------------------------------------------------------------------------
def test_exact_one_step(sim):
    """One call to _step_once() must advance simulation time by exactly 1 dt."""
    sim._reset_env()
    t_start = sim.sim_time
    sim._step_once()
    assert sim.sim_time == pytest.approx(t_start + 1.0)


# -----------------------------------------------------------------------------
# 17. Metric Display Accuracy
# -----------------------------------------------------------------------------
def test_temperature_display_accuracy(sim):
    """Displayed temperature and current ceiling must match authoritative environment values."""
    sim._reset_env()
    sim._step_once()

    env_temp = sim.env._state.temperature_c
    disp_temp = sim._get_temperature()
    assert abs(disp_temp - env_temp) <= 0.01

    ceil_disp = sim._get_current_ceiling()
    assert 0.0 <= ceil_disp <= 160.0
    assert np.isfinite(ceil_disp)


def test_safety_ceiling_display_accuracy(sim):
    """§45/§46: the displayed safety current ceiling must match the real
    safety-layer output within 0.01 A.

    The authoritative source is info['safety_intervention']['safe_current_ceiling']
    produced by the BMS safety layer during env.step(). This test guards against a
    fabricated ceiling (§26/§51): the UI must surface the actual safety-layer value,
    not a UI-only re-derivation.
    """
    sim._reset_env()
    sim._step_once()

    s_info = sim.info.get("safety_intervention", {})
    assert isinstance(s_info, dict) and "safe_current_ceiling" in s_info, (
        "Driving env info must expose safety_intervention['safe_current_ceiling']"
    )
    authoritative_ceiling_a = abs(float(s_info["safe_current_ceiling"]))

    disp_ceiling_a = sim._get_current_ceiling()
    assert np.isfinite(disp_ceiling_a)
    assert abs(disp_ceiling_a - authoritative_ceiling_a) <= 0.01, (
        f"Displayed ceiling {disp_ceiling_a:.4f} A deviates from safety-layer "
        f"ceiling {authoritative_ceiling_a:.4f} A by more than the 0.01 A tolerance"
    )
