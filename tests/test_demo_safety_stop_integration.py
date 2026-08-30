"""
Integration tests for the Demo-Mode Safety Stop + Benchmark Freeze feature (§14, §15).

These exercise the *wired* InteractiveSimulator (not the isolated controller/state
machine, which are covered by test_driving_thermal_acceptance.py). They verify the
end-to-end behavior mandated by the master plan:

  §14  Once a safety stop is requested in Demo Mode, the real EVEnergyEnv benchmark
       is FROZEN: reference-cycle demand, battery power, power deficit and tracking
       error stop evolving while the demo vehicle is visually decelerated/stopped.
       The battery still cools -- via the authoritative ECM passive-cooling model at
       zero current, NEVER a UI-only 'temperature -= 1' decrement (§51).

  §15  The Demo UI splits state into two panels that are never merged:
       LAST BENCHMARK STATE (frozen snapshot) vs DEMO SAFETY STATE (live demo layer).

  Research vs Demo isolation: Research Mode is NEVER frozen and a manual stop there
  is only flagged as an intervention, never a benchmark freeze.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from app.interactive_ev_simulator import InteractiveSimulator, DEMO_STOP_STATES
from app.thermal_state_machine import ThermalState


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------
def _set_state_temp(sim, temp_c: float) -> None:
    """Inject a pack temperature onto the authoritative battery state so cooling
    is observable (the sim starts near ambient). Uses BatteryState.copy() so the
    ECM's own step() still produces the next state normally."""
    for env in sim._walk_envs():
        st = getattr(env, "_state", None)
        if st is not None:
            new = st.copy()
            new.temperature_c = float(temp_c)
            env._state = new
            return
    raise AssertionError("No environment with a battery _state was found")


def _make_demo_sim() -> InteractiveSimulator:
    s = InteractiveSimulator()
    s.ui.mode = "driving"
    s.ui.controller = "ppo"
    s.ui.ambient_c = 25.0
    s.ui.sim_mode = "demo"
    s._load_mode()
    s._reset_env()
    return s


@pytest.fixture
def demo_sim():
    s = _make_demo_sim()
    yield s
    try:
        import pygame
        pygame.quit()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1. Manual demo stop freezes the benchmark and the frozen state does not evolve
# -----------------------------------------------------------------------------
def test_manual_demo_stop_freezes_benchmark(demo_sim):
    """A manual Demo-Mode stop must freeze the benchmark (§14): the drive-cycle
    reference and the captured snapshot must stop evolving while the demo vehicle
    decelerates, yet the battery must still cool via the real ECM."""
    sim = demo_sim
    sim._fast_forward_to_first_motion()
    _set_state_temp(sim, 54.0)  # make the pack hot so cooling is measurable

    assert not sim.benchmark_frozen, "Should not be frozen before a stop is requested"
    moving_speed_mps = sim._display_speed_mps
    assert moving_speed_mps > 0.0, "Vehicle should be moving before the stop"

    sim.trigger_stop_vehicle()

    assert sim.benchmark_frozen is True, "Manual demo stop must freeze the benchmark"
    assert sim.safety_stop_ctrl.state.is_active is True
    assert sim.safety_stop_ctrl.state.manually_stopped is True
    assert sim.ui.thermal_state in DEMO_STOP_STATES

    snapshot_at_freeze = dict(sim.benchmark_snapshot)
    assert snapshot_at_freeze, "A benchmark snapshot must be captured on freeze"

    cycle = sim._get_drive_cycle()
    ref_speed_frozen = cycle.current_speed()
    temp_at_freeze = snapshot_at_freeze["temperature_c"]

    # Advance many frozen steps -- the demo layer evolves, the benchmark does not.
    for _ in range(120):
        sim._step_once()

    # §14: benchmark (drive cycle) did NOT advance while frozen
    assert cycle.current_speed() == pytest.approx(ref_speed_frozen), (
        "Drive-cycle reference must not advance while the benchmark is frozen"
    )
    # §14: the captured snapshot is immutable while frozen
    assert sim.benchmark_snapshot == snapshot_at_freeze, (
        "Frozen benchmark snapshot must not change while stopped"
    )
    # Demo vehicle decelerated to rest (controlled stop)
    assert sim._display_speed_mps == pytest.approx(0.0, abs=0.05)
    # Real ECM passive cooling reduced the temperature (not a UI decrement)
    assert sim._get_temperature() < temp_at_freeze - 0.5, (
        "Battery must cool via the authoritative ECM while stopped"
    )
    # Live benchmark power / deficit are zero while frozen (battery at zero current)
    assert abs(sim._get_power()) < 1e-6
    assert abs(sim._get_power_deficit()) < 1e-6
    # Still frozen until an explicit resume
    assert sim.benchmark_frozen is True


# -----------------------------------------------------------------------------
# 2. Automatic freeze when a thermal safety stop is triggered by high temperature
# -----------------------------------------------------------------------------
def test_auto_thermal_stop_freezes_benchmark(demo_sim):
    """When the pack exceeds the critical cutoff in Demo Mode, the state machine
    must progress DERATING_ACTIVE -> CRITICAL -> STOP_REQUESTED, and the benchmark
    must auto-freeze without any manual action."""
    sim = demo_sim
    sim._fast_forward_to_first_motion()

    froze = False
    # Set initial state and step naturally: DERATING_ACTIVE → CRITICAL (step 1),
    # CRITICAL → STOP_REQUESTED (step 2) → benchmark freezes.
    # We force over-cutoff temperature before every step so the ECM cannot cool
    # it below 55°C within a single timestep.
    _set_state_temp(sim, 60.0)
    sim.ui.thermal_state = ThermalState.DERATING_ACTIVE
    for _ in range(6):
        _set_state_temp(sim, 60.0)
        sim._step_once()
        if sim.benchmark_frozen:
            froze = True
            break

    assert froze is True, "Over-cutoff temperature must auto-freeze the benchmark"
    assert sim.ui.thermal_state in DEMO_STOP_STATES
    assert sim.safety_stop_ctrl.state.is_active is True
    # An automatic (thermal) stop is NOT a manual intervention
    assert sim.safety_stop_ctrl.state.manually_stopped is False


# -----------------------------------------------------------------------------
# 3. Passive cooling advances the REAL ECM state (never a UI decrement) (§51)
# -----------------------------------------------------------------------------
def test_frozen_cooling_uses_real_ecm(demo_sim):
    """_advance_passive_cooling must move the authoritative battery state toward
    ambient at zero current: temperature strictly decreases, SOC is conserved, and
    the temperature never drops below ambient."""
    sim = demo_sim
    _set_state_temp(sim, 55.0)

    ambient = float(sim._get_env_attr("_ambient_temp_c", 25.0))
    state_before = sim._get_env_attr("_state", None)
    t_before = state_before.temperature_c
    soc_before = state_before.soc

    temps = [t_before]
    for _ in range(60):
        sim._advance_passive_cooling(1.0)
        temps.append(sim._get_env_attr("_state", None).temperature_c)

    assert all(np.isfinite(temps))
    assert all(temps[i + 1] <= temps[i] + 1e-9 for i in range(len(temps) - 1)), (
        "Cooling must be monotonic (authoritative ECM, not a UI decrement)"
    )
    assert temps[-1] < t_before, "Temperature must decrease toward ambient"
    assert temps[-1] >= ambient - 1e-6, "Temperature cannot fall below ambient"
    # Zero-current cooling conserves charge
    assert sim._get_env_attr("_state", None).soc == pytest.approx(soc_before, abs=1e-6)


# -----------------------------------------------------------------------------
# 4. Manual resume unfreezes the benchmark and clears the snapshot
# -----------------------------------------------------------------------------
def test_resume_unfreezes_benchmark(demo_sim):
    """After the pack has cooled to a safe-to-resume temperature, a manual resume
    must clear the freeze and the captured snapshot, re-enabling normal stepping."""
    sim = demo_sim
    sim._fast_forward_to_first_motion()
    _set_state_temp(sim, 54.0)
    sim.trigger_stop_vehicle()
    assert sim.benchmark_frozen is True

    # Drive the recovery path: force the pack below the safe-resume threshold and
    # let the frozen demo loop progress STOPPED -> COOLING -> SAFE_TO_RESUME.
    _set_state_temp(sim, 41.0)
    reached_safe = False
    for _ in range(400):
        sim._step_once()
        if sim.ui.thermal_state == ThermalState.SAFE_TO_RESUME:
            reached_safe = True
            break
    assert reached_safe, "Cooling must eventually reach SAFE_TO_RESUME"
    assert sim.benchmark_frozen is True, "Still frozen until the operator resumes"

    sim.trigger_resume()

    assert sim.benchmark_frozen is False, "Resume must unfreeze the benchmark"
    assert sim.benchmark_snapshot == {}, "Resume must clear the frozen snapshot"
    assert sim.safety_stop_ctrl.state.is_active is False


# -----------------------------------------------------------------------------
# 5. Dual-panel display values: snapshot while frozen, live otherwise (§15)
# -----------------------------------------------------------------------------
def test_dual_panel_display_values(demo_sim):
    """_benchmark_display_values feeds the LAST BENCHMARK STATE panel. While frozen
    it must return the immutable snapshot; otherwise the live benchmark values."""
    sim = demo_sim
    sim._fast_forward_to_first_motion()

    expected_keys = {
        "reference_speed_kmh", "battery_power_kw", "power_deficit_w",
        "soc_pct", "temperature_c", "ceiling_a",
    }

    live_vals = sim._benchmark_display_values()
    assert set(live_vals.keys()) == expected_keys
    assert all(np.isfinite(v) for v in live_vals.values())

    _set_state_temp(sim, 54.0)
    sim.trigger_stop_vehicle()
    assert sim.benchmark_frozen is True

    frozen_vals = sim._benchmark_display_values()
    assert set(frozen_vals.keys()) == expected_keys
    for k in expected_keys:
        assert frozen_vals[k] == pytest.approx(sim.benchmark_snapshot[k]), (
            f"Frozen panel value '{k}' must equal the captured snapshot"
        )

    # The DEMO SAFETY STATE cooling status is a (label, color) pair
    label, color = sim._demo_cooling_status()
    assert isinstance(label, str) and label
    assert isinstance(color, tuple) and len(color) == 3


# -----------------------------------------------------------------------------
# 6. Research Mode is never frozen (strict Research/Demo isolation)
# -----------------------------------------------------------------------------
def test_research_mode_never_freezes():
    """Research Mode must never freeze the benchmark, even at over-cutoff
    temperatures, and a manual stop there is an intervention -- not a freeze."""
    s = InteractiveSimulator()
    s.ui.mode = "driving"
    s.ui.controller = "ppo"
    s.ui.ambient_c = 25.0
    s.ui.sim_mode = "research"
    s._load_mode()
    s._reset_env()
    try:
        s._fast_forward_to_first_motion()
        for _ in range(10):
            _set_state_temp(s, 60.0)
            s._step_once()
            assert s.benchmark_frozen is False, "Research Mode must never freeze"

        # Manual stop in Research Mode: flags intervention, does not freeze.
        s.trigger_stop_vehicle()
        assert s.benchmark_frozen is False
        assert s.benchmark_snapshot == {}
    finally:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# 7. A demo stop preserves the underlying research drive cycle (never mutated)
# -----------------------------------------------------------------------------
def test_stop_does_not_mutate_reference_cycle(demo_sim):
    """The demo safe-stop overrides only the *displayed* vehicle speed. The
    underlying drive-cycle reference (the research trace) must be untouched, so
    resuming continues the exact same benchmark."""
    sim = demo_sim
    sim._fast_forward_to_first_motion()
    cycle = sim._get_drive_cycle()

    ref_speed_before = cycle.current_speed()
    _set_state_temp(sim, 54.0)
    sim.trigger_stop_vehicle()

    # Frozen advancement must not touch the reference cycle at all.
    for _ in range(30):
        sim._step_once()

    assert cycle.current_speed() == pytest.approx(ref_speed_before), (
        "The research drive-cycle reference must never be mutated by a demo stop"
    )
    # And the demo override drove the *displayed* speed toward zero.
    assert sim._display_speed_mps <= ref_speed_before + 1e-9
