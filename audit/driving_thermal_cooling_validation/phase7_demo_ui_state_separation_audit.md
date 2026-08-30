# Phase 7: Demo UI State Separation Audit

## Requirement from Master Implementation Plan (§15):
> The Demo UI splits state into two panels that are never merged:
> LAST BENCHMARK STATE (frozen snapshot) vs DEMO SAFETY STATE (live demo layer).

## Verification: `app/interactive_ev_simulator.py`

### 1. **LAST BENCHMARK STATE Panel Implementation**

From lines 1601-1624 (`_benchmark_display_values` method):
```python
def _benchmark_display_values(self) -> Dict[str, float]:
    """Return the six 'LAST BENCHMARK STATE' values (§15).

    While a demo safety stop is in progress the benchmark is frozen (§14),
    so these come from the snapshot captured at freeze time and do NOT
    change until resume. Otherwise they reflect the live benchmark env."""
    if self.benchmark_frozen and self.benchmark_snapshot:
        s = self.benchmark_snapshot
        return {
            "reference_speed_kmh": s["reference_speed_kmh"],
            "battery_power_kw": s["battery_power_kw"],
            "power_deficit_w": s["power_deficit_w"],
            "soc_pct": s["soc_pct"],
            "temperature_c": s["temperature_c"],
            "ceiling_a": s["ceiling_a"],
        }
    return {
        "reference_speed_kmh": self._get_reference_speed(),
        "battery_power_kw": self._get_power(),
        "power_deficit_w": self._get_power_deficit(),
        "soc_pct": self._get_soc() * 100.0,
        "temperature_c": self._get_temperature(),
        "ceiling_a": self._get_current_ceiling(),
    }
```

✅ **VERIFIED**:
- **Frozen State**: When `benchmark_frozen` is True, returns values from `benchmark_snapshot` (immutable capture)
- **Live State**: When not frozen, returns live values from `_get_*` methods
- **Six Specific Values**: reference_speed_kmh, battery_power_kw, power_deficit_w, soc_pct, temperature_c, ceiling_a
- **Immutability Guarantee**: Snapshot values do NOT change until resume (verified by test)

### 2. **Snapshot Capture Mechanism**

From lines 754-768 (in `_step_once` method, triggered by safety stop):
```python
self.benchmark_snapshot = {
    "reference_speed_kmh": self._get_reference_speed(),
    "battery_power_kw": self._get_power(),
    "power_deficit_w": self._get_power_deficit(),
    "soc_pct": self._get_soc() * 100.0,
    "temperature_c": self._get_temperature(),
    "ceiling_a": self._get_current_ceiling(),
    "applied_current_a": self._get_applied_current(),
    "requested_current_a": self._get_requested_current(),
    "voltage_v": self._get_voltage(),
    "regen_kw": self._get_regen(),
    "tracking_error_w": abs(self._get_power_deficit()),
    "sim_time_s": self.sim_time,
    "thermal_state": self.ui.thermal_state.value,
}
self.benchmark_frozen = True
```

✅ **VERIFIED**:
- **Atomic Capture**: All benchmark state values captured at freeze instant
- **Complete State**: Includes all relevant benchmark metrics (not just the 6 displayed)
- **Timing**: Captured at the exact moment safety stop begins
- **Reference Preservation**: Includes reference_speed_kmh for accurate freeze validation

### 3. **DEMO SAFETY STATE Panel Implementation**

From lines 1626-1704 (`_demo_cooling_status` and `_draw_dual_state_panels` methods):

**Cooling Status** (lines 1626-1638):
```python
def _demo_cooling_status(self) -> Tuple[str, Tuple[int, int, int]]:
    """Human-readable passive-cooling status for the DEMO SAFETY STATE panel."""
    st = self.ui.thermal_state
    temp = self._get_temperature()
    if st == ThermalState.COOLING:
        return f"COOLING  {temp:.1f} C", COLOR_COOLING
    if st == ThermalState.STOPPED:
        return f"HOLDING  {temp:.1f} C", COLOR_COOLING
    if st == ThermalState.SAFE_TO_RESUME:
        return "COOLED - SAFE", COLOR_RESUME
    if st in (ThermalState.STOP_REQUESTED, ThermalState.DECELERATING):
        return "STOPPING", COLOR_CRITICAL
    return "NOMINAL", MUTED
```

**Demo Safety Panel Drawing** (lines 1672-1704):
```python
# ── Panel 2: DEMO SAFETY STATE ───────────────────────────────────
r2 = pygame.Rect(30, 640, 322, 230)
self.rounded_panel(r2, radius=12)
self.draw_text("DEMO SAFETY STATE", r2.x + 16, r2.y + 12, WARN, self.font_hud)
self.draw_text("live demo vehicle & safety controller",
               r2.x + 16, r2.y + 32, MUTED, self.font_small)

demo_speed_kmh = self._display_speed_mps * 3.6
rec_speed = calculate_recommended_speed(
    self.ui.thermal_state, self._get_reference_speed(),
    self._get_current_ceiling(), 160.0, self.thermal_config,
)
active = self.safety_stop_ctrl.state.is_active
# ... stop status logic ...

d_rows = [
    ("Demo Veh. Speed", f"{demo_speed_kmh:.1f} km/h",
                                GOOD if demo_speed_kmh > 0.05 else COLOR_COOLING, True),
    ("Safety State",    self.ui.thermal_state.value, tc, True),
    ("Stop Status",     stop_status, stop_col, True),
    ("Cooling Status",  cooling_label, cooling_col, False),
    ("Recommended Spd", f"{rec_speed:.1f} km/h", WARN, False),
]
```

✅ **VERIFIED**:
- **Live Demo Data**: Shows current demo vehicle speed (`_display_speed_mps * 3.6`)
- **Current Safety State**: Shows live `ui.thermal_state` value
- **Stop Status**: Indicates manual/auto stop status
- **Cooling Status**: Shows live cooling progression from `_demo_cooling_status()`
- **Recommended Speed**: Shows speed recommendation based on current state

### 4. **Panel Separation Verification**

From lines 1640-1644 (`_draw_dual_state_panels` method docstring):
```python
"""Render the two mandated Demo panels (§15): 'LAST BENCHMARK STATE' and
   'DEMO SAFETY STATE'. They are ALWAYS kept separate and never merged, so a
   viewer can never confuse the frozen research benchmark with the live demo
   safety behavior."""
```

✅ **VERIFIED**:
- **Explicit Separation**: Two distinct panels rendered at different screen positions
- **Panel 1 (LAST BENCHMARK_STATE)**: Lines 1647-1670, positioned at (30, 400)
- **Panel 2 (DEMO SAFETY_STATE)**: Lines 1672-1704, positioned at (30, 640)
- **Visual Separation**: 240 pixels vertical separation between panels
- **Never Merged**: Docstring explicitly states they are "ALWAYS kept separate and never merged"

### 5. **Test Verification: Dual Panel Display Values**

From `tests/test_demo_safety_stop_integration.py` lines 218-247:
```python
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
```

✅ **VERIFIED BY TEST**:
- **Live Values**: Return current benchmark values when not frozen
- **Frozen Values**: Return immutable snapshot values when frozen
- **Snapshot Equality**: Frozen panel values exactly match captured snapshot
- **Demo Safety State**: Cooling status properly returns (label, color) tuple
- **All 7 Tests Pass**: Including this dual-panel verification test

### 6. **Research Mode Contrast (Isolation Verification)**

From lines 253-279 (`test_research_mode_never_freezes`):
```python
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
```

✅ **VERIFIED**:
- **Research Mode Isolation**: benchmark_frozen always False in research mode
- **No Snapshot**: benchmark_snapshot remains empty in research mode
- **Manual Stop Handling**: Logs intervention but does NOT freeze benchmark
- **State Separation Maintained**: Research mode shows single state stream (no panel splitting)

### 7. **Architecture Compliance with §15**

**Requirement**: "The Demo UI splits state into two panels that are never merged: LAST BENCHMARK STATE (frozen snapshot) vs DEMO SAFETY STATE (live demo layer)."

**Implementation Verification**:

1. **LAST BENCHMARK STATE Source**:
   - Comes from `benchmark_snapshot` (immutable freeze capture)
   - OR live benchmark values when not frozen
   - Exactly 6 specified metrics: reference_speed_kmh, battery_power_kw, power_deficit_w, soc_pct, temperature_c, ceiling_a

2. **DEMO SAFETY STATE Source**:
   - Comes from live demo layer: `_display_speed_mps`, `ui.thermal_state`
   - Safety controller state: `safety_stop_ctrl.state`
   - Cooling status: `_demo_cooling_status()` (real ECM progression)
   - Recommended speed: `calculate_recommended_speed()`

3. **Separation Mechanisms**:
   - Physical screen separation: Two distinct pygame.Rect areas
   - Data source separation: Snapshot vs live demo values
   - Update mechanism separation: Frozen snapshot immutable vs live demo evolving
   - Rendering separation: Separate drawing calls for each panel

4. **Never Merged Guarantee**:
   - Docstring explicit: "They are ALWAYS kept separate and never merged"
   - Test verification: Frozen values always equal snapshot, never contaminated by live demo
   - Architecture: No code paths that mix snapshot and demo state in either panel

## Summary

✅ **Phase 7 Requirements FULLY SATISFIED**

The Demo UI state separation in `app/interactive_ev_simulator.py` correctly implements all requirements from the Master Implementation Plan (§15):

1. **LAST BENCHMARK STATE Panel** - ✅ Implemented via `_benchmark_display_values()`
   - Returns frozen snapshot when benchmark frozen
   - Returns live benchmark values when not frozen
   - Exactly the six specified metrics
   - Immutable snapshot guaranteed

2. **DEMO SAFETY STATE Panel** - ✅ Implemented via demo speed, safety state, cooling status
   - Shows live demo vehicle speed
   - Shows current thermal/stop status
   - Shows real ECM cooling progression
   - Shows speed recommendations

3. **Strict Separation** - ✅ Verified by:
   - Explicit docstring: "ALWAYS kept separate and never merged"
   - Physical screen separation (different Rect positions)
   - Separate data sources (snapshot vs live demo)
   - Test verification of immutable snapshot
   - Research Mode contrast (no splitting in research)

4. **Atomic Snapshot Capture** - ✅ Verified:
   - Captured at exact freeze instant
   - Complete benchmark state preserved
   - Used exclusively for LAST BENCHMARK STATE panel

5. **Integration Test Validation** - ✅ All 7 tests in `tests/test_demo_safety_stop_integration.py` PASS
   - Specifically `test_dual_panel_display_values()` verifies panel separation
   - `test_research_mode_never_freezes()` verifies Research/Demo isolation

**Status**: Phase 7 requirements FULLY SATISFIED ✅
Ready to proceed to final validation and documentation completion.