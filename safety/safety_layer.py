"""
Shared rule-based safety layer.

Applied identically to the PPO controller and every baseline controller,
guaranteeing fair comparison. Pure function: no hidden state, no side
effects other than the returned intervention info (caller is responsible
for logging).

Interface:
    safe_current, info = safety_layer(requested_current, state, config)

SAFETY LAYER v2 (semantics fix): computes the safe current CEILING first
(i_max * derating_multiplier), then clamps the request against that ceiling
directly — applied = min(requested, ceiling). The original v1 implementation
capped the request at i_max FIRST, then multiplied by the derating factor
(applied = min(requested, i_max) * mult), which double-derated any request
below i_max: requesting exactly the ceiling got derated a second time on top
of its own reduction, making "always request >= i_max" the unique way to
reach the true ceiling and making any self-limiting strategy actively worse
than just maxing out — regardless of reward shaping. v2 fixes this: the
applied current is now monotonically non-decreasing in the requested current
(see tests/test_safety.py monotonicity tests), so a controller that requests
exactly the safe ceiling now actually receives the safe ceiling.

NOTE: results from runs generated before this fix (see project run history
prior to run_008) were produced under v1's double-derating semantics and are
not directly comparable to v2 results — all controllers (CC/CCCV/Adaptive/
PPO) experienced the old, non-monotonic clamp behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SafetyInfo:
    requested_current: float
    safe_current_ceiling: float    # i_max * combined derating multiplier — the true allowed ceiling
    applied_current: float
    intervention_type: str          # "none", "current_limit", "temperature", "voltage_taper", "soc_taper"
    magnitude: float                 # normalized [0,1]: how much the request was reduced
    derating_multiplier: float       # combined multiplier actually applied (1.0 = no derating)

    def as_dict(self) -> Dict:
        return {
            "requested_current": self.requested_current,
            "safe_current_ceiling": self.safe_current_ceiling,
            "applied_current": self.applied_current,
            "type": self.intervention_type,
            "magnitude": self.magnitude,
            "derating_multiplier": self.derating_multiplier,
        }

    @property
    def intervened(self) -> bool:
        return self.intervention_type != "none"


def _linear_derate(value: float, start: float, full: float) -> float:
    """Return a multiplier in [0,1]: 1.0 below `start`, 0.0 at/above `full`, linear between."""
    if full <= start:
        return 1.0 if value < start else 0.0
    if value <= start:
        return 1.0
    if value >= full:
        return 0.0
    return 1.0 - (value - start) / (full - start)


def state_based_current_multiplier(state, config: Dict) -> float:
    """The fraction of i_max the safety layer currently permits, based only
    on state (temperature, SoC) — no request or voltage estimate needed.

    Voltage tapering is deliberately excluded here: its multiplier depends
    on the terminal voltage AT the requested current (a circular
    dependency), and in this system's dynamics the voltage taper zone
    heavily overlaps the SoC taper zone near full charge, so SoC alone is
    already a strong proxy. Used both as an observation feature (so the
    policy has direct access to "how much can I safely ask for right now"
    instead of having to infer it) and to compute the over-request penalty
    in the reward function.
    """
    soc = state.soc if hasattr(state, "soc") else state["soc"]
    temp = state.temperature_c if hasattr(state, "temperature_c") else state["temperature_c"]

    temp_mult = _linear_derate(temp, config["t_derate_start_c"], config["t_hard_cutoff_c"])
    soc_mult = _linear_derate(soc, config["soc_taper_start"], config["soc_taper_full"])
    return min(temp_mult, soc_mult)


def safety_layer(requested_current_a: float, state, config: Dict, estimated_voltage: float = None):
    """Clamp a requested charging current to keep the battery within safe bounds.

    v2 semantics: computes the safe ceiling (i_max * combined derating
    multiplier) first, then clamps the request against that ceiling.
    Monotonic in the request: applied_current never decreases as
    requested_current_a increases, for fixed state/estimated_voltage.

    Args:
        requested_current_a: current requested by the controller (A), charging positive.
        state: object/dict with attributes/keys `soc` and `temperature_c` (a BatteryState works).
        config: safety.yaml loaded as a dict.
        estimated_voltage: optional pre-computed terminal voltage at the requested current,
            used for voltage tapering. If None, voltage tapering is skipped (caller can
            re-check post-hoc, or this arg can be supplied by the environment which has
            the ECM available to estimate it).

    Returns:
        (safe_current_a, SafetyInfo)
    """
    soc = state.soc if hasattr(state, "soc") else state["soc"]
    temp = state.temperature_c if hasattr(state, "temperature_c") else state["temperature_c"]

    i_max = config["i_max_a"]

    # --- Compute each rule's multiplier independently (pure state/estimate, no request involved) ---
    temp_mult = _linear_derate(temp, config["t_derate_start_c"], config["t_hard_cutoff_c"])
    soc_mult = _linear_derate(soc, config["soc_taper_start"], config["soc_taper_full"])

    volt_mult = 1.0
    if estimated_voltage is not None:
        volt_mult = _linear_derate(estimated_voltage, config["v_taper_start"], config["v_hard_max"])

    # Most restrictive rule wins (min, not product — avoids unrealistic
    # compounding when multiple mild derates are simultaneously active).
    combined_mult = min(temp_mult, soc_mult, volt_mult)

    # --- The actual safe ceiling, computed ONCE, independent of the request ---
    safe_ceiling = i_max * combined_mult

    # --- Clamp the request against the ceiling directly (monotonic) ---
    applied_current = max(0.0, min(requested_current_a, safe_ceiling))

    # --- Determine which rule (if any) is responsible for the binding constraint ---
    if applied_current < requested_current_a - 1e-9:
        if combined_mult >= 1.0:
            intervention_type = "current_limit"
        elif combined_mult == temp_mult:
            intervention_type = "temperature"
        elif combined_mult == soc_mult:
            intervention_type = "soc_taper"
        else:
            intervention_type = "voltage_taper"
    else:
        intervention_type = "none"

    magnitude = 0.0
    if requested_current_a > 1e-9:
        magnitude = max(0.0, 1.0 - (applied_current / requested_current_a))

    info = SafetyInfo(
        requested_current=requested_current_a,
        safe_current_ceiling=safe_ceiling,
        applied_current=applied_current,
        intervention_type=intervention_type,
        magnitude=magnitude,
        derating_multiplier=combined_mult,
    )
    return applied_current, info

def state_based_discharge_multiplier(state, config: Dict) -> float:
    """Discharge-side counterpart to state_based_current_multiplier(): the
    fraction of discharge_i_max_a currently permitted, based on state
    (temperature -- reused from the charging-side rule since heat
    generation is symmetric; SoC -- a new low-SoC taper, mirroring the
    existing high-SoC taper's shape). New, additive -- does not affect
    state_based_current_multiplier() or safety_layer()."""
    soc = state.soc if hasattr(state, "soc") else state["soc"]
    temp = state.temperature_c if hasattr(state, "temperature_c") else state["temperature_c"]

    temp_mult = _linear_derate(temp, config["t_derate_start_c"], config["t_hard_cutoff_c"])
    # Low-SoC taper: symmetric to the existing soc_taper (which ramps
    # 1.0->0.0 as soc rises from soc_taper_start to soc_taper_full).
    # Here we want 1.0->0.0 as soc FALLS from soc_discharge_taper_start
    # to soc_discharge_empty -- implemented by calling the same
    # _linear_derate helper on (1 - soc) against the mirrored thresholds.
    soc_discharge_mult = _linear_derate(
        1.0 - soc, 1.0 - config["soc_discharge_taper_start"], 1.0 - config["soc_discharge_empty"]
    )
    return min(temp_mult, soc_discharge_mult)


def safety_layer_bidirectional(requested_current_a: float, state, config: Dict, estimated_voltage: float = None):
    """Bidirectional safety layer for the driving-EMS extension. NEW,
    additive function -- does not modify safety_layer() or any of its
    behavior. For any non-negative request, delegates to safety_layer()
    unchanged (byte-for-byte identical charging behavior, verified in
    tests/test_safety_bidirectional.py). Only the negative-current
    (discharge) branch is new logic.

    Sign convention: same as the rest of the project -- positive current
    = charging (into the battery), negative = discharge (out of the
    battery, e.g. propulsion or as the destination for recovered
    regenerative power after environment/drivetrain_model.py has already
    computed how much is available -- this function only enforces what
    the BATTERY can safely accept, it doesn't know about vehicle/motor
    limits).

    Returns:
        (applied_current_a, SafetyInfo) -- same shape as safety_layer(),
        with intervention_type possibly "discharge_current_limit",
        "discharge_temperature", "discharge_soc_taper", or
        "discharge_voltage_taper" for the new branch.
    """
    if requested_current_a >= 0.0:
        return safety_layer(requested_current_a, state, config, estimated_voltage)

    soc = state.soc if hasattr(state, "soc") else state["soc"]
    temp = state.temperature_c if hasattr(state, "temperature_c") else state["temperature_c"]

    discharge_i_max = config["discharge_i_max_a"]

    temp_mult = _linear_derate(temp, config["t_derate_start_c"], config["t_hard_cutoff_c"])
    soc_discharge_mult = _linear_derate(
        1.0 - soc, 1.0 - config["soc_discharge_taper_start"], 1.0 - config["soc_discharge_empty"]
    )

    volt_mult = 1.0
    if estimated_voltage is not None:
        # Undervoltage taper: mirrors the charging-side overvoltage taper,
        # but ramps down as voltage FALLS toward v_hard_min instead of
        # rising toward v_hard_max. _linear_derate(value, start, full)
        # returns 1.0 below start / 0.0 at-or-above full -- for a falling
        # quantity we mirror it the same way state_based_discharge_multiplier
        # mirrors the SoC taper above.
        volt_mult = _linear_derate(
            -estimated_voltage, -config["v_undervoltage_taper_start"], -config["v_hard_min"]
        )

    combined_mult = min(temp_mult, soc_discharge_mult, volt_mult)
    safe_ceiling_magnitude = discharge_i_max * combined_mult  # magnitude, i.e. >= 0
    requested_magnitude = -requested_current_a  # positive

    applied_magnitude = max(0.0, min(requested_magnitude, safe_ceiling_magnitude))
    applied_current = -applied_magnitude

    if applied_magnitude < requested_magnitude - 1e-9:
        if combined_mult >= 1.0:
            intervention_type = "discharge_current_limit"
        elif combined_mult == temp_mult:
            intervention_type = "discharge_temperature"
        elif combined_mult == soc_discharge_mult:
            intervention_type = "discharge_soc_taper"
        else:
            intervention_type = "discharge_voltage_taper"
    else:
        intervention_type = "none"

    magnitude = 0.0
    if requested_magnitude > 1e-9:
        magnitude = max(0.0, 1.0 - (applied_magnitude / requested_magnitude))

    info = SafetyInfo(
        requested_current=requested_current_a,
        safe_current_ceiling=-safe_ceiling_magnitude,
        applied_current=applied_current,
        intervention_type=intervention_type,
        magnitude=magnitude,
        derating_multiplier=combined_mult,
    )
    return applied_current, info
