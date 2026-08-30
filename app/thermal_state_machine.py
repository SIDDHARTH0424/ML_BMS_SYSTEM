"""
Authoritative Thermal State Machine and Driver Guidance for RL-BMS-Driving.

Defines exact 9 thermal states, state-relative hysteresis, driver guidance text,
and physical speed recommendations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple
import yaml
from pathlib import Path


class ThermalState(str, Enum):
    OPTIMAL = "OPTIMAL"
    ELEVATED_THERMAL = "ELEVATED_THERMAL"
    DERATING_ACTIVE = "DERATING_ACTIVE"
    CRITICAL = "CRITICAL"
    STOP_REQUESTED = "STOP_REQUESTED"
    DECELERATING = "DECELERATING"
    STOPPED = "STOPPED"
    COOLING = "COOLING"
    SAFE_TO_RESUME = "SAFE_TO_RESUME"


DRIVER_GUIDANCE: Dict[str, str] = {
    ThermalState.OPTIMAL.value: "NORMAL OPERATION",
    ThermalState.ELEVATED_THERMAL.value: (
        "THERMAL STRESS DETECTED\n\n"
        "Avoid aggressive acceleration.\n"
        "Reduce sustained high-power demand."
    ),
    ThermalState.DERATING_ACTIVE.value: (
        "BATTERY PROTECTION ACTIVE\n\n"
        "Reduce vehicle speed.\n"
        "Reduce sustained power demand."
    ),
    ThermalState.CRITICAL.value: (
        "CRITICAL BATTERY TEMPERATURE\n\n"
        "STOP VEHICLE\n"
        "ALLOW BATTERY TO COOL"
    ),
    ThermalState.STOP_REQUESTED.value: (
        "CRITICAL BATTERY TEMPERATURE\n\n"
        "STOP VEHICLE\n"
        "THERMAL CUTOFF ACTIVE"
    ),
    ThermalState.DECELERATING.value: (
        "VEHICLE DECELERATING\n\n"
        "THERMAL SAFETY STOP IN PROGRESS"
    ),
    ThermalState.STOPPED.value: (
        "VEHICLE STOPPED\n\n"
        "BATTERY PROTECTION ACTIVE"
    ),
    ThermalState.COOLING.value: (
        "BATTERY COOLING\n\n"
        "VEHICLE MUST REMAIN STOPPED"
    ),
    ThermalState.SAFE_TO_RESUME.value: (
        "BATTERY TEMPERATURE SAFE\n\n"
        "PRESS RESUME TO CONTINUE"
    ),
}


def validate_thermal_config(config: Dict[str, Any]) -> None:
    """Validate thermal configuration schema and physical threshold ordering."""
    cfg = config.get("thermal_management", config)
    rec = cfg.get("recovery", {})
    safety = cfg.get("safety_derating", {})
    regions = cfg.get("thermal_regions", {})
    hyst = cfg.get("hysteresis", {})
    spd = cfg.get("speed_recommendation", {})
    demo = cfg.get("demo_stop", {})

    safe_resume = float(rec.get("safe_resume_temperature_c", 42.0))
    crit_to_cooling = float(rec.get("critical_to_cooling_threshold_c", 52.0))
    cutoff_temp = float(safety.get("cutoff_temp_c", 55.0))
    start_temp = float(safety.get("start_temp_c", 45.0))
    rated_current = float(safety.get("rated_current_a", 160.0))

    opt_thresh = float(regions.get("optimal", {}).get("threshold_c", 33.0))
    derate_min = float(regions.get("derating", {}).get("min_temp_c", 45.0))
    crit_min = float(regions.get("critical", {}).get("min_temp_c", 55.0))

    elev_exit = float(hyst.get("elevated_exit_c", 32.5))
    derate_exit = float(hyst.get("derating_exit_c", 44.5))

    min_spd_ratio = float(spd.get("minimum_speed_ratio", 0.30))
    max_spd_ratio = float(spd.get("maximum_speed_ratio", 1.00))

    stop_thresh = float(demo.get("stop_speed_threshold_kmh", 0.01))
    max_decel = float(demo.get("max_deceleration_mps2", 2.0))

    if not (safe_resume < crit_to_cooling < cutoff_temp):
        raise ValueError(
            f"Invalid thermal recovery thresholds: safe_resume_temperature_c ({safe_resume}) "
            f"< critical_to_cooling_threshold_c ({crit_to_cooling}) < cutoff_temp_c ({cutoff_temp})"
        )

    if not (opt_thresh <= derate_min <= crit_min):
        raise ValueError(
            f"Thermal region bounds must be ordered: optimal ({opt_thresh}) <= derating ({derate_min}) <= critical ({crit_min})"
        )

    if not (elev_exit < opt_thresh):
        raise ValueError(
            f"Hysteresis elevated_exit_c ({elev_exit}) must be strictly less than optimal threshold ({opt_thresh})"
        )

    if not (derate_exit < derate_min):
        raise ValueError(
            f"Hysteresis derating_exit_c ({derate_exit}) must be strictly less than derating min ({derate_min})"
        )

    if not (rated_current > 0.0):
        raise ValueError(f"rated_current_a must be positive, got {rated_current}")

    if not (cutoff_temp > start_temp):
        raise ValueError(f"cutoff_temp_c ({cutoff_temp}) must exceed start_temp_c ({start_temp})")

    if not (0.0 <= min_spd_ratio <= 1.0 and 0.0 <= max_spd_ratio <= 1.0 and min_spd_ratio <= max_spd_ratio):
        raise ValueError(
            f"Invalid speed recommendation ratios: 0 <= {min_spd_ratio} <= {max_spd_ratio} <= 1"
        )

    if stop_thresh < 0.0:
        raise ValueError(f"stop_speed_threshold_kmh must be non-negative, got {stop_thresh}")

    if max_decel <= 0.0:
        raise ValueError(f"max_deceleration_mps2 must be positive, got {max_decel}")


def load_thermal_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load and validate thermal configuration."""
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Thermal config file not found: {p}")
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        root = Path(__file__).resolve().parents[1]
        candidates = [
            root / "configs" / "final_driving" / "thermal_management.yaml",
            root / "configs" / "thermal_management.yaml",
        ]
        chosen_path = None
        for c in candidates:
            if c.exists():
                chosen_path = c
                break

        if chosen_path is not None:
            with open(chosen_path, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {
                "thermal_management": {
                    "mode": "research",
                    "thermal_regions": {
                        "optimal": {"threshold_c": 33.0},
                        "elevated_stress": {"min_temp_c": 33.0, "max_temp_c": 45.0},
                        "derating": {"min_temp_c": 45.0, "max_temp_c": 55.0},
                        "critical": {"min_temp_c": 55.0},
                    },
                    "safety_derating": {
                        "rated_current_a": 160.0,
                        "start_temp_c": 45.0,
                        "cutoff_temp_c": 55.0,
                    },
                    "hysteresis": {
                        "elevated_exit_c": 32.5,
                        "derating_exit_c": 44.5,
                    },
                    "recovery": {
                        "critical_to_cooling_threshold_c": 52.0,
                        "safe_resume_temperature_c": 42.0,
                    },
                    "speed_recommendation": {
                        "enabled": True,
                        "active_speed_control": False,
                        "minimum_speed_ratio": 0.30,
                        "maximum_speed_ratio": 1.00,
                    },
                    "cooling": {
                        "source": "ecm",
                        "require_validated_passive_cooling": True,
                    },
                    "demo_stop": {
                        "enabled": True,
                        "max_deceleration_mps2": 2.0,
                        "stop_speed_threshold_kmh": 0.01,
                    },
                }
            }

    validate_thermal_config(cfg)
    return cfg


def determine_state(
    current_state: str | ThermalState,
    temperature_c: float,
    vehicle_speed_kmh: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
    mode: str = "research",
) -> ThermalState:
    """Authoritative state transition function with state-relative hysteresis.

    Hysteresis is evaluated INSIDE this function.
    """
    if isinstance(current_state, ThermalState):
        curr = current_state
    else:
        try:
            curr = ThermalState(str(current_state).upper())
        except ValueError:
            curr = ThermalState.OPTIMAL

    cfg = (config or {}).get("thermal_management", config or {})
    hyst = cfg.get("hysteresis", {})
    rec = cfg.get("recovery", {})
    regions = cfg.get("thermal_regions", {})

    t_elevated_enter = float(regions.get("optimal", {}).get("threshold_c", 33.0))
    t_derating_enter = float(regions.get("derating", {}).get("min_temp_c", 45.0))
    t_critical_enter = float(regions.get("critical", {}).get("min_temp_c", 55.0))

    t_elevated_exit = float(hyst.get("elevated_exit_c", 32.5))
    t_derating_exit = float(hyst.get("derating_exit_c", 44.5))

    t_crit_to_cool = float(rec.get("critical_to_cooling_threshold_c", 52.0))
    t_safe_resume = float(rec.get("safe_resume_temperature_c", 42.0))
    stop_speed_thresh = float(cfg.get("demo_stop", {}).get("stop_speed_threshold_kmh", 0.01))

    is_demo = (mode.lower() == "demo")

    if curr == ThermalState.OPTIMAL:
        if temperature_c >= t_elevated_enter:
            return ThermalState.ELEVATED_THERMAL
        return ThermalState.OPTIMAL

    elif curr == ThermalState.ELEVATED_THERMAL:
        if temperature_c >= t_derating_enter:
            return ThermalState.DERATING_ACTIVE
        elif temperature_c < t_elevated_exit:
            return ThermalState.OPTIMAL
        return ThermalState.ELEVATED_THERMAL

    elif curr == ThermalState.DERATING_ACTIVE:
        if temperature_c >= t_critical_enter:
            return ThermalState.CRITICAL
        elif temperature_c < t_derating_exit:
            return ThermalState.ELEVATED_THERMAL
        return ThermalState.DERATING_ACTIVE

    elif curr == ThermalState.CRITICAL:
        if is_demo:
            if vehicle_speed_kmh <= stop_speed_thresh:
                return ThermalState.COOLING if temperature_c <= t_crit_to_cool else ThermalState.STOPPED
            return ThermalState.STOP_REQUESTED
        else:
            # In research mode, trace-following continues; transitions back if temperature cools
            if temperature_c < t_derating_exit:
                return ThermalState.ELEVATED_THERMAL
            elif temperature_c < t_derating_enter:
                return ThermalState.DERATING_ACTIVE
            return ThermalState.CRITICAL

    elif curr == ThermalState.STOP_REQUESTED:
        if vehicle_speed_kmh <= stop_speed_thresh:
            return ThermalState.STOPPED
        return ThermalState.DECELERATING

    elif curr == ThermalState.DECELERATING:
        if vehicle_speed_kmh <= stop_speed_thresh:
            return ThermalState.STOPPED
        return ThermalState.DECELERATING

    elif curr == ThermalState.STOPPED:
        if temperature_c <= t_crit_to_cool:
            return ThermalState.COOLING
        return ThermalState.STOPPED

    elif curr == ThermalState.COOLING:
        if temperature_c <= t_safe_resume and vehicle_speed_kmh <= stop_speed_thresh:
            return ThermalState.SAFE_TO_RESUME
        return ThermalState.COOLING

    elif curr == ThermalState.SAFE_TO_RESUME:
        # Requires explicit manual resume to exit
        return ThermalState.SAFE_TO_RESUME

    return ThermalState.OPTIMAL


def calculate_recommended_speed(
    state: ThermalState | str,
    reference_speed_kmh: float,
    current_ceiling_a: float = 160.0,
    rated_current_a: float = 160.0,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Calculate the driver-guidance recommended speed based on thermal state and current ceiling.

    Unconditional rule (§48):
    if state in {CRITICAL, STOP_REQUESTED, DECELERATING, STOPPED, COOLING}:
        recommended_speed_kmh = 0.0
    """
    if isinstance(state, ThermalState):
        st = state
    else:
        st = ThermalState(str(state).upper())

    if st in {
        ThermalState.CRITICAL,
        ThermalState.STOP_REQUESTED,
        ThermalState.DECELERATING,
        ThermalState.STOPPED,
        ThermalState.COOLING,
    }:
        return 0.0

    cfg = (config or {}).get("thermal_management", config or {})
    rec_cfg = cfg.get("speed_recommendation", {})
    min_ratio = float(rec_cfg.get("minimum_speed_ratio", 0.30))
    max_ratio = float(rec_cfg.get("maximum_speed_ratio", 1.00))

    if st in {ThermalState.ELEVATED_THERMAL, ThermalState.DERATING_ACTIVE}:
        if rated_current_a > 0.0:
            r_I = max(0.0, abs(current_ceiling_a) / rated_current_a)
        else:
            r_I = 1.0
        ratio = max(min_ratio, min(max_ratio, r_I))
        return float(reference_speed_kmh * ratio)

    # OPTIMAL, SAFE_TO_RESUME
    return float(reference_speed_kmh)


def get_driver_guidance(state: ThermalState | str) -> str:
    """Return the authoritative driver guidance text for the given state."""
    if isinstance(state, ThermalState):
        val = state.value
    else:
        val = str(state).upper()
    return DRIVER_GUIDANCE.get(val, "NORMAL OPERATION")
