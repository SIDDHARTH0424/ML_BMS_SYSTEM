"""
Evaluation metric calculations, centralised so every controller
(PPO and baselines) is scored identically.

All functions take raw per-step episode arrays (numpy arrays or lists)
and return a single scalar metric.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def charging_time_s(dt_seconds: float, n_steps: int) -> float:
    return dt_seconds * n_steps


def peak_temperature_c(temps: Sequence[float]) -> float:
    return float(np.max(temps))


def average_temperature_c(temps: Sequence[float]) -> float:
    return float(np.mean(temps))


def final_soc(socs: Sequence[float]) -> float:
    return float(socs[-1])


def safety_interventions(intervention_flags: Sequence[bool]) -> int:
    return int(np.sum(intervention_flags))


def current_smoothness(currents: Sequence[float]) -> float:
    """Mean absolute step-to-step current change. Lower is smoother."""
    currents = np.asarray(currents, dtype=float)
    if len(currents) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(currents))))


def energy_efficiency(input_energy_wh: float, stored_energy_wh: float) -> float:
    """Fraction of the charger's input energy that actually raises the
    cell's stored (OCV-referenced) energy, vs. being lost to internal
    resistance.

    v3 fix: an earlier version computed this as
    delivered/(delivered+dissipated) where delivered = I*V_terminal
    (already inclusive of the I^2*R0 drop, since V_terminal = OCV+I*R0+Vrc)
    and dissipated = I^2*R0 was then added AGAIN in the denominator — double
    counting the R0 loss and producing a physically inconsistent ratio, not
    a real efficiency figure. This version instead compares the charger's
    true input energy (I*V_terminal, computed once) against the energy that
    actually raises the OCV-referenced stored state (I*OCV) — no term is
    counted twice.
    """
    if input_energy_wh <= 0:
        return 0.0
    return float(np.clip(stored_energy_wh / input_energy_wh, 0.0, 1.0))


def average_input_power_w(input_energy_wh: float, charging_time_s: float) -> float:
    """Mean electrical power delivered AT THE BATTERY TERMINALS over the
    episode, in Watts. Renamed from average_charging_power_w — this is
    input power at the terminals (I*V_terminal), not power that ends up
    stored in the cell's electrochemical energy (see energy_efficiency for
    that distinction)."""
    if charging_time_s <= 0:
        return 0.0
    return float(input_energy_wh * 3600.0 / charging_time_s)


def energy_per_percent_soc_wh(input_energy_wh: float, delta_soc: float) -> float:
    """Wh of charger input energy spent per 1% of SoC gained — a direct,
    physically unambiguous efficiency figure that doesn't require
    decomposing losses at all."""
    delta_soc_pct = delta_soc * 100.0
    if delta_soc_pct <= 0:
        return float("nan")
    return float(input_energy_wh / delta_soc_pct)


def voltage_stability(voltages: Sequence[float]) -> float:
    """Standard deviation of terminal voltage over the episode.

    NOTE: this is a coarse proxy, not a true stability metric — voltage
    naturally rises over the course of a charge (SoC increases -> OCV
    increases), so a high std here can simply reflect a wide SoC range
    covered, not erratic/unstable control. A tighter metric (max dV/dt,
    CV-phase tracking error, or std after de-trending the expected SoC-driven
    rise) would better isolate genuine instability; not yet implemented.
    """
    return float(np.std(voltages))


def target_reached(final_soc_value: float, target_soc: float) -> bool:
    return final_soc_value >= target_soc


def target_shortfall(final_soc_value: float, target_soc: float) -> float:
    """How far short of the target the episode finished. 0 if reached or exceeded."""
    return max(0.0, target_soc - final_soc_value)


def summarize_episode(episode_log: Dict[str, List[float]], dt_seconds: float,
                       target_soc: float = 0.95, initial_soc: float = None) -> Dict[str, float]:
    """Compute the full evaluation metric set for one episode's logged arrays.

    Expects episode_log to contain keys: 'temperature_c', 'soc', 'current_a',
    'voltage_v', 'safety_intervention', 'input_energy_wh', 'stored_energy_wh'.

    v3: added target_reached / time_to_target / target_shortfall — a policy
    that truncates without reaching target_soc (e.g. charges partway then
    stalls) must not be scored identically to one that genuinely completes.
    Also fixed energy accounting (see energy_efficiency docstring) — this
    function now expects 'input_energy_wh' / 'stored_energy_wh' rather than
    the old 'delivered_energy_wh' / 'dissipated_energy_wh' keys.

    v3.1 fix: `initial_soc` is now an explicit parameter. episode_log["soc"]
    only contains POST-step values (both episode runners append to the log
    after calling ecm.step/env.step), so episode_log["soc"][0] is the SoC
    after the first step, not the true reset value — a small but real bias
    in energy_per_percent_soc_wh. Callers should pass the actual reset SoC;
    if omitted, falls back to the old (slightly biased) inference for
    backward compatibility.
    """
    n_steps = len(episode_log.get("soc", []))
    dt_total = charging_time_s(dt_seconds, n_steps)
    input_wh = sum(episode_log.get("input_energy_wh", [0.0]))
    stored_wh = sum(episode_log.get("stored_energy_wh", [0.0]))
    final_soc_value = final_soc(episode_log["soc"])
    if initial_soc is None:
        initial_soc = episode_log["soc"][0] if episode_log.get("soc") else final_soc_value
    reached = target_reached(final_soc_value, target_soc)
    return {
        "charging_time_s": dt_total,
        "peak_temperature_c": peak_temperature_c(episode_log["temperature_c"]),
        "average_temperature_c": average_temperature_c(episode_log["temperature_c"]),
        "final_soc": final_soc_value,
        "safety_interventions": safety_interventions(episode_log["safety_intervention"]),
        "current_smoothness": current_smoothness(episode_log["current_a"]),
        "energy_efficiency": energy_efficiency(input_wh, stored_wh),
        "average_input_power_w": average_input_power_w(input_wh, dt_total),
        "energy_per_percent_soc_wh": energy_per_percent_soc_wh(
            input_wh, final_soc_value - initial_soc
        ),
        "voltage_stability": voltage_stability(episode_log["voltage_v"]),
        "target_reached": reached,
        "time_to_target_s": dt_total if reached else float("nan"),
        "target_shortfall": target_shortfall(final_soc_value, target_soc),
    }


def aggregate_runs(metric_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate mean/std across multiple episode metric dicts, per metric key.

    Fix: the previous implementation used np.mean/np.std directly, so a
    single NaN in one run's value for a key (e.g. time_to_target_s for a
    run that never reached target_soc — see summarize_episode) propagated
    NaN into mean/std for ALL runs' values of that key, silently hiding
    every other run's real result.

    This version uses nanmean/nanstd (ignoring NaN entries) but does NOT
    blindly trust them: it also reports how many runs actually contributed
    a valid (non-NaN) value vs. how many were NaN (e.g. target not reached),
    so failures are visible rather than averaged away. If every value for a
    key is NaN, the aggregate mean/std are explicitly reported as NaN too
    (nanmean/nanstd would otherwise emit a RuntimeWarning and return NaN
    anyway; we short-circuit to avoid the warning and make the "no valid
    data" case explicit).
    """
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    result: Dict[str, Dict[str, float]] = {}
    for k in keys:
        values = np.array([m[k] for m in metric_dicts], dtype=float)
        valid_mask = ~np.isnan(values)
        n_valid = int(np.sum(valid_mask))
        n_failed = int(len(values) - n_valid)
        if n_valid == 0:
            mean_val = float("nan")
            std_val = float("nan")
        else:
            mean_val = float(np.nanmean(values))
            std_val = float(np.nanstd(values))
        result[k] = {
            "mean": mean_val,
            "std": std_val,
            "valid_runs": n_valid,
            "failed_runs": n_failed,
        }
    return result

# ---------------------------------------------------------------------- #
# Driving-EMS-specific metrics (Phase 1 extension). Charging-time-style
# metrics above (charging_time_s, energy_efficiency's Wh-based framing,
# etc.) don't map cleanly onto a driving episode -- these are the
# first-class metrics for that problem instead, per the requested
# addition to the rule-based evaluation.
# ---------------------------------------------------------------------- #

def wh_per_km(net_energy_wh: float, distance_km: float) -> float:
    """Net battery energy consumed per km driven. Lower is more efficient.
    net_energy_wh should already net out any regen recovered (i.e. total
    discharge energy minus total recovered regen energy) -- see
    driving_energy_wh_breakdown() below for how the two are kept separate
    before being combined here."""
    if distance_km <= 0:
        return float("nan")
    return net_energy_wh / distance_km


def driving_energy_wh_breakdown(applied_powers_w: Sequence[float], dt_seconds: float) -> Dict[str, float]:
    """Splits a signed applied-battery-power trace (positive=charging/regen,
    negative=discharge, matching this project's convention) into:
        discharge_energy_wh      -- total energy drawn FROM the battery (propulsion)
        regen_energy_wh          -- total energy recovered INTO the battery (regen)
        net_energy_wh            -- discharge_energy_wh - regen_energy_wh
    All non-negative except net_energy_wh, which can be negative if more
    energy was recovered than consumed over the window (e.g. a long
    descent)."""
    powers = np.asarray(applied_powers_w, dtype=float)
    discharge_wh = float(np.sum(np.abs(np.minimum(powers, 0.0))) * dt_seconds / 3600.0)
    regen_wh = float(np.sum(np.maximum(powers, 0.0)) * dt_seconds / 3600.0)
    return {
        "discharge_energy_wh": discharge_wh,
        "regen_energy_wh": regen_wh,
        "net_energy_wh": discharge_wh - regen_wh,
    }


def regen_recovery_fraction(regen_energy_wh: float, available_regen_energy_wh: float) -> float:
    """Fraction of the mechanically-available regenerative energy that was
    actually captured (vs. lost to friction braking). NaN if nothing was
    available to recover (avoids a misleading 0/0 -> 0.0)."""
    if available_regen_energy_wh <= 1e-9:
        return float("nan")
    return float(np.clip(regen_energy_wh / available_regen_energy_wh, 0.0, 1.0))


def minimum_soc(socs: Sequence[float]) -> float:
    return float(np.min(socs))


def distance_km(speeds_mps: Sequence[float], dt_seconds: float) -> float:
    return float(np.sum(speeds_mps) * dt_seconds / 1000.0)
