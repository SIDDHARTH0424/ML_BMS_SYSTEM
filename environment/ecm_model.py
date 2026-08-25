"""
1RC Thevenin Equivalent Circuit Model (ECM) for a lithium-ion (NMC) battery.

State: SoC, terminal voltage, RC polarisation voltage, temperature, SoH.
Integration: Euler (default) or RK4, selected via config.

This module has NO dependency on Gymnasium, RL, or the safety layer —
it is a pure physics simulator so it can be validated in isolation
before anything else is built on top of it (per "Physics Before AI").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class BatteryState:
    soc: float                # 0..1
    v_rc: float                # RC branch (polarisation) voltage, volts
    temperature_c: float       # pack temperature, deg C
    soh: float = 1.0           # state of health, 1.0 = new
    ah_throughput: float = 0.0  # cumulative |current|*dt integrated, for SoH tracking

    def copy(self) -> "BatteryState":
        return BatteryState(
            soc=self.soc,
            v_rc=self.v_rc,
            temperature_c=self.temperature_c,
            soh=self.soh,
            ah_throughput=self.ah_throughput,
        )


class BatteryECM:
    """1RC Thevenin ECM with NMC OCV(SoC) lookup and lumped thermal model."""

    def __init__(self, config: Dict):
        self.capacity_ah = float(config["nominal_capacity_ah"])
        self.r0 = float(config["r0_ohm"])
        self.r1 = float(config["r1_ohm"])
        self.c1 = float(config["c1_farad"])

        ocv_pts = config["ocv_soc_points"]
        self._ocv_soc = np.asarray(ocv_pts["soc"], dtype=float)
        self._ocv_v = np.asarray(ocv_pts["ocv_v"], dtype=float)

        self.mass_kg = float(config["mass_kg"])
        self.cp = float(config["specific_heat_j_per_kgk"])
        self.h = float(config["convection_h"])
        self.area_m2 = float(config["surface_area_m2"])

        self.v_max = float(config["v_max"])
        self.v_min = float(config["v_min"])
        self.t_max_c = float(config["t_max_c"])
        self.i_max_a = float(config["i_max_a"])

        self.soh_degradation_per_ah = float(config["soh_degradation_per_ah_throughput"])

        self.dt = float(config.get("dt_seconds", 1.0))
        self.integration_method = str(config.get("integration_method", "euler")).lower()
        if self.integration_method not in ("euler", "rk4"):
            raise ValueError(f"Unknown integration_method: {self.integration_method}")

    # ------------------------------------------------------------------ #
    # OCV lookup
    # ------------------------------------------------------------------ #
    def ocv(self, soc: float) -> float:
        soc_clamped = float(np.clip(soc, 0.0, 1.0))
        return float(np.interp(soc_clamped, self._ocv_soc, self._ocv_v))

    # ------------------------------------------------------------------ #
    # Derivatives (used by both Euler and RK4)
    # ------------------------------------------------------------------ #
    def _derivatives(self, state: BatteryState, current_a: float, ambient_temp_c: float):
        """Return d(soc)/dt, d(v_rc)/dt, d(temperature)/dt for given state and current.

        Convention: positive current = charging.
        """
        d_soc_dt = current_a / (self.capacity_ah * 3600.0)

        d_vrc_dt = (current_a / self.c1) - (state.v_rc / (self.r1 * self.c1))

        # Standard 1RC resistive-loss formulation: I^2*R0 (ohmic) +
        # Vrc^2/R1 (polarization resistance dissipation). NOT current_a*v_rc:
        # that term is the TOTAL power entering the R1||C1 branch, which
        # includes energy being stored in C1 as well as heat dissipated in
        # R1 — using it as heat over-counts during charging transients and,
        # more visibly, gives exactly zero heat during relaxation (I=0 but
        # Vrc>0 still decaying through R1, which is physically dissipating
        # stored energy the whole time). At steady state (Vrc=I*R1) the two
        # formulations coincide (I*Vrc = I^2*R1 = Vrc^2/R1), so this only
        # changes transient behavior, not steady-state — but this system's
        # RC time constant (~30s) is short relative to episode length, and
        # smoothness-penalized current changes happen often, so transients
        # are common enough to matter here.
        q_gen = self.heat_generation_w(state, current_a)
        q_loss = self.h * self.area_m2 * (state.temperature_c - ambient_temp_c)
        d_temp_dt = (q_gen - q_loss) / (self.mass_kg * self.cp)

        return d_soc_dt, d_vrc_dt, d_temp_dt

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, state: BatteryState, current_a: float, ambient_temp_c: float) -> BatteryState:
        """Advance the battery state by one timestep under the given applied current.

        NOTE: current_a is assumed to already be safety-clamped by the caller
        (the safety layer) — this model does not enforce policy limits itself,
        only reports the resulting physical state.
        """
        dt = self.dt

        if self.integration_method == "euler":
            d_soc, d_vrc, d_temp = self._derivatives(state, current_a, ambient_temp_c)
            new_soc = state.soc + d_soc * dt
            new_vrc = state.v_rc + d_vrc * dt
            new_temp = state.temperature_c + d_temp * dt
        else:  # rk4
            def deriv_at(s: BatteryState):
                return self._derivatives(s, current_a, ambient_temp_c)

            k1 = deriv_at(state)
            s2 = BatteryState(
                soc=state.soc + 0.5 * dt * k1[0],
                v_rc=state.v_rc + 0.5 * dt * k1[1],
                temperature_c=state.temperature_c + 0.5 * dt * k1[2],
                soh=state.soh,
                ah_throughput=state.ah_throughput,
            )
            k2 = deriv_at(s2)
            s3 = BatteryState(
                soc=state.soc + 0.5 * dt * k2[0],
                v_rc=state.v_rc + 0.5 * dt * k2[1],
                temperature_c=state.temperature_c + 0.5 * dt * k2[2],
                soh=state.soh,
                ah_throughput=state.ah_throughput,
            )
            k3 = deriv_at(s3)
            s4 = BatteryState(
                soc=state.soc + dt * k3[0],
                v_rc=state.v_rc + dt * k3[1],
                temperature_c=state.temperature_c + dt * k3[2],
                soh=state.soh,
                ah_throughput=state.ah_throughput,
            )
            k4 = deriv_at(s4)

            new_soc = state.soc + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            new_vrc = state.v_rc + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            new_temp = state.temperature_c + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        new_soc = float(np.clip(new_soc, 0.0, 1.0))

        # SoH bookkeeping (monitoring only — not fed into reward)
        new_ah_throughput = state.ah_throughput + abs(current_a) * dt / 3600.0
        new_soh = max(
            0.0,
            state.soh - self.soh_degradation_per_ah * abs(current_a) * dt / 3600.0,
        )

        return BatteryState(
            soc=new_soc,
            v_rc=new_vrc,
            temperature_c=new_temp,
            soh=new_soh,
            ah_throughput=new_ah_throughput,
        )

    def terminal_voltage(self, state: BatteryState, current_a: float) -> float:
        """Terminal voltage under the positive-current-charging convention.

        V_t = V_OC(SoC) + I*R0 + V_RC
        (matches the RL-BMS doc's V_t = V_OC - I*R0 - V_RC, which uses the
        opposite discharge-positive sign convention; both express the same
        physics — see tests/test_ecm.py for the sign check.)
        """
        return self.ocv(state.soc) + current_a * self.r0 + state.v_rc

    def heat_generation_w(self, state: BatteryState, current_a: float) -> float:
        """Instantaneous ohmic + polarization heat generation (W), same formula
        already used internally by _derivatives — exposed here so callers (the
        reward function, diagnostics) don't need to duplicate it.

        q_gen = I^2 * R0 + V_rc^2 / R1
        """
        return (current_a ** 2) * self.r0 + (state.v_rc ** 2) / self.r1

    def reset_state(self, initial_soc: float, ambient_temp_c: float) -> BatteryState:
        return BatteryState(soc=initial_soc, v_rc=0.0, temperature_c=ambient_temp_c, soh=1.0, ah_throughput=0.0)
