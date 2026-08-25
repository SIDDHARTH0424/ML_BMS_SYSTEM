"""
Reduced-order longitudinal vehicle dynamics model.

Deliberately simpler than a full vehicle simulator: a point-mass
longitudinal force balance (acceleration + rolling resistance + aero drag
+ road grade), producing a wheel power demand from (speed, acceleration,
grade). No lateral dynamics, no suspension, no tire slip model.

Sign convention (documented here, not inherited from anywhere else --
this is a new subsystem):
    P_wheel > 0  -> propulsion demand (vehicle needs power at the wheels)
    P_wheel < 0  -> braking/deceleration opportunity (wheels could supply
                    power back, subject to the drivetrain/regen model,
                    not decided by this module)
    v == 0       -> P_wheel = 0 (no aerodynamic or motion-dependent terms
                    can act; the vehicle isn't moving)

This module has no dependency on the battery, safety layer, or RL stack --
same "physics before AI" separation as environment/ecm_model.py, so it can
be validated in isolation (see tests/test_vehicle_dynamics.py) before
anything else is built on top of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VehicleForces:
    f_accel: float
    f_roll: float
    f_aero: float
    f_grade: float
    f_tractive: float
    p_wheel: float


class VehicleDynamics:
    """Longitudinal point-mass vehicle model. Pure function of
    (speed, acceleration, grade) -- no internal state, no history."""

    def __init__(self, config: dict):
        self.mass_kg = float(config["mass_kg"])
        self.cd = float(config["drag_coefficient"])
        self.frontal_area_m2 = float(config["frontal_area_m2"])
        self.crr = float(config["rolling_resistance_coefficient"])
        self.wheel_radius_m = float(config["wheel_radius_m"])
        self.rho = float(config["air_density_kg_m3"])
        self.g = float(config["gravity_m_s2"])

    def compute(self, speed_mps: float, acceleration_mps2: float, road_grade_rad: float = 0.0) -> VehicleForces:
        """Compute the longitudinal force balance and wheel power for a
        given instantaneous (speed, acceleration, grade).

        road_grade_rad: signed grade angle in radians (positive = uphill).
        Callers passing degrees must convert first -- this module works in
        radians throughout, since that's what math.sin expects; the
        drive-cycle interface (Phase 2) is responsible for any unit
        conversion at its own boundary.

        speed_mps is assumed >= 0 throughout this project (the drive-cycle
        interface, Phase 2, validates this) -- rolling resistance and
        aerodynamic drag below are written as magnitudes opposing forward
        motion, not signed for reverse travel, consistent with that
        assumption. Negative speed is not a supported input.
        """
        v = float(speed_mps)

        # At exactly zero speed, aerodynamic drag has no meaning (no
        # relative airflow) and there's nothing tractive to report as
        # power (P = F*v = 0 regardless of F when v = 0) -- handled by
        # the multiplication below, but F_aero is explicitly zeroed here
        # too so the force breakdown itself is physically sensible at
        # v=0, not just the final power number.
        f_accel = self.mass_kg * acceleration_mps2
        f_roll = self.mass_kg * self.g * self.crr if v != 0.0 else 0.0
        f_aero = 0.5 * self.rho * self.cd * self.frontal_area_m2 * (v ** 2)
        f_grade = self.mass_kg * self.g * math.sin(road_grade_rad)

        f_tractive = f_accel + f_roll + f_aero + f_grade
        p_wheel = f_tractive * v if v != 0.0 else 0.0

        return VehicleForces(
            f_accel=f_accel, f_roll=f_roll, f_aero=f_aero, f_grade=f_grade,
            f_tractive=f_tractive, p_wheel=p_wheel,
        )
