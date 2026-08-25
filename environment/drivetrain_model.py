"""
Reduced-order drivetrain/motor efficiency model.

Deliberately NOT a detailed electromagnetic motor model (per task §11) --
a simple efficiency-factor conversion between wheel power and battery
power, in both the propulsion and regeneration directions, using
separate configurable efficiencies for each (not assumed equal).

Sign convention (consistent with environment/vehicle_dynamics.py and
audit/vehicle_integration_plan.md):
    P_wheel > 0 (propulsion demand) -> battery power drawn (positive
                 battery_power here means power flowing OUT of the
                 battery, i.e. discharge -- the OPPOSITE sign of this
                 project's existing charging convention, "positive
                 current = charging". This module works in a
                 propulsion-positive convention internally and documents
                 it explicitly; converting to the project's
                 charging-positive current convention is the caller's
                 job at the battery-coupling boundary (Phase 4/5), not
                 this module's.
    P_wheel < 0 (braking opportunity) -> available_regenerative_power_w
                 is reported as a non-negative magnitude (how much power
                 COULD be recovered into the battery), capped by
                 motor_max_power_w, max_regen_power_w, and regen_efficiency.
                 It is NOT automatically applied anywhere -- battery
                 charge-acceptance and safety-layer limits (Phase 4)
                 further constrain what's actually usable, which this
                 module has no knowledge of.

No energy creation: regen_efficiency and propulsion_efficiency are both
enforced to lie in (0, 1], and available regenerative power is always
computed as LESS than or equal to the mechanical power available at the
wheels (never more) -- verified in tests/test_drivetrain.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrivetrainOutput:
    traction_power_w: float          # = wheel power actually delivered/absorbed at the wheels
    motor_power_w: float             # power at the motor shaft (before/after conversion losses)
    battery_power_w: float           # propulsion: positive = drawn from battery; 0 when braking (see available_regenerative_power_w)
    drivetrain_losses_w: float       # non-negative, energy dissipated in this conversion step
    available_regenerative_power_w: float  # non-negative; 0 when propelling


class DrivetrainModel:
    def __init__(self, config: dict):
        self.motor_max_power_w = float(config["motor_max_power_w"])
        self.propulsion_efficiency = float(config["propulsion_efficiency"])
        self.regen_efficiency = float(config["regen_efficiency"])
        self.max_regen_power_w = float(config["max_regen_power_w"])

        for name, val in [("propulsion_efficiency", self.propulsion_efficiency),
                           ("regen_efficiency", self.regen_efficiency)]:
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {val}")
        if self.motor_max_power_w <= 0.0:
            raise ValueError(f"motor_max_power_w must be > 0, got {self.motor_max_power_w}")
        if self.max_regen_power_w < 0.0:
            raise ValueError(f"max_regen_power_w must be >= 0, got {self.max_regen_power_w}")

    def compute(self, p_wheel_w: float) -> DrivetrainOutput:
        if p_wheel_w >= 0.0:
            return self._propulsion(p_wheel_w)
        return self._regeneration(p_wheel_w)

    # ------------------------------------------------------------------ #
    def _propulsion(self, p_wheel_w: float) -> DrivetrainOutput:
        # Motor/drivetrain can only deliver up to its rated max power --
        # traction power actually achievable is capped there. This is a
        # capability limit, not a demand-satisfaction guarantee: if
        # p_wheel_w exceeds motor_max_power_w, the caller (Phase 6,
        # ev_energy_env.py) is responsible for tracking the resulting
        # power deficit (task §20) -- this module just reports what the
        # drivetrain itself can physically deliver.
        traction_power_w = min(p_wheel_w, self.motor_max_power_w)
        motor_power_w = traction_power_w
        # P_battery = P_wheel / eta_total (task §11.1) -- efficiency < 1
        # means MORE power must be drawn from the battery than reaches
        # the wheels, so battery_power_w > motor_power_w.
        battery_power_w = motor_power_w / self.propulsion_efficiency if motor_power_w > 0.0 else 0.0
        losses_w = battery_power_w - motor_power_w
        return DrivetrainOutput(
            traction_power_w=traction_power_w, motor_power_w=motor_power_w,
            battery_power_w=battery_power_w, drivetrain_losses_w=losses_w,
            available_regenerative_power_w=0.0,
        )

    def _regeneration(self, p_wheel_w: float) -> DrivetrainOutput:
        # p_wheel_w is negative (braking); magnitude is the mechanical
        # power available at the wheels to potentially recover.
        mechanical_available_w = -p_wheel_w  # positive magnitude

        # Cap by what the motor can absorb acting as a generator, and by
        # the drivetrain's own configured regen power limit.
        motor_power_w = min(mechanical_available_w, self.motor_max_power_w, self.max_regen_power_w)

        # Conversion losses reduce what actually reaches the battery --
        # this is why available_regenerative_power_w < mechanical_available_w
        # whenever regen_efficiency < 1 (no energy creation, verified by test).
        available_regenerative_power_w = motor_power_w * self.regen_efficiency
        losses_w = motor_power_w - available_regenerative_power_w

        return DrivetrainOutput(
            traction_power_w=p_wheel_w, motor_power_w=motor_power_w,
            battery_power_w=0.0, drivetrain_losses_w=losses_w,
            available_regenerative_power_w=available_regenerative_power_w,
        )
