"""
Demo Safety Stop Controller for RL-BMS-Driving.

Provides vehicle-level controlled deceleration, stopped hold, real ECM cooling progression,
and safe-to-resume gating for Demo Mode demonstrations.
Does NOT modify or replace the authoritative BMS safety layer (safety/safety_layer.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from app.thermal_state_machine import ThermalState, determine_state


@dataclass
class DemoStopState:
    is_active: bool = False
    demo_speed_mps: float = 0.0
    initial_stop_speed_mps: float = 0.0
    deceleration_mps2: float = 2.0
    stop_speed_threshold_kmh: float = 0.01
    manually_stopped: bool = False


class DemoSafetyStopController:
    """Demo-only vehicle-level safety stop controller."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = self.config.get("thermal_management", self.config)
        demo_cfg = cfg.get("demo_stop", {})
        
        self.deceleration_mps2 = float(demo_cfg.get("max_deceleration_mps2", 2.0))
        self.stop_threshold_kmh = float(demo_cfg.get("stop_speed_threshold_kmh", 0.01))
        self.state = DemoStopState(
            deceleration_mps2=self.deceleration_mps2,
            stop_speed_threshold_kmh=self.stop_threshold_kmh,
        )

    def trigger_stop(self, current_speed_mps: float, manual: bool = False) -> None:
        """Trigger demo safety stop."""
        self.state.is_active = True
        self.state.manually_stopped = manual
        self.state.demo_speed_mps = max(0.0, float(current_speed_mps))
        self.state.initial_stop_speed_mps = self.state.demo_speed_mps

    def step(
        self,
        dt_s: float,
        reference_speed_mps: float,
        temperature_c: float,
        current_thermal_state: ThermalState,
    ) -> Tuple[float, ThermalState, bool]:
        """Advance the demo stop state by dt.

        Returns:
            (applied_demo_speed_mps, next_thermal_state, override_active)
        """
        if not self.state.is_active:
            # Check if thermal state entered CRITICAL or STOP_REQUESTED
            if current_thermal_state in {
                ThermalState.CRITICAL,
                ThermalState.STOP_REQUESTED,
            }:
                self.trigger_stop(reference_speed_mps, manual=False)
            else:
                next_st = determine_state(
                    current_thermal_state,
                    temperature_c,
                    vehicle_speed_kmh=reference_speed_mps * 3.6,
                    config=self.config,
                    mode="demo",
                )
                return reference_speed_mps, next_st, False

        # If stop is active, apply controlled deceleration
        current_speed_kmh = self.state.demo_speed_mps * 3.6
        if current_speed_kmh > self.state.stop_speed_threshold_kmh:
            # Decelerate
            new_speed = max(0.0, self.state.demo_speed_mps - self.deceleration_mps2 * dt_s)
            self.state.demo_speed_mps = new_speed
            current_speed_kmh = new_speed * 3.6

        # Determine next state
        next_st = determine_state(
            current_thermal_state,
            temperature_c,
            vehicle_speed_kmh=current_speed_kmh,
            config=self.config,
            mode="demo",
        )

        return self.state.demo_speed_mps, next_st, True

    def can_resume(self, temperature_c: float, current_thermal_state: ThermalState) -> bool:
        """Check if conditions are safe for manual resume."""
        cfg = self.config.get("thermal_management", self.config)
        t_resume = float(cfg.get("recovery", {}).get("safe_resume_temperature_c", 42.0))
        return (
            current_thermal_state == ThermalState.SAFE_TO_RESUME
            and temperature_c <= t_resume
            and (self.state.demo_speed_mps * 3.6) <= self.state.stop_speed_threshold_kmh
        )

    def resume(self, temperature_c: float, current_thermal_state: ThermalState) -> Tuple[bool, ThermalState]:
        """Perform manual safe resume. Returns (success, next_state)."""
        if not self.can_resume(temperature_c, current_thermal_state):
            return False, current_thermal_state

        self.state.is_active = False
        self.state.manually_stopped = False
        self.state.demo_speed_mps = 0.0

        # State upon resume determined by current safe temperature
        next_st = determine_state(
            ThermalState.OPTIMAL,
            temperature_c,
            vehicle_speed_kmh=0.0,
            config=self.config,
            mode="demo",
        )
        return True, next_st

    def reset(self) -> None:
        """Reset the controller."""
        self.state = DemoStopState(
            deceleration_mps2=self.deceleration_mps2,
            stop_speed_threshold_kmh=self.stop_threshold_kmh,
        )
