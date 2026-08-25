"""Constant Current - Constant Voltage (CCCV) baseline controller.

Industry-standard charging strategy: charge at fixed current until the
terminal voltage reaches the CV setpoint, then hold that voltage by
tapering current down (approximated here via a simple proportional
controller on the voltage error, since the safety layer's own voltage
taper handles the physical realism / smoothness).
"""

from __future__ import annotations

from baselines.base_controller import BaseController


class CCCVController(BaseController):
    name = "cccv"

    def __init__(self, config: dict):
        self.cc_current_a = float(config["cc_current_a"])
        self.cv_voltage_v = float(config["cv_voltage_v"])
        self.cv_cutoff_current_a = float(config["cv_cutoff_current_a"])
        self.cv_proportional_gain = float(config.get("cv_proportional_gain", 0.1))
        self._in_cv_phase = False

    def reset(self) -> None:
        self._in_cv_phase = False

    def act(self, observation: dict) -> float:
        voltage = observation["terminal_voltage"]

        if not self._in_cv_phase and voltage >= self.cv_voltage_v:
            self._in_cv_phase = True

        if not self._in_cv_phase:
            return self.cc_current_a

        # CV phase: proportional taper to hold voltage near the setpoint.
        # Gain is tunable via configs/evaluation.yaml: cccv.cv_proportional_gain
        # (fraction of cc_current shed per volt of overshoot); clamped to
        # [cutoff, cc_current] so it decays monotonically.
        error = voltage - self.cv_voltage_v
        gain = self.cv_proportional_gain * self.cc_current_a
        requested = self.cc_current_a - gain * max(error, 0.0)
        return max(self.cv_cutoff_current_a, min(requested, self.cc_current_a))
