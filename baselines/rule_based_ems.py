"""
Rule-based driving-EMS baseline (task §21): a simple direct
power-following controller, used to sanity-check that
environment/ev_energy_env.py actually works before any PPO training is
attempted.

Policy (deliberately simple, not MPC, per §21's explicit instruction):
    - If propulsion is needed (wheel_power_norm > 0): request full
      discharge (action = -1.0) -- try to fully meet vehicle power
      demand, subject to whatever the drivetrain/battery/safety layer
      can actually supply (this controller doesn't second-guess those
      limits, exactly like a simple firmware controller wouldn't).
    - Else if regen is available (available_regen_norm > 0): request
      full charge (action = +1.0) -- use all available regenerative
      power, minimizing friction-braking loss.
    - Else: action = 0.0 (no meaningful power flow either direction).

Reads directly from the environment's own OBSERVATION_FIELDS ordering
(environment/ev_energy_env.py) rather than hardcoding indices separately,
so the two can't silently drift out of sync.
"""
from __future__ import annotations

import numpy as np

from environment.ev_energy_env import OBSERVATION_FIELDS

_WHEEL_POWER_IDX = OBSERVATION_FIELDS.index("wheel_power_norm")
_AVAILABLE_REGEN_IDX = OBSERVATION_FIELDS.index("available_regen_norm")


class RuleBasedEMS:
    name = "rule_based_direct_power_following"

    def reset(self) -> None:
        pass  # stateless controller, nothing to reset

    def act(self, observation: np.ndarray) -> float:
        wheel_power_norm = float(observation[_WHEEL_POWER_IDX])
        available_regen_norm = float(observation[_AVAILABLE_REGEN_IDX])

        if wheel_power_norm > 0.0:
            return -1.0  # full discharge -- try to fully meet propulsion demand
        elif available_regen_norm > 0.0:
            return 1.0  # full charge -- use all available regen
        return 0.0
