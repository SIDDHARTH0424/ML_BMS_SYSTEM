"""
Common interface every baseline controller implements, so the evaluation
harness can treat them uniformly alongside the PPO agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseController(ABC):
    name: str = "base"

    def reset(self) -> None:
        """Reset any internal controller state (e.g. CCCV phase). No-op by default."""
        pass

    @abstractmethod
    def act(self, observation: dict) -> float:
        """Return a *requested* charging current in Amps (pre-safety-layer).

        observation keys: soc, terminal_voltage, temperature_c,
        previous_current_a, ambient_temp_c
        """
        raise NotImplementedError
