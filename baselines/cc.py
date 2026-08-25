"""Constant Current (CC) baseline controller."""

from __future__ import annotations

from baselines.base_controller import BaseController


class ConstantCurrentController(BaseController):
    name = "cc"

    def __init__(self, config: dict):
        self.current_a = float(config["current_a"])

    def act(self, observation: dict) -> float:
        return self.current_a


class MaxCurrentController(BaseController):
    """Trivial control-experiment baseline: always requests the physical
    current ceiling, unconditionally, regardless of any state input.

    Exists to test whether PPO's apparent performance advantage over
    CC/CCCV/Adaptive comes from anything learned, or entirely from the
    safety layer it's wrapped in — see docs/results_and_discussion.md
    Section 4. If this trivial controller's evaluation numbers closely
    match the trained PPO policy's, that confirms the safety layer (not
    training) is the source of the observed advantage, since the seed-7
    policy was found to saturate to "always request max current" with zero
    measurable state-dependence (via training/policy_sensitivity_analysis.py).
    """
    name = "max_current"

    def __init__(self, config: dict):
        self.i_max_a = float(config["i_max_a"])

    def act(self, observation: dict) -> float:
        return self.i_max_a