"""Rule-Based Adaptive Charging baseline controller.

Charging current adjusted using predefined SoC-banded rules
(faster charging at low SoC, tapered at high SoC) — a simple
heuristic representative of common adaptive-charging firmware logic.
"""

from __future__ import annotations

import numpy as np

from baselines.base_controller import BaseController


class AdaptiveController(BaseController):
    name = "adaptive"

    def __init__(self, config: dict):
        self.soc_bands = np.asarray(config["soc_bands"], dtype=float)
        self.current_per_band = np.asarray(config["current_a_per_band"], dtype=float)
        if len(self.current_per_band) != len(self.soc_bands) - 1:
            raise ValueError("current_a_per_band must have one fewer entry than soc_bands")

    def act(self, observation: dict) -> float:
        soc = observation["soc"]
        band_idx = int(np.clip(np.searchsorted(self.soc_bands, soc, side="right") - 1,
                                0, len(self.current_per_band) - 1))
        return float(self.current_per_band[band_idx])
