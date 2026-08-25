"""Factory for building a BatteryChargingEnv from the on-disk config directory."""

from __future__ import annotations

from typing import Optional

from environment.battery_env import BatteryChargingEnv
from utils.config import load_config


def make_env(mode: str = "train", config_dir: Optional[str] = None) -> BatteryChargingEnv:
    kwargs = {"config_dir": config_dir} if config_dir else {}
    battery_cfg = load_config("battery", **kwargs)
    safety_cfg = load_config("safety", **kwargs)
    reward_cfg = load_config("reward", **kwargs)
    sim_cfg = load_config("simulation", **kwargs)
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode=mode)
