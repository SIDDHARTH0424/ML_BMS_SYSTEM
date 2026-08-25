"""
Configuration loading utility.

Loads YAML config files into plain dicts (with dot-access convenience),
and validates that required keys are present so failures happen at
load-time, not deep inside training.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


class ConfigDict(dict):
    """Dict subclass allowing attribute-style access: cfg.battery.r0_ohm"""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_config_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config_dict(v) for v in obj]
    return obj


def load_config(name: str, config_dir: str = CONFIG_DIR) -> ConfigDict:
    """Load a single YAML config file by name (e.g. 'battery' or 'battery.yaml')."""
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    path = os.path.join(config_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    return _to_config_dict(raw)


def load_all_configs(config_dir: str = CONFIG_DIR) -> ConfigDict:
    """Load every *.yaml file in config_dir into one namespace keyed by filename stem."""
    configs: Dict[str, Any] = {}
    for fname in sorted(os.listdir(config_dir)):
        if fname.endswith(".yaml"):
            stem = fname[: -len(".yaml")]
            configs[stem] = load_config(stem, config_dir=config_dir)
    return ConfigDict(configs)


def require_keys(cfg: Dict[str, Any], keys: Iterable[str], context: str = "") -> None:
    """Raise a clear error if any required key is missing from a config dict."""
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys {missing} in {context or 'config'}")


def snapshot_configs(config_dir: str, dest_dir: str) -> None:
    """Copy all config YAML files into a run directory for reproducibility."""
    import shutil

    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(config_dir):
        if fname.endswith(".yaml"):
            shutil.copy(os.path.join(config_dir, fname), os.path.join(dest_dir, fname))


def deep_copy(cfg: Any) -> Any:
    return copy.deepcopy(cfg)
