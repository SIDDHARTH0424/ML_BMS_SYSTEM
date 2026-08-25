"""Tests for utils/metrics.py's driving-EMS additions and training/evaluate_drive_ems.py."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (
    distance_km, driving_energy_wh_breakdown, minimum_soc,
    regen_recovery_fraction, wh_per_km,
)
from baselines.rule_based_ems import RuleBasedEMS
from training.evaluate_drive_ems import run_episode
from training.train_drive_ems import make_drive_ems_env, FIXTURE_DRIVE_CYCLE


def test_driving_energy_wh_breakdown_signs():
    # -100W (discharge) for 3600s = 100 Wh discharged, +50W (charge/regen) for 3600s = 50 Wh regen
    powers = [-100.0] * 3600 + [50.0] * 3600
    result = driving_energy_wh_breakdown(powers, dt_seconds=1.0)
    assert result["discharge_energy_wh"] == pytest.approx(100.0, rel=1e-3)
    assert result["regen_energy_wh"] == pytest.approx(50.0, rel=1e-3)
    assert result["net_energy_wh"] == pytest.approx(50.0, rel=1e-3)


def test_wh_per_km_basic():
    assert wh_per_km(net_energy_wh=200.0, distance_km=10.0) == pytest.approx(20.0)


def test_wh_per_km_zero_distance_is_nan():
    assert math.isnan(wh_per_km(net_energy_wh=200.0, distance_km=0.0))


def test_regen_recovery_fraction_bounds():
    assert regen_recovery_fraction(50.0, 100.0) == pytest.approx(0.5)
    assert regen_recovery_fraction(100.0, 100.0) == pytest.approx(1.0)
    assert math.isnan(regen_recovery_fraction(0.0, 0.0))  # nothing available -> NaN, not misleading 0.0


def test_minimum_soc():
    assert minimum_soc([0.5, 0.3, 0.6, 0.2, 0.9]) == pytest.approx(0.2)


def test_distance_km():
    # 10 m/s for 100 steps at dt=1s = 1000m = 1km
    assert distance_km([10.0] * 100, dt_seconds=1.0) == pytest.approx(1.0)


# Evaluator integration test: rule-based baseline on the synthetic fixture
def test_evaluate_rule_based_produces_sane_metrics():
    env = make_drive_ems_env(drive_cycle_path=FIXTURE_DRIVE_CYCLE, mode="eval")
    controller = RuleBasedEMS()
    result = run_episode(controller, env, initial_soc=0.5, ambient_temp_c=25.0, seed=42)

    for key in ["steps", "distance_km", "wh_per_km", "discharge_energy_wh", "regen_energy_wh",
                "net_energy_wh", "min_soc", "max_temperature_c", "avg_temperature_c",
                "safety_interventions", "safety_intervention_rate"]:
        assert key in result

    assert result["steps"] > 0
    assert result["distance_km"] > 0.0
    assert result["discharge_energy_wh"] >= 0.0
    assert result["regen_energy_wh"] >= 0.0
    assert 0.0 <= result["min_soc"] <= 1.0
    assert result["max_temperature_c"] >= result["avg_temperature_c"]
    assert result["safety_interventions"] >= 0
    # regen_recovery_fraction should be a valid fraction or NaN, never > 1 or < 0
    frac = result["regen_recovery_fraction"]
    assert math.isnan(frac) or (0.0 <= frac <= 1.0)
