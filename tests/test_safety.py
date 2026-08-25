import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ecm_model import BatteryState
from safety.safety_layer import safety_layer, state_based_current_multiplier
from utils.config import load_config


@pytest.fixture
def safety_config():
    return load_config("safety")


def make_state(soc=0.3, temp=25.0):
    return BatteryState(soc=soc, v_rc=0.0, temperature_c=temp)


# --------------------------------------------------------------------- #
# Current limiting
# --------------------------------------------------------------------- #
def test_current_capped_at_i_max(safety_config):
    state = make_state()
    applied, info = safety_layer(999.0, state, safety_config)
    assert applied == pytest.approx(safety_config["i_max_a"])
    assert info.intervened


def test_negative_current_clamped_to_zero(safety_config):
    state = make_state()
    applied, info = safety_layer(-50.0, state, safety_config)
    assert applied == 0.0


def test_request_within_bounds_passes_through_unchanged(safety_config):
    state = make_state(soc=0.3, temp=25.0)
    requested = 40.0
    applied, info = safety_layer(requested, state, safety_config)
    assert applied == pytest.approx(requested)
    assert not info.intervened


# --------------------------------------------------------------------- #
# Temperature derating
# --------------------------------------------------------------------- #
def test_no_derate_below_temp_threshold(safety_config):
    state = make_state(temp=safety_config["t_derate_start_c"] - 5.0)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(50.0)


def test_partial_derate_between_start_and_cutoff(safety_config):
    mid_temp = (safety_config["t_derate_start_c"] + safety_config["t_hard_cutoff_c"]) / 2
    state = make_state(temp=mid_temp)
    applied, info = safety_layer(safety_config["i_max_a"], state, safety_config)
    assert 0.0 < applied < safety_config["i_max_a"]
    assert info.intervention_type == "temperature"


def test_full_cutoff_at_or_above_hard_temp_limit(safety_config):
    state = make_state(temp=safety_config["t_hard_cutoff_c"] + 5.0)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# SoC tapering
# --------------------------------------------------------------------- #
def test_no_soc_taper_below_threshold(safety_config):
    state = make_state(soc=safety_config["soc_taper_start"] - 0.1)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(50.0)


def test_soc_taper_reduces_current_near_target(safety_config):
    mid_soc = (safety_config["soc_taper_start"] + safety_config["soc_taper_full"]) / 2
    state = make_state(soc=mid_soc)
    applied, info = safety_layer(safety_config["i_max_a"], state, safety_config)
    assert 0.0 < applied < safety_config["i_max_a"]
    assert info.intervention_type == "soc_taper"


def test_soc_taper_zero_current_at_full(safety_config):
    state = make_state(soc=safety_config["soc_taper_full"])
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# Voltage tapering
# --------------------------------------------------------------------- #
def test_voltage_taper_applies_when_estimate_provided(safety_config):
    state = make_state()
    mid_v = (safety_config["v_taper_start"] + safety_config["v_hard_max"]) / 2
    applied, info = safety_layer(safety_config["i_max_a"], state, safety_config, estimated_voltage=mid_v)
    assert 0.0 < applied < safety_config["i_max_a"]
    assert info.intervention_type == "voltage_taper"


def test_voltage_taper_skipped_without_estimate(safety_config):
    state = make_state()
    applied, info = safety_layer(50.0, state, safety_config, estimated_voltage=None)
    assert applied == pytest.approx(50.0)


def test_hard_voltage_max_forces_zero_current(safety_config):
    state = make_state()
    applied, info = safety_layer(
        50.0, state, safety_config, estimated_voltage=safety_config["v_hard_max"] + 5.0
    )
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# Multiple simultaneous constraints -> most restrictive wins
# --------------------------------------------------------------------- #
def test_most_restrictive_rule_dominates(safety_config):
    """High temp AND high SoC both active -> current should reflect the min of both multipliers."""
    hot = safety_config["t_hard_cutoff_c"]  # multiplier -> 0 at hard cutoff
    state = make_state(soc=0.5, temp=hot)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# Logging / info completeness
# --------------------------------------------------------------------- #
def test_safety_info_fields_present(safety_config):
    state = make_state()
    _, info = safety_layer(50.0, state, safety_config)
    d = info.as_dict()
    assert set(["requested_current", "applied_current", "type", "magnitude"]).issubset(d.keys())


def test_magnitude_zero_when_no_intervention(safety_config):
    state = make_state(soc=0.3, temp=25.0)
    _, info = safety_layer(30.0, state, safety_config)
    assert info.magnitude == pytest.approx(0.0)


def test_magnitude_one_when_fully_blocked(safety_config):
    state = make_state(soc=safety_config["soc_taper_full"])
    _, info = safety_layer(30.0, state, safety_config)
    assert info.magnitude == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# state_based_current_multiplier (v2: state-only ceiling, no request needed)
# --------------------------------------------------------------------- #
def test_state_multiplier_full_at_safe_state(safety_config):
    state = make_state(soc=0.3, temp=25.0)
    assert state_based_current_multiplier(state, safety_config) == pytest.approx(1.0)


def test_state_multiplier_reduced_near_soc_taper(safety_config):
    mid_soc = (safety_config["soc_taper_start"] + safety_config["soc_taper_full"]) / 2
    state = make_state(soc=mid_soc, temp=25.0)
    mult = state_based_current_multiplier(state, safety_config)
    assert 0.0 < mult < 1.0


def test_state_multiplier_reduced_near_temp_cutoff(safety_config):
    mid_temp = (safety_config["t_derate_start_c"] + safety_config["t_hard_cutoff_c"]) / 2
    state = make_state(soc=0.3, temp=mid_temp)
    mult = state_based_current_multiplier(state, safety_config)
    assert 0.0 < mult < 1.0


def test_state_multiplier_matches_actual_safety_layer_clamp(safety_config):
    """The state-only multiplier should match what safety_layer() actually
    applies for a request AT OR ABOVE i_max — the safety layer caps the
    request at i_max first, then derates, so the true ceiling (i_max *
    multiplier) is only reached when the request is >= i_max. A request
    BELOW i_max gets derated a second time on top of its own reduction
    (applied = requested * mult, not i_max * mult) — see the overrequest
    penalty tests in test_environment.py for the practical implication."""
    state = make_state(soc=0.95, temp=25.0)  # deep in SoC taper zone
    predicted_mult = state_based_current_multiplier(state, safety_config)
    applied, info = safety_layer(1000.0, state, safety_config)  # request >= i_max
    assert applied == pytest.approx(safety_config["i_max_a"] * predicted_mult, rel=1e-6)


# --------------------------------------------------------------------- #
# Safety Layer v2 semantics — deterministic tests per the fix design doc
# --------------------------------------------------------------------- #
def test_v2_no_derating_passthrough_and_cap(safety_config):
    """Test A: with multiplier=1.0 (safe state), requests below i_max pass
    through unchanged; requests above i_max are capped at i_max."""
    state = make_state(soc=0.3, temp=25.0)  # safe state, multiplier == 1.0
    i_max = safety_config["i_max_a"]

    applied, _ = safety_layer(100.0, state, safety_config)
    assert applied == pytest.approx(100.0)

    applied, _ = safety_layer(i_max, state, safety_config)
    assert applied == pytest.approx(i_max)

    applied, _ = safety_layer(i_max + 50.0, state, safety_config)
    assert applied == pytest.approx(i_max)


def test_v2_temperature_derating_ceiling_semantics(safety_config):
    """Test B: with an active temperature derating multiplier, the ceiling
    is i_max * multiplier — requesting exactly the ceiling should yield the
    ceiling (not double-derated), and requesting less than the ceiling
    should pass through unchanged."""
    from safety.safety_layer import state_based_current_multiplier

    mid_temp = (safety_config["t_derate_start_c"] + safety_config["t_hard_cutoff_c"]) / 2
    state = make_state(soc=0.3, temp=mid_temp)
    mult = state_based_current_multiplier(state, safety_config)
    i_max = safety_config["i_max_a"]
    ceiling = i_max * mult

    # Request above i_max -> capped at ceiling (not i_max)
    applied, _ = safety_layer(i_max, state, safety_config)
    assert applied == pytest.approx(ceiling, rel=1e-6)

    # Request exactly the ceiling -> get exactly the ceiling (the core v2 fix)
    applied, _ = safety_layer(ceiling, state, safety_config)
    assert applied == pytest.approx(ceiling, rel=1e-6)

    # Request below the ceiling -> passes through unchanged, no double-derating
    below_ceiling = ceiling * 0.5
    applied, _ = safety_layer(below_ceiling, state, safety_config)
    assert applied == pytest.approx(below_ceiling, rel=1e-6)


def test_v2_critical_temperature_zero_current(safety_config):
    """Test C: at or above the hard temperature cutoff, applied current is zero."""
    state = make_state(soc=0.3, temp=safety_config["t_hard_cutoff_c"])
    applied, _ = safety_layer(safety_config["i_max_a"], state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)

    state_over = make_state(soc=0.3, temp=safety_config["t_hard_cutoff_c"] + 10.0)
    applied, _ = safety_layer(safety_config["i_max_a"], state_over, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("temp,soc", [
    (25.0, 0.3),    # fully safe
    (50.0, 0.3),    # mid temperature derate
    (25.0, 0.95),   # mid SoC taper
    (52.0, 0.95),   # both active simultaneously
])
def test_v2_monotonicity_applied_never_decreases_with_request(safety_config, temp, soc):
    """Test D: for a fixed state, applied current must never decrease as the
    requested current increases. This is the core property the v1 bug
    violated (requesting less could, perversely, still be double-derated
    relative to requesting the true ceiling) and the v2 fix guarantees."""
    state = make_state(soc=soc, temp=temp)
    requests = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, safety_config["i_max_a"]]

    applied_values = [safety_layer(r, state, safety_config)[0] for r in requests]

    for a, b in zip(applied_values, applied_values[1:]):
        assert b >= a - 1e-9, (
            f"Non-monotonic: applied current decreased from {a} to {b} "
            f"as requested current increased (state: soc={soc}, temp={temp})"
        )


def test_v2_info_exposes_three_way_breakdown(safety_config):
    """The info dict should expose requested / ceiling / applied separately,
    plus the derating multiplier, for debugging and reporting."""
    state = make_state(soc=0.95, temp=25.0)
    _, info = safety_layer(safety_config["i_max_a"], state, safety_config)
    d = info.as_dict()
    assert set(["requested_current", "safe_current_ceiling", "applied_current",
                "type", "magnitude", "derating_multiplier"]).issubset(d.keys())
    assert d["safe_current_ceiling"] == pytest.approx(
        safety_config["i_max_a"] * info.derating_multiplier, rel=1e-6
    )