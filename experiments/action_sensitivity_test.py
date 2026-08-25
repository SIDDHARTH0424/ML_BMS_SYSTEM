"""
Action Sensitivity Test (Fix Item 6)
=====================================
Deterministic test: for the SAME battery state, evaluate several action
values and record whether changing the action actually changes the applied
current.  If all actions collapse to the same applied current over most of
the reachable charging state distribution, the environment provides
insufficient control authority for RL.

Usage:
    python -m experiments.action_sensitivity_test
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np

from environment.ecm_model import BatteryECM, BatteryState
from safety.safety_layer import safety_layer
from utils.config import load_config


# Fixed raw actions to test (in [-1, 1] policy output space)
RAW_ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Representative states across the reachable charging distribution
TEST_STATES = [
    {"label": "low_soc_low_temp",   "soc": 0.15, "temp_c": 20.0, "v_rc": 0.0},
    {"label": "low_soc_mid_temp",   "soc": 0.15, "temp_c": 30.0, "v_rc": 0.5},
    {"label": "mid_soc_low_temp",   "soc": 0.50, "temp_c": 20.0, "v_rc": 1.0},
    {"label": "mid_soc_mid_temp",   "soc": 0.50, "temp_c": 30.0, "v_rc": 1.5},
    {"label": "mid_soc_high_temp",  "soc": 0.50, "temp_c": 47.0, "v_rc": 2.0},
    {"label": "high_soc_low_temp",  "soc": 0.85, "temp_c": 20.0, "v_rc": 1.0},
    {"label": "high_soc_mid_temp",  "soc": 0.85, "temp_c": 30.0, "v_rc": 1.5},
    {"label": "taper_zone",         "soc": 0.92, "temp_c": 25.0, "v_rc": 1.0},
    {"label": "deep_taper",         "soc": 0.96, "temp_c": 25.0, "v_rc": 0.5},
    {"label": "high_temp_derate",   "soc": 0.50, "temp_c": 52.0, "v_rc": 2.0},
    {"label": "both_active",        "soc": 0.93, "temp_c": 48.0, "v_rc": 1.5},
]


def run_action_sensitivity_test() -> List[Dict]:
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")

    ecm = BatteryECM(battery_cfg)
    i_max = float(battery_cfg["i_max_a"])
    ambient = 25.0  # fixed ambient for this test

    rows: List[Dict] = []

    for state_spec in TEST_STATES:
        state = BatteryState(
            soc=state_spec["soc"],
            v_rc=state_spec["v_rc"],
            temperature_c=state_spec["temp_c"],
            soh=1.0,
            ah_throughput=0.0,
        )

        # Compute safe ceiling once (state-dependent, action-independent)
        v_est = ecm.terminal_voltage(state, i_max)

        for raw_action in RAW_ACTIONS:
            # Same remapping as battery_env.py step()
            action_val = (raw_action + 1.0) / 2.0
            requested_current = action_val * i_max

            applied_current, safety_info = safety_layer(
                requested_current, state, safety_cfg, estimated_voltage=v_est
            )

            # Step the battery to get next state
            new_state = ecm.step(state, applied_current, ambient)
            terminal_v = ecm.terminal_voltage(new_state, applied_current)

            # Compute reward (reuse battery_env logic inline)
            delta_soc = new_state.soc - state.soc
            progress = reward_cfg["weights"]["charging_progress"] * delta_soc
            temp_start = reward_cfg["temperature_penalty_start_c"]
            temp_excess = max(0.0, new_state.temperature_c - temp_start)
            temp_penalty = reward_cfg["weights"]["temperature_penalty"] * temp_excess
            safety_pen = reward_cfg["weights"]["safety_penalty"] * safety_info.magnitude
            overrequest_a = max(0.0, requested_current - applied_current)
            overrequest_pen = reward_cfg["weights"]["overrequest_penalty"] * (overrequest_a / i_max)
            time_pen = reward_cfg["weights"]["time_penalty"]
            total_reward = progress - temp_penalty - safety_pen - overrequest_pen - time_pen

            q_gen = ecm.heat_generation_w(state, applied_current)

            rows.append({
                "state_label": state_spec["label"],
                "soc": state_spec["soc"],
                "temperature_c": state_spec["temp_c"],
                "v_rc": state_spec["v_rc"],
                "raw_action": raw_action,
                "requested_current_a": round(requested_current, 4),
                "safe_ceiling_a": round(safety_info.safe_current_ceiling, 4),
                "applied_current_a": round(applied_current, 4),
                "ceiling_active": safety_info.safe_current_ceiling < i_max - 1e-6,
                "intervention": safety_info.intervention_type,
                "next_soc": round(new_state.soc, 6),
                "next_temp_c": round(new_state.temperature_c, 4),
                "q_gen_w": round(q_gen, 4),
                "reward": round(total_reward, 6),
            })

    return rows


def analyze_results(rows: List[Dict]) -> None:
    """Print summary analysis answering the key questions."""
    print("=" * 80)
    print("ACTION SENSITIVITY TEST — RESULTS")
    print("=" * 80)
    print()

    # Per-state analysis
    states = {}
    for r in rows:
        label = r["state_label"]
        if label not in states:
            states[label] = []
        states[label].append(r)

    total_states = len(states)
    collapsed_states = 0
    partial_states = 0

    for label, state_rows in states.items():
        applied_currents = [r["applied_current_a"] for r in state_rows]
        unique_applied = len(set(round(a, 2) for a in applied_currents))
        ceiling = state_rows[0]["safe_ceiling_a"]
        ceiling_active = state_rows[0]["ceiling_active"]

        current_range = max(applied_currents) - min(applied_currents)
        is_collapsed = current_range < 1.0  # less than 1A variation

        if is_collapsed:
            collapsed_states += 1
        elif unique_applied < len(RAW_ACTIONS):
            partial_states += 1

        print(f"State: {label} (SoC={state_rows[0]['soc']}, T={state_rows[0]['temperature_c']}°C)")
        print(f"  Safe ceiling: {ceiling:.1f}A  {'(ACTIVE — below i_max)' if ceiling_active else '(= i_max, not binding)'}")
        print(f"  Applied current range: {min(applied_currents):.1f}A — {max(applied_currents):.1f}A  (span={current_range:.1f}A)")
        print(f"  Unique applied values: {unique_applied}/{len(RAW_ACTIONS)}")
        if is_collapsed:
            print(f"  *** COLLAPSED: all actions produce same applied current ***")
        print(f"  Actions:")
        for r in state_rows:
            print(f"    action={r['raw_action']:+5.1f} -> req={r['requested_current_a']:7.1f}A "
                  f"-> applied={r['applied_current_a']:7.1f}A  "
                  f"next_SoC={r['next_soc']:.5f}  next_T={r['next_temp_c']:.3f}  "
                  f"reward={r['reward']:+.5f}  [{r['intervention']}]")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total test states: {total_states}")
    print(f"States where all actions collapse to same applied current: {collapsed_states}/{total_states}")
    print(f"States with partial collapse (fewer unique applied values than actions): {partial_states}/{total_states}")
    print(f"States with full control authority (all 5 actions produce distinct applied current): "
          f"{total_states - collapsed_states - partial_states}/{total_states}")
    print()

    if collapsed_states > total_states * 0.5:
        print("VERDICT: Over half of tested states show collapsed control authority.")
        print("  -> The environment provides INSUFFICIENT control authority for RL")
        print("     in a substantial fraction of the reachable state space.")
    elif collapsed_states > 0:
        print("VERDICT: Some states show collapsed control authority, but many states")
        print("  retain meaningful action differentiation.")
        print("  -> Control authority exists in the bulk-charging phase but is")
        print("     limited near safety boundaries (expected by design).")
    else:
        print("VERDICT: All tested states show meaningful action sensitivity.")
        print("  -> The environment provides adequate control authority for RL.")


def main():
    rows = run_action_sensitivity_test()

    # Save CSV
    out_dir = os.path.join("audit")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "action_sensitivity_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Raw results written to {out_path}")
    print()

    analyze_results(rows)


if __name__ == "__main__":
    main()
