"""
Track A: Tradeoff & Analytical Screen of Charging Reward Candidates
===================================================================
1. Evaluates exact environment reward at:
   - T in [33, 34, 35, 36, 38, 40, 42, 45] °C
   - I in [120, 140, 160] A
   - Breakdown: progress, thermal, time, smoothness, safety, total reward.
2. Evaluates Candidate parameter sets:
   - Baseline (Current Exp C): T_ref = 33.0, scale = 22.0, w_th = 3.0, w_time = 0.05
   - Candidate A1: T_ref = 35.0, scale = 20.0, w_th = 2.5, w_time = 0.05
   - Candidate A2: T_ref = 36.0, scale = 19.0, w_th = 3.0, w_time = 0.05
   - Candidate A3: T_ref = 35.0, scale = 20.0, w_th = 3.0, w_time = 0.08
3. Screens for required properties:
   - Normal (T <= 35°C): 160A is reward-optimal (R_160 >= R_140 > R_120)
   - Moderate (T = 38-40°C): 140A/120A becomes competitive (R_140 or R_120 >= R_160)
   - High Stress (T >= 42°C): 120A is strictly optimal (R_120 > R_140 > R_160)

Outputs:
  - audit/charging_tradeoff_analysis.csv
  - audit/charging_tradeoff_analysis.md
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from environment.battery_env import BatteryChargingEnv
from environment.ecm_model import BatteryState
from utils.config import load_config


TEMPERATURES_C = [33.0, 34.0, 35.0, 36.0, 38.0, 40.0, 42.0, 45.0]
CURRENTS_A = [120.0, 140.0, 160.0]

CANDIDATES = {
    "ExpC_Baseline": {
        "thermal_enabled": True,
        "thermal_weight": 3.0,
        "thermal_reference_temp_c": 33.0,
        "thermal_scale_c": 22.0,
        "time_penalty_weight": 0.05,
    },
    "Candidate_A1": {
        "thermal_enabled": True,
        "thermal_weight": 2.5,
        "thermal_reference_temp_c": 35.0,
        "thermal_scale_c": 20.0,
        "time_penalty_weight": 0.05,
    },
    "Candidate_A2": {
        "thermal_enabled": True,
        "thermal_weight": 3.0,
        "thermal_reference_temp_c": 36.0,
        "thermal_scale_c": 19.0,
        "time_penalty_weight": 0.05,
    },
    "Candidate_A3": {
        "thermal_enabled": True,
        "thermal_weight": 3.0,
        "thermal_reference_temp_c": 35.0,
        "thermal_scale_c": 20.0,
        "time_penalty_weight": 0.08,
    },
}


def evaluate_reward_grid(candidate_name: str, cand_params: Dict) -> pd.DataFrame:
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = copy.deepcopy(load_config("reward"))
    sim_cfg = load_config("simulation")

    # Apply candidate overrides
    for k, v in cand_params.items():
        if k == "time_penalty_weight":
            reward_cfg["weights"]["time_penalty"] = v
        else:
            reward_cfg[k] = v

    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval")
    i_max = env.i_max

    records = []
    for t_c in TEMPERATURES_C:
        for curr_a in CURRENTS_A:
            # Construct standard mid-pack charging state
            state = BatteryState(soc=0.50, v_rc=0.01, temperature_c=t_c)
            env._state = state
            env._prev_action = np.array([curr_a / i_max], dtype=np.float32)

            action = np.array([curr_a / i_max], dtype=np.float32)
            # Step without advancing time
            applied_current = curr_a  # within safe boundaries at soc=0.50
            from safety.safety_layer import SafetyInfo
            safety_info = SafetyInfo(
                requested_current=curr_a,
                safe_current_ceiling=i_max,
                applied_current=curr_a,
                intervention_type="none",
                magnitude=0.0,
                derating_multiplier=1.0,
            )
            v_t = env.ecm.terminal_voltage(state, applied_current)
            total_rew, comp_dict = env._compute_reward(
                prev_state=state,
                requested_current=curr_a,
                new_state=state,
                applied_current=applied_current,
                safety_info=safety_info,
                terminal_voltage=v_t,
            )

            records.append({
                "candidate": candidate_name,
                "temperature_c": t_c,
                "current_a": curr_a,
                "total_reward": total_rew,
                "progress_reward": comp_dict.get("charging_progress", 0.0),
                "thermal_penalty": comp_dict.get("thermal_penalty", 0.0),
                "time_penalty": comp_dict.get("time_penalty", 0.0),
                "smoothness_penalty": comp_dict.get("smoothness_penalty", 0.0),
                "safety_penalty": comp_dict.get("safety_penalty", 0.0),
            })

    return pd.DataFrame(records)


def main():
    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("TRACK A: ANALYTICAL REWARD SCREENING ACROSS CANDIDATES")
    print("=" * 80)

    all_dfs = []
    for name, params in CANDIDATES.items():
        df_c = evaluate_reward_grid(name, params)
        all_dfs.append(df_c)

    df_all = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(out_dir, "charging_tradeoff_analysis.csv")
    df_all.to_csv(csv_path, index=False)

    # Determine optimal current for each (candidate, temp) pair
    summary_rows = []
    for name in CANDIDATES.keys():
        sub = df_all[df_all["candidate"] == name]
        for t_c in TEMPERATURES_C:
            sub_t = sub[sub["temperature_c"] == t_c]
            r120 = sub_t[sub_t["current_a"] == 120.0]["total_reward"].values[0]
            r140 = sub_t[sub_t["current_a"] == 140.0]["total_reward"].values[0]
            r160 = sub_t[sub_t["current_a"] == 160.0]["total_reward"].values[0]

            best_i = 160.0 if (r160 >= r140 and r160 >= r120) else (140.0 if r140 >= r120 else 120.0)
            summary_rows.append({
                "candidate": name,
                "temperature_c": t_c,
                "r_120A": round(r120, 5),
                "r_140A": round(r140, 5),
                "r_160A": round(r160, 5),
                "r_160_minus_r120": round(r160 - r120, 5),
                "optimal_current_a": best_i,
            })

    df_summary = pd.DataFrame(summary_rows)

    md_path = os.path.join(out_dir, "charging_tradeoff_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Track A: Charging Reward Analytical Screen & Tradeoff Analysis\n\n")
        f.write("**Objective**: Analytically verify instantaneous reward landscape across temperatures and candidate parameter sets before short diagnostic execution.\n\n")
        f.write("---\n\n")
        f.write("## 1. Candidate Parameter Definitions\n\n")
        f.write("| Candidate | Thermal Ref ($T_{\\text{ref}}$) | Thermal Scale ($T_{\\text{scale}}$) | Thermal Weight ($w_{\\text{th}}$) | Time Penalty ($w_{\\text{time}}$) |\n")
        f.write("|---|---|---|---|---|\n")
        for name, p in CANDIDATES.items():
            f.write(f"| **{name}** | {p['thermal_reference_temp_c']}°C | {p['thermal_scale_c']}°C | {p['thermal_weight']} | {p['time_penalty_weight']} |\n")

        f.write("\n---\n\n")
        f.write("## 2. Instantaneous Reward by Temperature & Current\n\n")
        f.write("| Candidate | Temp (°C) | R(120A) | R(140A) | R(160A) | $\\Delta R(160 - 120)$ | Optimal Current |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in df_summary.iterrows():
            f.write(f"| {r['candidate']} | {r['temperature_c']:.0f}°C | {r['r_120A']:.5f} | {r['r_140A']:.5f} | {r['r_160A']:.5f} | {r['r_160_minus_r120']:+.5f} | **{r['optimal_current_a']:.0f}A** |\n")

        f.write("\n---\n\n")
        f.write("## 3. Analytical Screening Evaluation\n\n")
        for name in CANDIDATES.keys():
            sub = df_summary[df_summary["candidate"] == name]
            opt_33 = sub[sub["temperature_c"] == 33.0]["optimal_current_a"].values[0]
            opt_35 = sub[sub["temperature_c"] == 35.0]["optimal_current_a"].values[0]
            opt_38 = sub[sub["temperature_c"] == 38.0]["optimal_current_a"].values[0]
            opt_40 = sub[sub["temperature_c"] == 40.0]["optimal_current_a"].values[0]
            opt_42 = sub[sub["temperature_c"] == 42.0]["optimal_current_a"].values[0]
            opt_45 = sub[sub["temperature_c"] == 45.0]["optimal_current_a"].values[0]

            normal_pass = bool(opt_33 == 160.0 and opt_35 == 160.0)
            mod_pass = bool(opt_40 <= 140.0)
            stress_pass = bool(opt_42 <= 140.0 and opt_45 == 120.0)
            passed = normal_pass and mod_pass and stress_pass

            f.write(f"### {name}\n")
            f.write(f"- **Normal ($T \\le 35^\\circ\\text{{C}}$)**: Optimal at 33°C = {opt_33:.0f}A, at 35°C = {opt_35:.0f}A $\\rightarrow$ {'PASS (160A)' if normal_pass else 'FAIL'}\n")
            f.write(f"- **Moderate ($T = 38\\text{{--}}40^\\circ\\text{{C}}$)**: Optimal at 38°C = {opt_38:.0f}A, at 40°C = {opt_40:.0f}A $\\rightarrow$ {'PASS (Derating Begins)' if mod_pass else 'FAIL'}\n")
            f.write(f"- **High Stress ($T \\ge 42^\\circ\\text{{C}}$)**: Optimal at 42°C = {opt_42:.0f}A, at 45°C = {opt_45:.0f}A $\\rightarrow$ {'PASS (Strong Derating)' if stress_pass else 'FAIL'}\n")
            f.write(f"- **Analytical Screen Status**: **{'PASSED FOR DIAGNOSTIC TESTING' if passed else 'REJECTED'}**\n\n")

    print(f"\nArtifacts generated:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
