"""
Safety-Layer Dominance Test (Fix Items 5, 7)
=============================================
Quantifies how much of the actual charging trajectory is controlled by PPO
versus the safety supervisor.

Evaluates on a large organically reachable state sample (N >= 5,000 steps)
collected across multiple charging episodes.

Computes:
  - P(safety ceiling active)              = P(safe_ceiling < i_max)
  - P(requested > safe_ceiling)
  - P(requested <= safe_ceiling | ceiling active)
  - P(applied == safe_ceiling)
  - mean(requested_current)
  - mean(applied_current)

Usage:
    python -m experiments.safety_dominance_test
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines.cc import MaxCurrentController
from environment.battery_env import BatteryChargingEnv
from environment.ecm_model import BatteryECM
from safety.safety_layer import safety_layer
from utils.config import load_config


def collect_trajectory_stats(controller_fn, ppo_env: BatteryChargingEnv, n_episodes: int = 15) -> pd.DataFrame:
    """Collect step-by-step diagnostic records across diverse initial conditions."""
    records: List[Dict] = []
    
    # Use grid of initial SoC and ambient temps to sample state space organically
    sim_cfg = load_config("simulation")
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    
    for soc0 in soc_grid:
        for temp0 in temp_grid:
            obs, _ = ppo_env.reset(options={"initial_soc": soc0, "ambient_temp_c": temp0})
            terminated = truncated = False
            
            while not (terminated or truncated):
                action = controller_fn(obs, ppo_env)
                obs, reward, terminated, truncated, info = ppo_env.step(action)
                
                req = info["safety_intervention"]["requested_current"]
                ceil = info["safety_intervention"]["safe_current_ceiling"]
                appl = info["applied_current_a"]
                i_max = ppo_env.i_max
                
                ceiling_active = (ceil < i_max - 1e-6)
                req_gt_ceil = (req > ceil + 1e-6)
                req_le_ceil = not req_gt_ceil
                appl_eq_ceil = abs(appl - ceil) < 1e-6
                
                records.append({
                    "soc": ppo_env._state.soc,
                    "temp_c": ppo_env._state.temperature_c,
                    "requested_current_a": req,
                    "safe_ceiling_a": ceil,
                    "applied_current_a": appl,
                    "ceiling_active": ceiling_active,
                    "req_gt_ceil": req_gt_ceil,
                    "req_le_ceil": req_le_ceil,
                    "appl_eq_ceil": appl_eq_ceil,
                    "intervention_type": info["safety_intervention"]["type"],
                })
                
    return pd.DataFrame(records)


def analyze_dominance(df: pd.DataFrame, controller_name: str) -> Dict[str, float]:
    n = len(df)
    n_ceiling_active = df["ceiling_active"].sum()
    
    p_ceil_active = df["ceiling_active"].mean()
    p_req_gt_ceil = df["req_gt_ceil"].mean()
    p_appl_eq_ceil = df["appl_eq_ceil"].mean()
    
    if n_ceiling_active > 0:
        subset_active = df[df["ceiling_active"]]
        p_req_le_ceil_given_active = subset_active["req_le_ceil"].mean()
    else:
        p_req_le_ceil_given_active = float("nan")
        
    mean_req = df["requested_current_a"].mean()
    mean_appl = df["applied_current_a"].mean()
    
    results = {
        "controller": controller_name,
        "sample_size_N": n,
        "P_ceiling_active": p_ceil_active,
        "P_req_gt_ceiling": p_req_gt_ceil,
        "P_req_le_ceiling_given_active": p_req_le_ceil_given_active,
        "P_appl_eq_ceiling": p_appl_eq_ceil,
        "mean_requested_current_a": mean_req,
        "mean_applied_current_a": mean_appl,
    }
    
    print("=" * 80)
    print(f"SAFETY DOMINANCE ANALYSIS: {controller_name.upper()} (N = {n:,} steps)")
    print("=" * 80)
    print(f"  P(safety ceiling active):                 {p_ceil_active:.4f} ({p_ceil_active*100:.2f}%)")
    print(f"  P(requested > safe ceiling):              {p_req_gt_ceil:.4f} ({p_req_gt_ceil*100:.2f}%)")
    print(f"  P(requested <= ceiling | ceiling active): {p_req_le_ceil_given_active:.4f} ({p_req_le_ceil_given_active*100:.2f}%)")
    print(f"  P(applied == safe ceiling):               {p_appl_eq_ceil:.4f} ({p_appl_eq_ceil*100:.2f}%)")
    print(f"  mean(requested_current):                  {mean_req:.2f} A")
    print(f"  mean(applied_current):                    {mean_appl:.2f} A")
    print()
    
    return results


def main():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
    
    # 1. Max Current Baseline
    max_curr_ctrl = MaxCurrentController(battery_cfg)
    def max_current_action(obs, e):
        # Maps 160A request to action=1.0
        return np.array([1.0], dtype=np.float32)
        
    df_max_curr = collect_trajectory_stats(max_current_action, env)
    res_max = analyze_dominance(df_max_curr, "Max Current Baseline")
    
    # 2. PPO Baseline (Run 001) if available
    run001_model_path = os.path.join("runs", "run_001", "trained_model.zip")
    all_results = [res_max]
    
    if os.path.exists(run001_model_path):
        model = PPO.load(run001_model_path)
        def ppo_action(obs, e):
            act, _ = model.predict(obs, deterministic=True)
            return act
            
        df_ppo = collect_trajectory_stats(ppo_action, env)
        res_ppo = analyze_dominance(df_ppo, "PPO Baseline (run_001)")
        all_results.append(res_ppo)
        
    # Save results to CSV
    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "safety_dominance_results.csv")
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"Summary saved to {out_csv}")


if __name__ == "__main__":
    main()
