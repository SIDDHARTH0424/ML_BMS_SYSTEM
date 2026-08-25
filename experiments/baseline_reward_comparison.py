"""
Baseline Reward Comparison & Thermal Engagement Test (Fix Items 10, 11)
========================================================================
1. Controlled comparison of all controllers in the same environment:
   - Max Current
   - Constant Current (CC)
   - CCCV
   - Adaptive
   - PPO baseline (run_001)
   Logs every reward component per-step to identify why Max Current is ranked
   as optimal under the original reward formulation.

2. Thermal Objective Engagement Test (Item 11):
   Deterministic test pushing the environment through representative
   temperatures to verify:
     - T below reference              -> near-zero thermal penalty
     - T moderately above reference   -> moderate penalty
     - T substantially above reference -> stronger penalty
     - higher q_gen                   -> stronger thermal penalty

Usage:
    python -m experiments.baseline_reward_comparison
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines.adaptive import AdaptiveController
from baselines.cc import ConstantCurrentController, MaxCurrentController
from baselines.cccv import CCCVController
from environment.battery_env import BatteryChargingEnv
from environment.ecm_model import BatteryECM, BatteryState
from safety.safety_layer import safety_layer
from utils.config import load_config


def run_controller_with_reward_logging(controller_name: str, controller, env: BatteryChargingEnv,
                                       sim_cfg: dict) -> Dict[str, float]:
    """Runs one controller across the evaluation grid, logging every reward component."""
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    
    comp_sums: Dict[str, float] = {}
    total_steps = 0
    total_reward = 0.0
    episodes_data = []
    
    for soc0 in soc_grid:
        for temp0 in temp_grid:
            obs, _ = env.reset(options={"initial_soc": soc0, "ambient_temp_c": temp0})
            if hasattr(controller, "reset"):
                controller.reset()
                
            terminated = truncated = False
            ep_rew = 0.0
            ep_steps = 0
            
            while not (terminated or truncated):
                if controller_name == "ppo":
                    action, _ = controller.predict(obs, deterministic=True)
                elif controller_name == "max_current":
                    action = np.array([1.0], dtype=np.float32)
                else:
                    v = env.ecm.terminal_voltage(env._state, env._prev_current_a)
                    obs_dict = {
                        "soc": env._state.soc,
                        "terminal_voltage": v,
                        "temperature_c": env._state.temperature_c,
                        "previous_current_a": env._prev_current_a,
                        "ambient_temp_c": temp0,
                    }
                    req = controller.act(obs_dict)
                    # Convert req A to [-1, 1] action space
                    action = np.array([(req / env.i_max) * 2.0 - 1.0], dtype=np.float32)
                    
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rew += reward
                ep_steps += 1
                total_reward += reward
                
                comps = info.get("reward_components", {})
                for k, v in comps.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v
                    
            total_steps += ep_steps
            episodes_data.append({
                "controller": controller_name,
                "initial_soc": soc0,
                "ambient_temp_c": temp0,
                "steps": ep_steps,
                "episode_reward": ep_rew,
                "final_soc": env._state.soc,
                "target_reached": info.get("target_reached", False),
            })
            
    mean_comps = {f"mean_{k}": v / max(1, total_steps) for k, v in comp_sums.items()}
    summary = {
        "controller": controller_name,
        "total_steps": total_steps,
        "mean_reward_per_step": total_reward / max(1, total_steps),
        "mean_episode_reward": total_reward / len(episodes_data),
        **mean_comps,
    }
    return summary, pd.DataFrame(episodes_data)


def test_thermal_objective_engagement(reward_config_to_test: dict) -> pd.DataFrame:
    """Deterministic reward test pushing representative temperatures and currents."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    sim_cfg = load_config("simulation")
    
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_config_to_test, sim_cfg)
    
    t_ref = reward_config_to_test.get("thermal_reference_temp_c", 33.0)
    t_scale = reward_config_to_test.get("thermal_scale_c", 22.0)
    q_ref = reward_config_to_test.get("thermal_q_reference_w", 1190.4)
    w_th = reward_config_to_test.get("thermal_weight", 0.5)
    
    test_conditions = [
        {"desc": "T well below ref (25C), full current 160A", "temp_c": 25.0, "current_a": 160.0},
        {"desc": "T below ref (30C), full current 160A",      "temp_c": 30.0, "current_a": 160.0},
        {"desc": "T at ref (33C), full current 160A",         "temp_c": 33.0, "current_a": 160.0},
        {"desc": "T mod above ref (38C), half current 80A",   "temp_c": 38.0, "current_a": 80.0},
        {"desc": "T mod above ref (38C), full current 160A",  "temp_c": 38.0, "current_a": 160.0},
        {"desc": "T subst above ref (45C), half current 80A", "temp_c": 45.0, "current_a": 80.0},
        {"desc": "T subst above ref (45C), full current 160A","temp_c": 45.0, "current_a": 160.0},
        {"desc": "T near cutoff (50C), zero current 0A",      "temp_c": 50.0, "current_a": 0.0},
        {"desc": "T near cutoff (50C), half current 80A",     "temp_c": 50.0, "current_a": 80.0},
        {"desc": "T near cutoff (50C), full current 160A",    "temp_c": 50.0, "current_a": 160.0},
    ]
    
    rows = []
    for cond in test_conditions:
        state = BatteryState(soc=0.5, v_rc=1.0, temperature_c=cond["temp_c"], soh=1.0, ah_throughput=0.0)
        appl_curr = cond["current_a"]
        q_gen = env.ecm.heat_generation_w(state, appl_curr)
        
        # Calculate thermal reward directly using env's formula
        thermal_excess = max(0.0, state.temperature_c - t_ref)
        normalized_excess = thermal_excess / t_scale
        normalized_q_gen = q_gen / q_ref
        thermal_reward = w_th * (normalized_excess ** 2) * normalized_q_gen
        
        rows.append({
            "condition": cond["desc"],
            "temperature_c": cond["temp_c"],
            "current_a": appl_curr,
            "q_gen_w": round(q_gen, 2),
            "thermal_excess_c": round(thermal_excess, 2),
            "thermal_penalty": round(thermal_reward, 6),
        })
        
    return pd.DataFrame(rows)


def main():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    eval_cfg = load_config("evaluation")
    
    print("=" * 80)
    print("1. CONTROLLED BASELINE REWARD COMPARISON (ORIGINAL REWARD)")
    print("=" * 80)
    
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
    
    controllers = {
        "max_current": MaxCurrentController(battery_cfg),
        "cc": ConstantCurrentController(eval_cfg["cc"]),
        "cccv": CCCVController(eval_cfg["cccv"]),
        "adaptive": AdaptiveController(eval_cfg["adaptive"]),
    }
    
    summaries = []
    
    for name, ctrl in controllers.items():
        s, _ = run_controller_with_reward_logging(name, ctrl, env, sim_cfg)
        summaries.append(s)
        
    # PPO Baseline if present
    run001_model_path = os.path.join("runs", "run_001", "trained_model.zip")
    if os.path.exists(run001_model_path):
        model = PPO.load(run001_model_path)
        s_ppo, _ = run_controller_with_reward_logging("ppo", model, env, sim_cfg)
        summaries.append(s_ppo)
        
    df_summary = pd.DataFrame(summaries)
    print(df_summary[["controller", "mean_episode_reward", "mean_reward_per_step", 
                      "mean_progress", "mean_smoothness_penalty", "mean_time_penalty", 
                      "mean_safety_penalty"]].to_string(index=False))
    print()
    
    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(out_dir, "baseline_reward_comparison.csv"), index=False)
    
    print("=" * 80)
    print("2. THERMAL OBJECTIVE ENGAGEMENT TEST (REVISED FORMULATION)")
    print("=" * 80)
    
    revised_reward_cfg = copy.deepcopy(reward_cfg)
    revised_reward_cfg["thermal_enabled"] = True
    revised_reward_cfg["thermal_reference_temp_c"] = 33.0
    revised_reward_cfg["thermal_scale_c"] = 22.0
    
    df_thermal = test_thermal_objective_engagement(revised_reward_cfg)
    print(df_thermal.to_string(index=False))
    print()
    
    df_thermal.to_csv(os.path.join(out_dir, "thermal_engagement_test.csv"), index=False)
    print(f"Artifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
