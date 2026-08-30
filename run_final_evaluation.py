"""
Final research evaluation script for RL-BMS-Driving.
Evaluates PPO and Rule-Based EMS across all standard drive cycles and seeds.
Saves raw trajectories and summary metrics.
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines.rule_based_ems import RuleBasedEMS
from environment.ev_energy_env import EVEnergyEnv
from training.train_drive_ems import make_drive_ems_env
from utils.config import load_config
from utils.metrics import (
    average_temperature_c,
    distance_km,
    driving_energy_wh_breakdown,
    minimum_soc,
    peak_temperature_c,
    regen_recovery_fraction,
    safety_interventions,
    wh_per_km,
)

STANDARD_CYCLES = {
    "epa_udds": os.path.join("data", "drive_cycles", "standard", "epa_udds", "cycle.csv"),
    "epa_hwfet": os.path.join("data", "drive_cycles", "standard", "epa_hwfet", "cycle.csv"),
    "epa_us06": os.path.join("data", "drive_cycles", "standard", "epa_us06", "cycle.csv"),
    "wltp_class3b": os.path.join("data", "drive_cycles", "standard", "wltp_class3b", "cycle.csv"),
}

def run_episode(
    controller,
    env: EVEnergyEnv,
    initial_soc: float = 0.5,
    ambient_temp_c: float = 25.0,
    seed: int = 42,
    is_ppo: bool = False,
    return_step_df: bool = False,
):
    obs, info = env.reset(seed=seed, options={"initial_soc": initial_soc, "ambient_temp_c": ambient_temp_c})
    if hasattr(controller, "reset"):
        controller.reset()

    socs = [env._state.soc]
    temps = [env._state.temperature_c]
    speeds = []
    applied_powers_w = []
    available_regen_w = []
    intervention_flags = []
    power_deficits_w = []
    step_records = []

    while True:
        speeds.append(env._drive_cycle.current_speed())
        if is_ppo:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            a = controller.act(obs)
            action = np.array([a], dtype=np.float32)

        obs, reward, term, trunc, step_info = env.step(action)

        socs.append(env._state.soc)
        temps.append(env._state.temperature_c)
        applied_powers_w.append(step_info["applied_power_w"])
        power_deficits_w.append(step_info["power_deficit_w"])
        available_regen = max(0.0, step_info["applied_power_w"]) + step_info["friction_braking_w"]
        available_regen_w.append(available_regen)
        intervention_flags.append(step_info["safety_intervention"]["type"] != "none")

        step_record = {
            "step": len(applied_powers_w),
            "speed_mps": speeds[-1],
            "applied_power_w": step_info["applied_power_w"],
            "power_deficit_w": step_info["power_deficit_w"],
            "friction_braking_w": step_info["friction_braking_w"],
            "soc": env._state.soc,
            "temperature_c": env._state.temperature_c,
            "reward": reward,
            **{f"rc_{k}": v for k, v in step_info["reward_components"].items()},
        }
        step_records.append(step_record)

        if term or trunc:
            break

    dt = env.dt
    dist_km = distance_km(speeds, dt)
    energy = driving_energy_wh_breakdown(applied_powers_w, dt)
    available_regen_energy_wh = float(np.sum(available_regen_w) * dt / 3600.0)
    total_deficit_wh = float(np.sum(power_deficits_w) * dt / 3600.0)

    metrics = {
        "steps": len(applied_powers_w),
        "distance_km": round(dist_km, 3),
        "mean_speed_kmh": round((dist_km / (len(applied_powers_w) * dt / 3600.0)) if dist_km > 0 else 0.0, 1),
        "discharge_energy_wh": round(energy["discharge_energy_wh"], 2),
        "regen_energy_wh": round(energy["regen_energy_wh"], 2),
        "net_energy_wh": round(energy["net_energy_wh"], 2),
        "wh_per_km": round(wh_per_km(energy["net_energy_wh"], dist_km), 2),
        "regen_recovery_fraction": round(regen_recovery_fraction(energy["regen_energy_wh"], available_regen_energy_wh), 4),
        "min_soc": round(minimum_soc(socs), 4),
        "final_soc": round(socs[-1], 4),
        "delta_soc": round(socs[0] - socs[-1], 4),
        "max_temperature_c": round(peak_temperature_c(temps), 2),
        "avg_temperature_c": round(average_temperature_c(temps), 2),
        "total_power_deficit_wh": round(total_deficit_wh, 2),
        "safety_interventions": safety_interventions(intervention_flags),
        "safety_intervention_rate": round(float(np.mean(intervention_flags)) if intervention_flags else 0.0, 4),
    }

    if return_step_df:
        df_steps = pd.DataFrame(step_records)
        return metrics, df_steps
    return metrics

def evaluate_all_cycles(controller, is_ppo: bool = False, controller_name: str = "RuleBasedEMS", config_dir: str | None = None):
    summary_rows = []
    all_step_data = []

    for cycle_id, cycle_path in STANDARD_CYCLES.items():
        if not os.path.exists(cycle_path):
            print(f"Skipping {cycle_id} (file not found: {cycle_path})")
            continue

        env = make_drive_ems_env(drive_cycle_path=cycle_path, mode="eval", config_dir=config_dir)
        metrics, df_steps = run_episode(controller, env, initial_soc=0.50, ambient_temp_c=25.0, is_ppo=is_ppo, return_step_df=True)
        df_steps["cycle_id"] = cycle_id
        df_steps["controller"] = controller_name

        summary_rows.append({"controller": controller_name, "cycle_id": cycle_id, **metrics})
        all_step_data.append(df_steps)

    df_summary = pd.DataFrame(summary_rows)
    df_all_steps = pd.concat(all_step_data, ignore_index=True) if all_step_data else pd.DataFrame()
    return df_summary, df_all_steps

def main():
    # Use final_driving configuration for evaluation
    config_dir = os.path.join("configs", "final_driving")
    out_dir = "audit/final_research"
    os.makedirs(out_dir, exist_ok=True)

    # Evaluate Rule-Based EMS (no seed)
    print("=" * 80)
    print("EVALUATING RULE-BASED EMS")
    print("=" * 80)
    rb_controller = RuleBasedEMS()
    rb_summary, rb_steps = evaluate_all_cycles(rb_controller, is_ppo=False, controller_name="RuleBasedEMS", config_dir=config_dir)
    print(rb_summary.to_string(index=False))
    rb_summary.to_csv(os.path.join(out_dir, "driving_rule_based_summary.csv"), index=False)
    rb_steps.to_csv(os.path.join(out_dir, "driving_rule_based_steps.csv"), index=False)
    print(f"Rule-Based summary saved to {os.path.join(out_dir, 'driving_rule_based_summary.csv')}")
    print(f"Rule-Based steps saved to {os.path.join(out_dir, 'driving_rule_based_steps.csv')}")

    # Evaluate PPO for each seed
    seeds = [7, 21, 42]
    for seed in seeds:
        model_path = f"final_models/driving_B3_100k_seed{seed}/ppo_driving_100000_steps.zip"
        print("=" * 80)
        print(f"EVALUATING PPO DRIVING MODEL SEED {seed}")
        print("=" * 80)
        ppo_controller = PPO.load(model_path, device="cpu")
        ppo_summary, ppo_steps = evaluate_all_cycles(ppo_controller, is_ppo=True, controller_name=f"PPO_seed{seed}", config_dir=config_dir)
        print(ppo_summary.to_string(index=False))
        ppo_summary.to_csv(os.path.join(out_dir, f"driving_ppo_seed{seed}_summary.csv"), index=False)
        ppo_steps.to_csv(os.path.join(out_dir, f"driving_ppo_seed{seed}_steps.csv"), index=False)
        print(f"PPO seed {seed} summary saved to {os.path.join(out_dir, f'driving_ppo_seed{seed}_summary.csv')}")
        print(f"PPO seed {seed} steps saved to {os.path.join(out_dir, f'driving_ppo_seed{seed}_steps.csv')}")

if __name__ == "__main__":
    main()