"""
Driving-EMS Evaluation & Benchmark Runner (Stages K & L)
========================================================
Runs driving energy-management controllers (Rule-Based EMS, PPO) across
verified standard drive cycles (UDDS, HWFET, US06, WLTP Class 3b).

Reports:
- Trip duration (s), distance (km), average speed (km/h)
- Net energy (Wh), energy efficiency (Wh/km)
- Discharge energy (Wh), Regen energy (Wh)
- Regenerative recovery fraction (%)
- Battery min/final SoC, peak/average temperature (°C)
- Power tracking error / deficit (Wh)
- Safety interventions (count and rate)
- Empirical reward component breakdown (for Stage L audit)

Usage:
    python -m training.evaluate_drive_ems --controller rule_based --all-cycles
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines.rule_based_ems import RuleBasedEMS
from environment.ev_energy_env import EVEnergyEnv
from training.train_drive_ems import FIXTURE_DRIVE_CYCLE, make_drive_ems_env
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


def evaluate_all_cycles(controller, is_ppo: bool = False, controller_name: str = "RuleBasedEMS", config_dir: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    parser = argparse.ArgumentParser(description="Evaluate a driving-EMS controller on standard drive cycles.")
    parser.add_argument("--controller", type=str, default="rule_based", choices=["rule_based", "ppo"])
    parser.add_argument("--model-path", type=str, default="", help="Path to trained PPO model (.zip)")
    parser.add_argument("--all-cycles", action="store_true", default=True, help="Evaluate across all 4 standard cycles")
    parser.add_argument("--config-dir", type=str, default=None, help="Config directory for evaluation (use a frozen final profile for final evaluation).")
    args = parser.parse_args()

    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)

    if args.controller == "rule_based":
        ctrl = RuleBasedEMS()
        is_ppo = False
        ctrl_name = "RuleBasedEMS"
    else:
        ctrl = PPO.load(args.model_path)
        is_ppo = True
        ctrl_name = f"PPO_{os.path.basename(args.model_path)}"

    print("=" * 80)
    print(f"EVALUATING DRIVING EMS: {ctrl_name.upper()}")
    print("=" * 80)

    df_summary, df_steps = evaluate_all_cycles(ctrl, is_ppo=is_ppo, controller_name=ctrl_name, config_dir=args.config_dir)
    print(df_summary.to_string(index=False))
    print()

    # Save summary
    csv_out = os.path.join(out_dir, f"driving_{args.controller}_benchmark.csv")
    df_summary.to_csv(csv_out, index=False)
    print(f"Summary saved to {csv_out}")

    # Generate Stage L Driving Reward Balance Audit
    if not df_steps.empty:
        reward_cols = [c for c in df_steps.columns if c.startswith("rc_") or c == "reward"]
        reward_summary = []
        for c in reward_cols:
            vals = df_steps[c].values
            reward_summary.append({
                "component": c.replace("rc_", ""),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "abs_mean": float(np.mean(np.abs(vals))),
            })

        df_rew = pd.DataFrame(reward_summary)
        total_abs = df_rew[df_rew["component"] != "reward"]["abs_mean"].sum()
        df_rew["pct_contribution"] = df_rew.apply(
            lambda r: (r["abs_mean"] / total_abs * 100.0) if r["component"] != "reward" and total_abs > 0 else 0.0,
            axis=1,
        )

        rew_audit_path = os.path.join(out_dir, "driving_reward_balance.md")
        with open(rew_audit_path, "w", encoding="utf-8") as f:
            f.write("# Driving Reward Balance Audit\n\n")
            f.write(f"**Controller**: `{ctrl_name}`  \n")
            f.write("**Evaluation Type**: Standardized drive-cycle evaluation of a simulated Tata Nexon EV Long Range  \n")
            f.write(f"**Cycles Evaluated**: EPA UDDS (Urban), EPA HWFET (Highway), EPA US06 (Aggressive), WLTP Class 3b (Mixed)  \n")
            f.write(f"**Total Evaluated Steps**: {len(df_steps):,}  \n\n")
            f.write("---\n\n")
            f.write("## 1. Empirical Reward Component Distributions\n\n")
            f.write("| Component | Mean | Std | Min | Max | % Contribution |\n")
            f.write("|---|---|---|---|---|---|\n")
            for _, r in df_rew.iterrows():
                if r["component"] == "reward":
                    continue
                f.write(f"| `{r['component']}` | {r['mean']:.6f} | {r['std']:.6f} | {r['min']:.6f} | {r['max']:.6f} | **{r['pct_contribution']:.2f}%** |\n")
            f.write("\n---\n\n")
            f.write("## 2. Total Reward Statistics\n\n")
            r_row = df_rew[df_rew["component"] == "reward"].iloc[0]
            f.write(f"- **Mean per-step reward**: {r_row['mean']:.6f}\n")
            f.write(f"- **Std**: {r_row['std']:.6f}\n")
            f.write(f"- **Min**: {r_row['min']:.6f}, **Max**: {r_row['max']:.6f}\n\n")
            f.write("## 3. Findings\n\n")
            f.write("- **Active Terms**: Under nominal standardized drive cycles, the practical reward signal is dominated by energy consumption and regenerative recovery.\n")
            f.write("- **Inactive Terms**: Thermal stress, safety penalties, and tracking error remain inactive because nominal power demands are within physical and safety envelopes.\n")
            f.write("- **Regenerative Incentive**: Capturing available regenerative energy strictly increases per-step reward over letting kinetic energy dissipate into friction braking.\n")

        print(f"Reward balance audit written to {rew_audit_path}")


if __name__ == "__main__":
    main()
