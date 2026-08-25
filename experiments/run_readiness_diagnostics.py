"""
Final Long-Training Readiness Diagnostics Runner
================================================
Runs the controlled candidate diagnostics for Track A (Charging) and Track B (Driving).

TRACK A:
  Candidate A1: T_ref=35.0, scale=20.0, w_th=2.5, w_time=0.05
  Candidate A2: T_ref=36.0, scale=19.0, w_th=3.0, w_time=0.05
  - 50k steps per seed [7, 21, 42] with 75% normal [15,35] + 25% stress [35,45] sampler.
  - Evaluates standard 15 scenarios and 6 extended stress scenarios.
  - Strict time gate: baseline = 2094.6s, max allowed = 2199.3s (+5.0%).

TRACK B:
  Candidate B1: w_regen=1.0, w_energy=0.2, w_track=1.5, ent_coef=0.005
  Candidate B2: w_regen=1.2, w_energy=0.2, w_track=1.5, ent_coef=0.010
  - 50k steps per seed [7, 21, 42] on WLTP Class 3b.
  - Evaluates UDDS, HWFET, US06, WLTP Class 3b.
  - Strict gate: cross-cycle mean Wh/km <= 129.16, regen > 85%.

Saves:
  - audit/charging_candidates_metrics.csv
  - audit/charging_candidates_stress_eval.csv
  - audit/driving_candidates_benchmark.csv
  - audit/readiness_diagnostic_report.md
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from baselines.rule_based_ems import RuleBasedEMS
from environment.battery_env import BatteryChargingEnv
from environment.ev_energy_env import EVEnergyEnv
from training.evaluate_drive_ems import STANDARD_CYCLES, evaluate_all_cycles
from training.train_drive_ems import make_drive_ems_env
from utils.config import load_config
from utils.seed import set_global_seed


SEEDS = [7, 21, 42]
TIMESTEPS_CHARGING = 50000
TIMESTEPS_DRIVING = 50000

MAX_ALLOWED_CHARGING_TIME_S = 2199.3  # 2094.6 * 1.05 (+5.0%)
RULE_BASED_WH_PER_KM = 129.16


# =========================================================================== #
# TRACK A: Charging Diagnostics
# =========================================================================== #

class ChargingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ambients: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                amb = info.get("ambient_temp_c")
                if amb is not None and not np.isnan(float(amb)):
                    self.episode_ambients.append(float(amb))
        return True


def train_and_eval_charging_candidate(cand_name: str, cand_cfg: Dict, seed: int) -> Tuple[Dict, List[Dict]]:
    set_global_seed(seed)
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = copy.deepcopy(load_config("reward"))
    sim_cfg = copy.deepcopy(load_config("simulation"))
    ppo_cfg = load_config("ppo")

    # Apply candidate overrides
    for k, v in cand_cfg.items():
        if k == "time_penalty_weight":
            reward_cfg["weights"]["time_penalty"] = v
        else:
            reward_cfg[k] = v

    sim_cfg["train"]["mixed_ambient_sampler"] = {
        "p_stress": 0.25,
        "normal_range_c": [15.0, 35.0],
        "stress_range_c": [35.0, 45.0],
    }

    raw_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env = Monitor(raw_env)

    model = PPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=dict(net_arch=ppo_cfg["policy_kwargs"]["net_arch"]),
        target_kl=ppo_cfg.get("target_kl"),
        seed=seed,
        verbose=0,
        use_sde=False,
    )

    cb = ChargingCallback()
    model.learn(total_timesteps=TIMESTEPS_CHARGING, callback=cb)

    # 1. Standard 15-scenario evaluation
    eval_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
    dt = eval_env.dt
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]

    times, peak_temps, mean_req_is, mean_appl_is, reached_list = [], [], [], [], []
    q_gens = []

    for s0 in soc_grid:
        for t0 in temp_grid:
            obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
            temps, reqs, apps = [eval_env._state.temperature_c], [], []
            tot_q = 0.0
            term = trunc = False
            steps = 0
            while not (term or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(act)
                reqs.append(info["requested_current"])
                apps.append(info["applied_current"])
                temps.append(eval_env._state.temperature_c)
                tot_q += info["q_gen"] * dt
                steps += 1

            times.append(steps * dt)
            peak_temps.append(max(temps))
            mean_req_is.append(np.mean(reqs))
            mean_appl_is.append(np.mean(apps))
            q_gens.append(tot_q)
            reached_list.append(bool(info.get("target_reached")))

    n_stress_ep = sum(1 for a in cb.episode_ambients if a >= 35.0)
    std_summary = {
        "candidate": cand_name,
        "seed": seed,
        "reached_target_all": all(reached_list),
        "mean_charging_time_s": float(np.mean(times)),
        "max_temperature_max_c": float(np.max(peak_temps)),
        "mean_requested_current_a": float(np.mean(mean_req_is)),
        "mean_applied_current_a": float(np.mean(mean_appl_is)),
        "cumulative_q_gen_j": float(np.mean(q_gens)),
        "total_train_episodes": len(cb.episode_ambients),
        "actual_stress_fraction": (n_stress_ep / max(1, len(cb.episode_ambients))) if cb.episode_ambients else float("nan"),
        "mean_train_ambient_c": float(np.mean(cb.episode_ambients)) if cb.episode_ambients else float("nan"),
    }

    # 2. Extended Stress evaluation (45C & 50C)
    stress_rows = []
    for s0 in [0.10, 0.20, 0.30]:
        for t0 in [45.0, 50.0]:
            obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
            temps, reqs, apps = [eval_env._state.temperature_c], [], []
            tot_q = 0.0
            term = trunc = False
            steps = 0
            while not (term or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(act)
                reqs.append(info["requested_current"])
                apps.append(info["applied_current"])
                temps.append(eval_env._state.temperature_c)
                tot_q += info["q_gen"] * dt
                steps += 1

            stress_rows.append({
                "candidate": cand_name,
                "seed": seed,
                "initial_soc": s0,
                "ambient_temp_c": t0,
                "reached_target": bool(info.get("target_reached")),
                "charging_time_s": steps * dt,
                "peak_temperature_c": float(np.max(temps)),
                "mean_requested_current_a": float(np.mean(reqs)),
                "mean_applied_current_a": float(np.mean(apps)),
                "cumulative_q_gen_j": tot_q,
            })

    # Save model checkpoint
    ckpt_dir = os.path.join("runs", f"charging_{cand_name}", f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "trained_model.zip"))

    return std_summary, stress_rows


# =========================================================================== #
# TRACK B: Driving Diagnostics
# =========================================================================== #

def train_and_eval_driving_candidate(cand_name: str, cand_weights: Dict, ent_coef: float, seed: int) -> Tuple[List[Dict], Dict]:
    set_global_seed(seed)
    ppo_cfg = load_config("ppo_drive_ems")
    energy_cfg = copy.deepcopy(load_config("energy_management"))

    for k, v in cand_weights.items():
        energy_cfg[k] = v

    # Custom environment factory with candidate energy config
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    vehicle_cfg = load_config("vehicle")
    drivetrain_cfg = load_config("drivetrain")

    raw_env = EVEnergyEnv(
        battery_config=battery_cfg,
        safety_config=safety_cfg,
        vehicle_config=vehicle_cfg,
        drivetrain_config=drivetrain_cfg,
        energy_config=energy_cfg,
        drive_cycle_path=STANDARD_CYCLES["wltp_class3b"],
        mode="train",
    )
    env = Monitor(raw_env)

    model = PPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ent_coef,
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=dict(net_arch=ppo_cfg["policy_kwargs"]["net_arch"]),
        target_kl=ppo_cfg.get("target_kl"),
        seed=seed,
        verbose=0,
        use_sde=False,
    )

    model.learn(total_timesteps=TIMESTEPS_DRIVING)

    # Evaluate across all 4 cycles
    df_summary, _ = evaluate_all_cycles(model, is_ppo=True, controller_name=f"{cand_name}_seed{seed}")
    df_summary["seed"] = seed
    df_summary["candidate"] = cand_name
    df_summary["ent_coef"] = ent_coef

    # Save model checkpoint
    ckpt_dir = os.path.join("runs", f"driving_{cand_name}", f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "trained_model.zip"))

    eval_records = df_summary.to_dict("records")
    mean_wh = float(df_summary["wh_per_km"].mean())
    mean_regen = float(df_summary["regen_recovery_fraction"].mean()) * 100.0

    summary = {
        "candidate": cand_name,
        "seed": seed,
        "ent_coef": ent_coef,
        "mean_wh_per_km": mean_wh,
        "mean_regen_recovery_pct": mean_regen,
    }
    return eval_records, summary


# =========================================================================== #
# Main Execution & Reporting
# =========================================================================== #

def main():
    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING FINAL LONG-TRAINING READINESS DIAGNOSTICS")
    print("=" * 80)

    # --- TRACK A CANDIDATES ---
    cand_a_configs = {
        "Cand_A1": {"thermal_enabled": True, "thermal_weight": 2.5, "thermal_reference_temp_c": 35.0, "thermal_scale_c": 20.0, "time_penalty_weight": 0.05},
        "Cand_A2": {"thermal_enabled": True, "thermal_weight": 3.0, "thermal_reference_temp_c": 36.0, "thermal_scale_c": 19.0, "time_penalty_weight": 0.05},
    }

    track_a_std_metrics = []
    track_a_stress_metrics = []

    print("\n>>> TRACK A: RUNNING CHARGING CANDIDATES DIAGNOSTIC <<<")
    for c_name, c_cfg in cand_a_configs.items():
        print(f"\n--- Testing Track A: {c_name} ---")
        for s in SEEDS:
            ckpt_path = os.path.join("runs", f"charging_{c_name}", f"seed_{s}", "trained_model.zip")
            if os.path.exists(ckpt_path):
                print(f"  Found existing checkpoint for {c_name} Seed {s}, loading...")
                model = PPO.load(ckpt_path)
                battery_cfg = load_config("battery")
                safety_cfg = load_config("safety")
                reward_cfg = copy.deepcopy(load_config("reward"))
                sim_cfg = copy.deepcopy(load_config("simulation"))
                for k, v in c_cfg.items():
                    if k == "time_penalty_weight":
                        reward_cfg["weights"]["time_penalty"] = v
                    else:
                        reward_cfg[k] = v
                eval_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
                dt = eval_env.dt
                soc_grid = sim_cfg["eval"]["initial_soc_grid"]
                temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
                times, peak_temps, mean_req_is, mean_appl_is, reached_list = [], [], [], [], []
                q_gens = []
                for s0 in soc_grid:
                    for t0 in temp_grid:
                        obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
                        temps, reqs, apps = [eval_env._state.temperature_c], [], []
                        tot_q = 0.0
                        term = trunc = False
                        steps = 0
                        while not (term or trunc):
                            act, _ = model.predict(obs, deterministic=True)
                            obs, r, term, trunc, info = eval_env.step(act)
                            reqs.append(info["requested_current"])
                            apps.append(info["applied_current"])
                            temps.append(eval_env._state.temperature_c)
                            tot_q += info["q_gen"] * dt
                            steps += 1
                        times.append(steps * dt)
                        peak_temps.append(max(temps))
                        mean_req_is.append(np.mean(reqs))
                        mean_appl_is.append(np.mean(apps))
                        q_gens.append(tot_q)
                        reached_list.append(bool(info.get("target_reached")))
                std_res = {
                    "candidate": c_name,
                    "seed": s,
                    "reached_target_all": all(reached_list),
                    "mean_charging_time_s": float(np.mean(times)),
                    "max_temperature_max_c": float(np.max(peak_temps)),
                    "mean_requested_current_a": float(np.mean(mean_req_is)),
                    "mean_applied_current_a": float(np.mean(mean_appl_is)),
                    "cumulative_q_gen_j": float(np.mean(q_gens)),
                }
                # Stress evaluation
                stress_res = []
                for s0 in [0.10, 0.20, 0.30]:
                    for t0 in [45.0, 50.0]:
                        obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
                        temps, reqs, apps = [eval_env._state.temperature_c], [], []
                        tot_q = 0.0
                        term = trunc = False
                        steps = 0
                        while not (term or trunc):
                            act, _ = model.predict(obs, deterministic=True)
                            obs, r, term, trunc, info = eval_env.step(act)
                            reqs.append(info["requested_current"])
                            apps.append(info["applied_current"])
                            temps.append(eval_env._state.temperature_c)
                            tot_q += info["q_gen"] * dt
                            steps += 1
                        stress_res.append({
                            "candidate": c_name,
                            "seed": s,
                            "initial_soc": s0,
                            "ambient_temp_c": t0,
                            "reached_target": bool(info.get("target_reached")),
                            "charging_time_s": steps * dt,
                            "peak_temperature_c": float(np.max(temps)),
                            "mean_requested_current_a": float(np.mean(reqs)),
                            "mean_applied_current_a": float(np.mean(apps)),
                            "cumulative_q_gen_j": tot_q,
                        })
            else:
                print(f"  Training Seed {s} (50k steps)...")
                std_res, stress_res = train_and_eval_charging_candidate(c_name, c_cfg, s)
            track_a_std_metrics.append(std_res)
            track_a_stress_metrics.extend(stress_res)
            dt_pct = (std_res['mean_charging_time_s'] - 2094.6) / 2094.6 * 100.0
            print(f"    Std Eval Seed {s}: Time={std_res['mean_charging_time_s']:.1f}s ({dt_pct:+.2f}%), "
                  f"Peak_T={std_res['max_temperature_max_c']:.2f}C, "
                  f"Req_I={std_res['mean_requested_current_a']:.1f}A, "
                  f"Gate_Time={'PASS' if std_res['mean_charging_time_s'] <= MAX_ALLOWED_CHARGING_TIME_S else 'FAIL'}")

    df_a_std = pd.DataFrame(track_a_std_metrics)
    df_a_stress = pd.DataFrame(track_a_stress_metrics)
    df_a_std.to_csv(os.path.join(out_dir, "charging_candidates_metrics.csv"), index=False)
    df_a_stress.to_csv(os.path.join(out_dir, "charging_candidates_stress_eval.csv"), index=False)

    # --- TRACK B CANDIDATES ---
    cand_b_configs = {
        "Cand_B1": {"weights": {"w_regen_recovery": 1.0, "w_energy_cost": 0.2, "w_tracking_error": 1.5}, "ent_coef": 0.005},
        "Cand_B2": {"weights": {"w_regen_recovery": 1.2, "w_energy_cost": 0.2, "w_tracking_error": 1.5}, "ent_coef": 0.010},
    }

    track_b_eval_rows = []
    track_b_summaries = []

    print("\n>>> TRACK B: RUNNING DRIVING CANDIDATES DIAGNOSTIC <<<")
    for c_name, c_info in cand_b_configs.items():
        print(f"\n--- Testing Track B: {c_name} ---")
        for s in SEEDS:
            print(f"  Training Seed {s} (50k steps, ent_coef={c_info['ent_coef']})...")
            eval_rows, summ = train_and_eval_driving_candidate(c_name, c_info["weights"], c_info["ent_coef"], s)
            track_b_eval_rows.extend(eval_rows)
            track_b_summaries.append(summ)
            print(f"    Eval Seed {s}: Mean Wh/km={summ['mean_wh_per_km']:.2f} "
                  f"(vs 129.16: {summ['mean_wh_per_km'] - RULE_BASED_WH_PER_KM:+.2f}), "
                  f"Regen={summ['mean_regen_recovery_pct']:.1f}%, "
                  f"Gate_Whkm={'PASS' if summ['mean_wh_per_km'] <= RULE_BASED_WH_PER_KM else 'FAIL'}")

    df_b_eval = pd.DataFrame(track_b_eval_rows)
    df_b_summ = pd.DataFrame(track_b_summaries)
    df_b_eval.to_csv(os.path.join(out_dir, "driving_candidates_benchmark.csv"), index=False)

    # --- GENERATE COMPREHENSIVE REPORT ---
    md_path = os.path.join(out_dir, "readiness_diagnostic_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Final Long-Training Readiness Diagnostic Report\n\n")
        f.write(f"**Evaluation Timestamp**: {pd.Timestamp.now().isoformat()}  \n\n")
        f.write("---\n\n")

        # Track A Table
        f.write("## 1. Track A: Charging BMS Candidate Results\n\n")
        f.write("Strict Time Gate: $\\le 2199.3\\text{s}$ ($+5.0\\%$ vs $2094.6\\text{s}$ baseline)\n\n")
        f.write("| Candidate | Seed | Target Reached (15/15) | Mean Charging Time (s) | $\\Delta$ Time vs Baseline (%) | Peak Temp (°C) | Mean Req Current (A) | Time Gate Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in df_a_std.iterrows():
            dt_pct = (r["mean_charging_time_s"] - 2094.6) / 2094.6 * 100.0
            gate_pass = bool(r["mean_charging_time_s"] <= MAX_ALLOWED_CHARGING_TIME_S)
            f.write(f"| {r['candidate']} | {r['seed']} | {r['reached_target_all']} | {r['mean_charging_time_s']:.1f} | {dt_pct:+.2f}% | {r['max_temperature_max_c']:.2f} | {r['mean_requested_current_a']:.1f} | **{'PASS' if gate_pass else 'FAIL'}** |\n")

        f.write("\n---\n\n")

        # Track B Table
        f.write("## 2. Track B: Driving EMS Candidate Results\n\n")
        f.write("Strict Efficiency Gate: Cross-Cycle Mean $\\le 129.16\\text{ Wh/km}$ and Regen Recovery $> 85\\%$\n\n")
        f.write("| Candidate | Seed | ent_coef | Mean Wh/km | $\\Delta$ vs Rule-Based (129.16) | Regen Recovery (%) | Gate Status |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in df_b_summ.iterrows():
            d_wh = r["mean_wh_per_km"] - RULE_BASED_WH_PER_KM
            gate_pass = bool(r["mean_wh_per_km"] <= RULE_BASED_WH_PER_KM and r["mean_regen_recovery_pct"] > 85.0)
            f.write(f"| {r['candidate']} | {r['seed']} | {r['ent_coef']} | {r['mean_wh_per_km']:.2f} | {d_wh:+.2f} | {r['mean_regen_recovery_pct']:.1f}% | **{'PASS' if gate_pass else 'FAIL'}** |\n")

    print("\n" + "=" * 80)
    print("READINESS DIAGNOSTIC RUN COMPLETE")
    print(f"Report saved to {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
