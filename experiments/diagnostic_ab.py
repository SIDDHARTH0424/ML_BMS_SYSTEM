"""
Diagnostic Experiments A/B/C
====================================================
Runs controlled diagnostic experiments:
  - Experiment A: Original PPO baseline (thermal_enabled: false)
  - Experiment B: PPO with state-aware thermal reward (thermal_enabled: true, ref=33C, normal training dist)
  - Experiment C (NEW): PPO with state-aware thermal reward + HIGH-AMBIENT STRESS-TEST distribution
    (75% episodes from 15-35C normal, 25% from 35-45C stress-test; documented in configs/simulation.yaml)
    NOTE: The stress-test distribution is a CONTROLLED EXPERIMENT to expose the thermal
    reward mechanism to PPO. It is NOT a claim about normal operating conditions.

Seeds: 7, 21, 42
Budget: 50,000 steps per seed
Architecture: Original PPO (use_sde=False, squash_output=False, target_kl=0.01)

Outputs:
  - audit/diagnostic_ab_seed_metrics.csv        (A/B only, immutable — not regenerated)
  - audit/diagnostic_ab_training_curves.csv     (A/B only, immutable — not regenerated)
  - audit/diagnostic_abc_seed_metrics.csv       (A/B/C combined)
  - audit/diagnostic_abc_training_curves.csv    (A/B/C combined)
  - audit/charging_stress_eval.csv              (Exp C stress-grid results, separate table)

Usage:
    python -m experiments.diagnostic_ab
"""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from baselines.adaptive import AdaptiveController
from baselines.cc import ConstantCurrentController, MaxCurrentController
from baselines.cccv import CCCVController
from environment.battery_env import BatteryChargingEnv
from environment.ecm_model import BatteryECM, BatteryState
from safety.safety_layer import safety_layer
from utils.config import load_config
from utils.metrics import charging_time_s, energy_efficiency, energy_per_percent_soc_wh, final_soc, peak_temperature_c, target_reached, target_shortfall
from utils.seed import set_global_seed


SEEDS = [7, 21, 42]
TOTAL_TIMESTEPS = 50000


class ComponentAccumulator(BaseCallback):
    def __init__(self):
        super().__init__()
        self.sums: Dict[str, float] = {}
        self.counts = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_ambients: List[float] = []
        self.total_completed: int = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            comps = info.get("reward_components")
            if comps:
                self.counts += 1
                for k, v in comps.items():
                    self.sums[k] = self.sums.get(k, 0.0) + v
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                amb = info.get("ambient_temp_c")
                if amb is not None:
                    amb_val = float(amb)
                    if not (np.isnan(amb_val) or np.isinf(amb_val)):
                        self.episode_ambients.append(amb_val)
                self.total_completed += 1
        return True

    def means(self) -> Dict[str, float]:
        return {k: v / max(1, self.counts) for k, v in self.sums.items()}

    def get_rollout_metrics(self, model: PPO) -> Dict[str, float]:
        if len(self.episode_rewards) > 0:
            ep_rew = float(np.mean(self.episode_rewards))
            ep_len = float(np.mean(self.episode_lengths))
            n_eps = len(self.episode_rewards)
        elif hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
            ep_rew = float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
            ep_len = float(np.mean([ep["l"] for ep in model.ep_info_buffer]))
            n_eps = len(model.ep_info_buffer)
        else:
            ep_rew = float("nan")
            ep_len = float("nan")
            n_eps = 0

        # Ambient sampling statistics
        ambs = self.episode_ambients
        n_stress = sum(1 for a in ambs if a > 35.0)
        n_normal = len(ambs) - n_stress
        actual_stress_frac = (n_stress / max(1, len(ambs))) if ambs else float("nan")
        mean_amb = float(np.mean(ambs)) if ambs else float("nan")

        return {
            "rollout/ep_rew_mean": ep_rew,
            "rollout/ep_len_mean": ep_len,
            "rollout/episodes_in_chunk": float(n_eps),
            "rollout/total_episodes": float(self.total_completed),
            "train/actual_normal_episodes": float(n_normal),
            "train/actual_stress_episodes": float(n_stress),
            "train/actual_stress_fraction": actual_stress_frac,
            "train/mean_ambient_c": mean_amb,
        }

    def reset(self):
        self.sums = {}
        self.counts = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_ambients = []


def make_training_env(reward_overrides: dict, sim_overrides: dict = None) -> BatteryChargingEnv:
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    sim_cfg = load_config("simulation")
    reward_cfg = copy.deepcopy(load_config("reward"))
    reward_cfg.update(reward_overrides)
    if sim_overrides:
        sim_cfg = copy.deepcopy(sim_cfg)
        for k, v in sim_overrides.items():
            sim_cfg["train"][k] = v
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")


def train_diagnostic(seed: int, reward_overrides: dict, exp_name: str,
                     sim_overrides: dict = None) -> Tuple[PPO, List[Dict], Dict]:
    set_global_seed(seed)
    ppo_cfg = load_config("ppo")

    raw_env = make_training_env(reward_overrides, sim_overrides=sim_overrides)
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

    cb = ComponentAccumulator()
    training_log = []

    n_chunks = int(np.ceil(TOTAL_TIMESTEPS / ppo_cfg["n_steps"]))
    steps_per_chunk = ppo_cfg["n_steps"]

    all_sampled_ambients = []

    for c in range(n_chunks):
        cb.reset()
        model.learn(total_timesteps=steps_per_chunk, reset_num_timesteps=False, callback=cb)
        snap = dict(model.logger.name_to_value)
        rollout_stats = cb.get_rollout_metrics(model)
        all_sampled_ambients.extend(cb.episode_ambients)

        # Policy std from action distribution parameter
        with torch.no_grad():
            if hasattr(model.policy, "log_std"):
                action_std = float(model.policy.log_std.exp().mean().item())
            else:
                action_std = float("nan")

        row = {
            "experiment": exp_name,
            "seed": seed,
            "total_steps": (c + 1) * steps_per_chunk,
            "rollout/ep_rew_mean": rollout_stats["rollout/ep_rew_mean"],
            "rollout/ep_len_mean": rollout_stats["rollout/ep_len_mean"],
            "rollout/episodes_in_chunk": rollout_stats["rollout/episodes_in_chunk"],
            "rollout/total_episodes": rollout_stats["rollout/total_episodes"],
            "train/approx_kl": snap.get("train/approx_kl", float("nan")),
            "train/clip_fraction": snap.get("train/clip_fraction", float("nan")),
            "train/entropy_loss": snap.get("train/entropy_loss", float("nan")),
            "train/explained_variance": snap.get("train/explained_variance", float("nan")),
            "train/value_loss": snap.get("train/value_loss", float("nan")),
            "train/policy_gradient_loss": snap.get("train/policy_gradient_loss", float("nan")),
            "train/std": action_std,
            "train/chunk_stress_fraction": rollout_stats["train/actual_stress_fraction"],
            "train/chunk_mean_ambient_c": rollout_stats["train/mean_ambient_c"],
        }
        for k, v in cb.means().items():
            row[f"rc_{k}"] = v
        training_log.append(row)

    # Cumulative training sampling stats across all completed episodes
    n_stress_tot = sum(1 for a in all_sampled_ambients if a >= 35.0)
    n_normal_tot = sum(1 for a in all_sampled_ambients if a < 35.0)
    tot_eps = len(all_sampled_ambients)
    run_sampling_stats = {
        "p_normal_target": 0.75 if (sim_overrides and "mixed_ambient_sampler" in sim_overrides) else 1.0,
        "p_stress_target": 0.25 if (sim_overrides and "mixed_ambient_sampler" in sim_overrides) else 0.0,
        "normal_episode_count": n_normal_tot,
        "stress_episode_count": n_stress_tot,
        "total_episodes": tot_eps,
        "actual_normal_fraction": (n_normal_tot / max(1, tot_eps)) if tot_eps > 0 else float("nan"),
        "actual_stress_fraction": (n_stress_tot / max(1, tot_eps)) if tot_eps > 0 else float("nan"),
        "mean_training_ambient_c": float(np.mean(all_sampled_ambients)) if tot_eps > 0 else float("nan"),
        "min_training_ambient_c": float(np.min(all_sampled_ambients)) if tot_eps > 0 else float("nan"),
        "max_training_ambient_c": float(np.max(all_sampled_ambients)) if tot_eps > 0 else float("nan"),
        "normal_ambient_range": "[15.0, 35.0]",
        "stress_ambient_range": "[35.0, 45.0]",
    }

    return model, training_log, run_sampling_stats


def evaluate_model_full(model: PPO, exp_name: str, seed: int) -> Tuple[Dict, pd.DataFrame]:
    """Full evaluation over the complete 15-scenario evaluation grid."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
    dt = env.dt
    target_s = env.target_soc
    i_max = env.i_max
    
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    
    all_episodes = []
    
    for soc0 in soc_grid:
        for temp0 in temp_grid:
            obs, _ = env.reset(options={"initial_soc": soc0, "ambient_temp_c": temp0})
            temps, reqs, apps = [env._state.temperature_c], [], []
            total_q_j, input_wh, stored_wh = 0.0, 0.0, 0.0
            interventions = 0
            soh0 = env._state.soh
            total_rew = 0.0
            comp_sums: Dict[str, float] = {}
            
            terminated = truncated = False
            steps = 0
            
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                prev_s = env._state
                obs, reward, terminated, truncated, info = env.step(action)
                
                appl = info["applied_current"]
                req = info["requested_current"]
                vt = info["terminal_voltage"]
                q = info["q_gen"]
                
                temps.append(env._state.temperature_c)
                reqs.append(req)
                apps.append(appl)
                total_q_j += q * dt
                input_wh += appl * vt * dt / 3600.0
                stored_wh += appl * env.ecm.ocv(env._state.soc) * dt / 3600.0
                if info["safety_intervention"]["type"] != "none":
                    interventions += 1
                total_rew += reward
                
                comps = info.get("reward_components", {})
                for ck, cv in comps.items():
                    comp_sums[ck] = comp_sums.get(ck, 0.0) + cv
                steps += 1
                
            reached = bool(info.get("target_reached"))
            final_s = env._state.soc
            soh_loss = soh0 - env._state.soh
            ch_time = charging_time_s(dt, steps)
            dpct = (final_s - soc0) * 100.0
            
            ep_res = {
                "experiment": exp_name,
                "seed": seed,
                "initial_soc": soc0,
                "ambient_temp_c": temp0,
                "reached_target": reached,
                "final_soc": final_s,
                "charging_time_s": ch_time,
                "target_shortfall": target_shortfall(final_s, target_s),
                "max_temperature_c": peak_temperature_c(temps),
                "mean_temperature_c": float(np.mean(temps)),
                "cumulative_q_gen_j": total_q_j,
                "mean_requested_current_a": float(np.mean(reqs)),
                "mean_applied_current_a": float(np.mean(apps)),
                "safety_intervention_rate": interventions / max(1, steps),
                "energy_efficiency": energy_efficiency(input_wh, stored_wh),
                "energy_per_percent_soc_wh": energy_per_percent_soc_wh(input_wh, final_s - soc0),
                "SoH_loss": soh_loss,
                "total_reward": total_rew,
                "mean_progress_reward": comp_sums.get("progress", 0.0) / max(1, steps),
                "mean_thermal_reward": comp_sums.get("thermal_reward", 0.0) / max(1, steps),
                "mean_safety_penalty": comp_sums.get("safety_penalty", 0.0) / max(1, steps),
                "mean_overrequest_penalty": comp_sums.get("overrequest_penalty", 0.0) / max(1, steps),
                "mean_smoothness_penalty": comp_sums.get("smoothness_penalty", 0.0) / max(1, steps),
            }
            all_episodes.append(ep_res)
            
    df_episodes = pd.DataFrame(all_episodes)
    
    # Aggregate across 15 scenarios for this seed
    seed_summary = {
        "experiment": exp_name,
        "seed": seed,
        "reached_target_all": bool(df_episodes["reached_target"].all()),
        "target_reached_rate": float(df_episodes["reached_target"].mean()),
        "final_soc_mean": float(df_episodes["final_soc"].mean()),
        "charging_time_mean_s": float(df_episodes["charging_time_s"].mean()),
        "target_shortfall_mean": float(df_episodes["target_shortfall"].mean()),
        "max_temperature_max_c": float(df_episodes["max_temperature_c"].max()),
        "mean_temperature_mean_c": float(df_episodes["mean_temperature_c"].mean()),
        "cumulative_q_gen_mean_j": float(df_episodes["cumulative_q_gen_j"].mean()),
        "mean_requested_current_a": float(df_episodes["mean_requested_current_a"].mean()),
        "mean_applied_current_a": float(df_episodes["mean_applied_current_a"].mean()),
        "safety_intervention_rate_mean": float(df_episodes["safety_intervention_rate"].mean()),
        "energy_efficiency_mean": float(df_episodes["energy_efficiency"].mean()),
        "energy_per_pct_soc_mean_wh": float(df_episodes["energy_per_percent_soc_wh"].mean()),
        "SoH_loss_mean": float(df_episodes["SoH_loss"].mean()),
        "total_reward_mean": float(df_episodes["total_reward"].mean()),
        "mean_progress_reward": float(df_episodes["mean_progress_reward"].mean()),
        "mean_thermal_reward": float(df_episodes["mean_thermal_reward"].mean()),
        "mean_safety_penalty": float(df_episodes["mean_safety_penalty"].mean()),
        "mean_overrequest_penalty": float(df_episodes["mean_overrequest_penalty"].mean()),
        "mean_smoothness_penalty": float(df_episodes["mean_smoothness_penalty"].mean()),
    }
    
    return seed_summary, df_episodes


def evaluate_baselines() -> pd.DataFrame:
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    eval_cfg = load_config("evaluation")
    
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
    dt = env.dt
    target_s = env.target_soc
    
    controllers = {
        "max_current": MaxCurrentController(battery_cfg),
        "cc": ConstantCurrentController(eval_cfg["cc"]),
        "cccv": CCCVController(eval_cfg["cccv"]),
        "adaptive": AdaptiveController(eval_cfg["adaptive"]),
    }
    
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    
    summary_rows = []
    
    for c_name, ctrl in controllers.items():
        ep_rows = []
        for soc0 in soc_grid:
            for temp0 in temp_grid:
                obs, _ = env.reset(options={"initial_soc": soc0, "ambient_temp_c": temp0})
                if hasattr(ctrl, "reset"):
                    ctrl.reset()
                temps, reqs, apps = [env._state.temperature_c], [], []
                total_q_j, input_wh, stored_wh = 0.0, 0.0, 0.0
                interventions = 0
                soh0 = env._state.soh
                steps = 0
                terminated = truncated = False
                
                while not (terminated or truncated):
                    if c_name == "max_current":
                        act = np.array([1.0], dtype=np.float32)
                    else:
                        v = env.ecm.terminal_voltage(env._state, env._prev_current_a)
                        obs_dict = {"soc": env._state.soc, "terminal_voltage": v, "temperature_c": env._state.temperature_c,
                                    "previous_current_a": env._prev_current_a, "ambient_temp_c": temp0}
                        req = ctrl.act(obs_dict)
                        act = np.array([(req / env.i_max) * 2.0 - 1.0], dtype=np.float32)
                        
                    obs, rew, terminated, truncated, info = env.step(act)
                    appl = info["applied_current"]
                    req = info["requested_current"]
                    vt = info["terminal_voltage"]
                    q = info["q_gen"]
                    temps.append(env._state.temperature_c)
                    reqs.append(req)
                    apps.append(appl)
                    total_q_j += q * dt
                    input_wh += appl * vt * dt / 3600.0
                    stored_wh += appl * env.ecm.ocv(env._state.soc) * dt / 3600.0
                    if info["safety_intervention"]["type"] != "none":
                        interventions += 1
                    steps += 1
                    
                reached = bool(info.get("target_reached"))
                final_s = env._state.soc
                ch_time = charging_time_s(dt, steps)
                ep_rows.append({
                    "controller": c_name,
                    "reached_target": reached,
                    "final_soc": final_s,
                    "charging_time_s": ch_time,
                    "max_temperature_c": peak_temperature_c(temps),
                    "mean_temperature_c": float(np.mean(temps)),
                    "cumulative_q_gen_j": total_q_j,
                    "mean_requested_current_a": float(np.mean(reqs)),
                    "mean_applied_current_a": float(np.mean(apps)),
                    "safety_intervention_rate": interventions / max(1, steps),
                    "energy_efficiency": energy_efficiency(input_wh, stored_wh),
                    "energy_per_percent_soc_wh": energy_per_percent_soc_wh(input_wh, final_s - soc0),
                    "SoH_loss": soh0 - env._state.soh,
                })
                
        df_c = pd.DataFrame(ep_rows)
        summary_rows.append({
            "controller": c_name,
            "reached_target_all": bool(df_c["reached_target"].all()),
            "charging_time_mean_s": float(df_c["charging_time_s"].mean()),
            "max_temperature_max_c": float(df_c["max_temperature_c"].max()),
            "mean_temperature_mean_c": float(df_c["mean_temperature_c"].mean()),
            "cumulative_q_gen_mean_j": float(df_c["cumulative_q_gen_j"].mean()),
            "mean_requested_current_a": float(df_c["mean_requested_current_a"].mean()),
            "mean_applied_current_a": float(df_c["mean_applied_current_a"].mean()),
            "safety_intervention_rate_mean": float(df_c["safety_intervention_rate"].mean()),
            "energy_efficiency_mean": float(df_c["energy_efficiency"].mean()),
            "energy_per_pct_soc_mean_wh": float(df_c["energy_per_percent_soc_wh"].mean()),
            "SoH_loss_mean": float(df_c["SoH_loss"].mean()),
        })
        
    return pd.DataFrame(summary_rows)


def evaluate_model_stress(model, exp_name: str, seed: int) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Evaluate a trained model on the STRESS-TEST grid (45C, 50C ambient).
    Results are reported in a SEPARATE table, NEVER averaged with the standard 15-scenario grid.
    """
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")

    stress_cfg = sim_cfg.get("eval_stress", {})
    soc_grid = stress_cfg.get("initial_soc_grid", [0.10, 0.20, 0.30])
    temp_grid = stress_cfg.get("ambient_temp_grid_c", [45.0, 50.0])

    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
    dt = env.dt
    i_max = env.i_max

    stress_rows = []
    all_step_data = []

    for soc0 in soc_grid:
        for temp0 in temp_grid:
            obs, _ = env.reset(options={"initial_soc": soc0, "ambient_temp_c": temp0})
            temps, reqs, apps = [env._state.temperature_c], [], []
            total_q_j, input_wh, stored_wh = 0.0, 0.0, 0.0
            interventions = 0
            steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                appl = info["applied_current"]
                req = info["requested_current"]
                vt = info["terminal_voltage"]
                q = info["q_gen"]
                temps.append(env._state.temperature_c)
                reqs.append(req)
                apps.append(appl)
                total_q_j += q * dt
                input_wh += appl * vt * dt / 3600.0
                stored_wh += appl * env.ecm.ocv(env._state.soc) * dt / 3600.0
                if info["safety_intervention"]["type"] != "none":
                    interventions += 1
                steps += 1

            reached = bool(info.get("target_reached"))
            final_s = env._state.soc

            # Separate high-T from low-T steps for derating analysis
            hot_reqs = [r for r, t in zip(reqs, temps[1:]) if t > 42.0]
            cold_reqs = [r for r, t in zip(reqs, temps[1:]) if t < 38.0]

            stress_rows.append({
                "experiment": exp_name,
                "seed": seed,
                "grid_type": "stress_test",
                "initial_soc": soc0,
                "ambient_temp_c": temp0,
                "reached_target": reached,
                "final_soc": final_s,
                "charging_time_s": dt * steps,
                "max_temperature_c": float(np.max(temps)),
                "mean_temperature_c": float(np.mean(temps)),
                "cumulative_q_gen_j": total_q_j,
                "mean_requested_current_a": float(np.mean(reqs)) if reqs else float("nan"),
                "mean_applied_current_a": float(np.mean(apps)) if apps else float("nan"),
                # Derating analysis: current in high-T vs low-T states
                "mean_req_current_hot_a": float(np.mean(hot_reqs)) if hot_reqs else float("nan"),
                "mean_req_current_cold_a": float(np.mean(cold_reqs)) if cold_reqs else float("nan"),
                "n_steps_hot": len(hot_reqs),
                "n_steps_cold": len(cold_reqs),
                "safety_intervention_rate": interventions / max(1, steps),
            })

    return stress_rows, pd.DataFrame(stress_rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostic Experiments A/B/C Runner")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "exp_a_original", "exp_b_revised_thermal", "exp_c_highambient"],
                        help="Which experiment to run (default: all)")
    args = parser.parse_args()

    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"STARTING DIAGNOSTIC EXPERIMENTS (Mode: {args.experiment.upper()})")
    print(f"Seeds: {SEEDS}, Budget: {TOTAL_TIMESTEPS} steps per run")
    print("=" * 80)
    print()

    sim_cfg = load_config("simulation")
    expC_mixed_cfg = sim_cfg.get("train_expC_mixed", {})
    p_stress = float(expC_mixed_cfg.get("p_stress", 0.25))

    experiments = {
        "exp_a_original": {
            "reward_overrides": {"thermal_enabled": False},
            "sim_overrides": None,
            "description": "Original PPO baseline (thermal_enabled: false, normal training dist [15-35C])",
        },
        "exp_b_revised_thermal": {
            "reward_overrides": {"thermal_enabled": True, "thermal_weight": 3.0,
                                  "thermal_reference_temp_c": 33.0, "thermal_scale_c": 22.0},
            "sim_overrides": None,
            "description": "Stage F thermal reward, normal training dist [15-35C]",
        },
        "exp_c_highambient": {
            "reward_overrides": {"thermal_enabled": True, "thermal_weight": 3.0,
                                  "thermal_reference_temp_c": 33.0, "thermal_scale_c": 22.0},
            "sim_overrides": {
                "mixed_ambient_sampler": {
                    "p_stress": p_stress,
                    "normal_range_c": [15.0, 35.0],
                    "stress_range_c": [35.0, 45.0],
                }
            },
            "description": (
                f"STRESS-TEST: thermal reward + TRUE MIXED SAMPLER ({int((1-p_stress)*100)}% normal [15-35C] + "
                f"{int(p_stress*100)}% stress [35-45C]). "
                "Logged per run. NOT a claim about normal operating conditions."
            ),
        },
    }

    selected_experiments = experiments if args.experiment == "all" else {args.experiment: experiments[args.experiment]}

    all_summaries = []
    all_training_curves = []
    exp_c_stress_results = []

    # If running only a subset, load existing summaries to preserve them
    existing_metrics_file = os.path.join(out_dir, "diagnostic_abc_seed_metrics.csv")
    if args.experiment != "all" and os.path.exists(existing_metrics_file):
        df_exist = pd.read_csv(existing_metrics_file)
        # Retain rows for experiments not being rerun
        df_keep = df_exist[~df_exist["experiment"].isin(selected_experiments.keys())]
        all_summaries.extend(df_keep.to_dict("records"))

    for exp_name, exp_cfg in selected_experiments.items():
        reward_overrides = exp_cfg["reward_overrides"]
        sim_overrides = exp_cfg["sim_overrides"]
        print(f"\n>>> Running Experiment: {exp_name.upper()} <<<")
        print(f"    {exp_cfg['description']}")
        for seed in SEEDS:
            print(f"  Training Seed {seed}...")
            model, curve, sampling_stats = train_diagnostic(seed, reward_overrides, exp_name, sim_overrides=sim_overrides)
            all_training_curves.extend(curve)

            print(f"  Sampling stats for Seed {seed}: "
                  f"target_stress={sampling_stats['p_stress_target']:.2f}, "
                  f"normal_eps={sampling_stats['normal_episode_count']}, "
                  f"stress_eps={sampling_stats['stress_episode_count']} "
                  f"(actual_stress_frac={sampling_stats['actual_stress_fraction']:.3f}), "
                  f"mean_ambient={sampling_stats['mean_training_ambient_c']:.1f}C")

            print(f"  Evaluating Seed {seed} across 15 standard scenarios...")
            summary, _ = evaluate_model_full(model, exp_name, seed)
            summary.update(sampling_stats)
            all_summaries.append(summary)
            print(f"    Seed {seed} standard eval: reached={summary['reached_target_all']}, "
                  f"mean_req={summary['mean_requested_current_a']:.2f}A, "
                  f"max_T={summary['max_temperature_max_c']:.2f}C, "
                  f"thermal_rew={summary['mean_thermal_reward']:.6f}")

            # Exp C also gets stress-grid evaluation (in a separate output)
            if exp_name == "exp_c_highambient":
                print(f"  Evaluating Seed {seed} on STRESS-TEST grid (45/50C ambient, separate table)...")
                stress_rows, _ = evaluate_model_stress(model, exp_name, seed)
                for r in stress_rows:
                    r.update(sampling_stats)
                exp_c_stress_results.extend(stress_rows)
                for r in stress_rows:
                    hot = r["mean_req_current_hot_a"]
                    cold = r["mean_req_current_cold_a"]
                    derating = "YES" if (not np.isnan(hot) and not np.isnan(cold) and hot < cold) else "N/A (no cold steps at 45/50C ambient)"
                    print(f"    Stress eval: amb={r['ambient_temp_c']:.0f}C, "
                          f"max_T={r['max_temperature_c']:.1f}C, "
                          f"req_hot={hot:.1f}A, req_cold={cold:.1f}A, "
                          f"derating_status={derating}")

            # Save model checkpoint for Exp C
            if exp_name == "exp_c_highambient":
                ckpt_dir = os.path.join("runs", "charging_expC_corrected", f"seed_{seed}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save(os.path.join(ckpt_dir, "trained_model.zip"))

    df_summaries = pd.DataFrame(all_summaries)
    df_curves = pd.DataFrame(all_training_curves)

    # Evaluate baselines
    print("\n>>> Evaluating Baseline Controllers <<<")
    df_baselines = evaluate_baselines()

    # Save A/B/C combined artifacts (both standard and v2 versioned names)
    df_summaries.to_csv(os.path.join(out_dir, "diagnostic_abc_seed_metrics.csv"), index=False)
    df_summaries.to_csv(os.path.join(out_dir, "diagnostic_abc_seed_metrics_v2.csv"), index=False)
    df_curves.to_csv(os.path.join(out_dir, "diagnostic_abc_training_curves.csv"), index=False)
    df_curves.to_csv(os.path.join(out_dir, "diagnostic_abc_training_curves_v2.csv"), index=False)
    df_baselines.to_csv(os.path.join(out_dir, "diagnostic_abc_baseline_comparison.csv"), index=False)

    if exp_c_stress_results:
        df_stress = pd.DataFrame(exp_c_stress_results)
        df_stress.to_csv(os.path.join(out_dir, "charging_stress_eval.csv"), index=False)
        df_stress.to_csv(os.path.join(out_dir, "charging_stress_eval_v2.csv"), index=False)
        print("\n--- EXP C STRESS-TEST GRID (SEPARATE TABLE — not averaged with standard grid) ---")
        stress_cols = ["seed", "ambient_temp_c", "reached_target", "max_temperature_c",
                       "mean_req_current_hot_a", "mean_req_current_cold_a", "n_steps_hot", "cumulative_q_gen_j"]
        print(df_stress[stress_cols].to_string(index=False))

    # Save Experiment C Reproducibility Record (Part 20)
    expC_meta_dir = os.path.join("runs", "charging_expC_corrected")
    os.makedirs(expC_meta_dir, exist_ok=True)
    import datetime, json
    expC_meta = {
        "experiment_name": "charging_expC_corrected",
        "seeds": SEEDS,
        "timesteps_per_seed": TOTAL_TIMESTEPS,
        "reward_configuration": {
            "thermal_enabled": True,
            "thermal_weight": 3.0,
            "thermal_reference_temp_c": 33.0,
            "thermal_scale_c": 22.0,
            "thermal_q_reference_w": 1190.4,
        },
        "sampling_distribution": {
            "p_normal_target": 0.75,
            "p_stress_target": 0.25,
            "normal_range_c": [15.0, 35.0],
            "stress_range_c": [35.0, 45.0],
        },
        "ppo_hyperparameters": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "target_kl": 0.01,
            "use_sde": False,
            "squash_output": False,
        },
        "git_status": "unversioned_local_workspace",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(expC_meta_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(expC_meta, f, indent=2)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC A/B/C EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print("\n--- PER-SEED STANDARD EVALUATION SUMMARY ---")
    cols_to_show = ["experiment", "seed", "reached_target_all", "charging_time_mean_s",
                    "max_temperature_max_c", "mean_requested_current_a", "mean_applied_current_a",
                    "actual_normal_episodes", "actual_stress_episodes", "actual_stress_fraction", "mean_training_ambient_c"]
    print(df_summaries[cols_to_show].to_string(index=False))

    print("\n--- BASELINES COMPARISON ---")
    print(df_baselines.to_string(index=False))
    print()
    print(f"A/B/C artifacts saved to {out_dir}/diagnostic_abc_*_v2.csv")
    print(f"Checkpoint and metadata saved to runs/charging_expC_corrected/")
    print(f"A/B-only artifacts (diagnostic_ab_*.csv) preserved and NOT overwritten.")


if __name__ == "__main__":
    main()
