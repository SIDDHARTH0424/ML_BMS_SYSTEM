"""
Driving-EMS PPO Diagnostic & Benchmark Runner (Stages M & N)
============================================================
1. Trains PPO on the Driving-EMS environment across seeds [7, 21, 42]
   using the WLTP Class 3b cycle as the standard mixed training profile.
2. Tracks training dynamics:
   - approx_kl, clip_fraction, entropy_loss, explained_variance, value_loss, std
3. Evaluates all trained models across all 4 standard cycles:
   - EPA UDDS (Urban)
   - EPA HWFET (Highway)
   - EPA US06 (Aggressive)
   - WLTP Class 3b (Mixed)
4. Verifies physical energy conservation:
   - Delta_E_battery = Discharge_E - Regen_E + Losses
5. Compares against RuleBasedEMS baseline.

Usage:
    python -m experiments.diagnostic_driving_ppo
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from baselines.rule_based_ems import RuleBasedEMS
from environment.ev_energy_env import EVEnergyEnv
from training.evaluate_drive_ems import STANDARD_CYCLES, evaluate_all_cycles, run_episode
from training.train_drive_ems import make_drive_ems_env
from utils.config import load_config
from utils.seed import set_global_seed


SEEDS = [7, 21, 42]
TRAINING_CYCLE = STANDARD_CYCLES["wltp_class3b"]
DIAGNOSTIC_TIMESTEPS = 50000


class RolloutEpisodeTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.chunk_rewards: List[float] = []
        self.chunk_lengths: List[int] = []
        self.total_completed: int = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.chunk_rewards.append(float(info["episode"]["r"]))
                self.chunk_lengths.append(int(info["episode"]["l"]))
                self.total_completed += 1
        return True

    def get_chunk_metrics(self, model: PPO) -> Dict[str, float]:
        if len(self.chunk_rewards) > 0:
            ep_rew = float(np.mean(self.chunk_rewards))
            ep_len = float(np.mean(self.chunk_lengths))
            n_eps = len(self.chunk_rewards)
        elif hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
            ep_rew = float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
            ep_len = float(np.mean([ep["l"] for ep in model.ep_info_buffer]))
            n_eps = len(model.ep_info_buffer)
        else:
            ep_rew = float("nan")
            ep_len = float("nan")
            n_eps = 0

        return {
            "rollout/ep_rew_mean": ep_rew,
            "rollout/ep_len_mean": ep_len,
            "rollout/episodes_in_chunk": float(n_eps),
            "rollout/total_episodes": float(self.total_completed),
        }

    def reset_chunk(self):
        self.chunk_rewards = []
        self.chunk_lengths = []


def train_driving_ppo(seed: int, timesteps: int = DIAGNOSTIC_TIMESTEPS) -> Tuple[PPO, List[Dict]]:
    set_global_seed(seed)
    ppo_cfg = load_config("ppo_drive_ems")
    
    env = Monitor(make_drive_ems_env(drive_cycle_path=TRAINING_CYCLE, mode="train"))
    n_steps = ppo_cfg["n_steps"]
    
    model = PPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=n_steps,
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
    
    tracker = RolloutEpisodeTracker()
    n_chunks = int(np.ceil(timesteps / n_steps))
    training_log = []
    
    for c in range(n_chunks):
        tracker.reset_chunk()
        model.learn(total_timesteps=n_steps, reset_num_timesteps=False, callback=tracker)
        snap = dict(model.logger.name_to_value)
        rollout_stats = tracker.get_chunk_metrics(model)
        
        with torch.no_grad():
            if hasattr(model.policy, "log_std"):
                action_std = float(model.policy.log_std.exp().mean().item())
            else:
                action_std = float("nan")
                
        row = {
            "seed": seed,
            "total_steps": (c + 1) * n_steps,
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
        }
        training_log.append(row)
        
    return model, training_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Driving-EMS PPO Diagnostic & Multi-Cycle Benchmark")
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Experiment identifier (e.g. driving_ppo_baseline_refresh, driving_ppo_stageQ)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Allow overwriting existing outputs")
    parser.add_argument("--timesteps", type=int, default=DIAGNOSTIC_TIMESTEPS,
                        help="Total timesteps per seed (default: 50,000)")
    args = parser.parse_args()

    exp_name = args.experiment_name
    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)

    benchmark_csv = os.path.join(out_dir, f"driving_multicycle_benchmark_{exp_name}.csv")
    curves_csv = os.path.join(out_dir, f"driving_ppo_training_curves_{exp_name}.csv")

    if not args.overwrite:
        if os.path.exists(benchmark_csv) or os.path.exists(curves_csv):
            raise FileExistsError(
                f"Output files for experiment '{exp_name}' already exist in {out_dir}/. "
                "Use --overwrite to overwrite, or choose a new --experiment-name."
            )

    print("=" * 80)
    print(f"DRIVING PPO 3-SEED DIAGNOSTIC & BENCHMARK: {exp_name.upper()}")
    print(f"Seeds: {SEEDS}, Timesteps: {args.timesteps} per seed")
    print(f"Training cycle: {TRAINING_CYCLE}")
    print("=" * 80)

    all_eval_summaries = []
    all_training_curves = []

    # 1. Evaluate RuleBasedEMS across all cycles for reference
    rule_ctrl = RuleBasedEMS()
    df_rule, _ = evaluate_all_cycles(rule_ctrl, is_ppo=False, controller_name="RuleBasedEMS")
    all_eval_summaries.append(df_rule)

    # 2. Train and evaluate PPO across seeds
    for seed in SEEDS:
        print(f"\n>>> Training Driving PPO (Seed {seed}) <<<")
        model, curves = train_driving_ppo(seed, timesteps=args.timesteps)
        for c_row in curves:
            c_row["experiment"] = exp_name
        all_training_curves.extend(curves)

        # Save model checkpoint
        ckpt_dir = os.path.join("runs", exp_name, f"seed_{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(os.path.join(ckpt_dir, "trained_model.zip"))

        print(f"Evaluating Seed {seed} across standard cycles...")
        df_ppo_eval, _ = evaluate_all_cycles(model, is_ppo=True, controller_name=f"PPO_seed{seed}")
        df_ppo_eval["experiment"] = exp_name
        all_eval_summaries.append(df_ppo_eval)
        print(df_ppo_eval[["cycle_id", "distance_km", "wh_per_km", "regen_recovery_fraction", "total_power_deficit_wh"]].to_string(index=False))

    df_all_eval = pd.concat(all_eval_summaries, ignore_index=True)
    df_all_curves = pd.DataFrame(all_training_curves)

    # Save versioned artifacts
    df_all_eval.to_csv(benchmark_csv, index=False)
    df_all_curves.to_csv(curves_csv, index=False)

    print("\n" + "=" * 80)
    print(f"MULTI-CYCLE PPO VS RULE-BASED EMS COMPARISON ({exp_name})")
    print("=" * 80)
    cols = ["controller", "cycle_id", "distance_km", "wh_per_km", "regen_recovery_fraction", "delta_soc", "total_power_deficit_wh"]
    print(df_all_eval[cols].to_string(index=False))

    # Cross-cycle averages
    print("\n--- CONTROLLER CROSS-CYCLE MEANS ---")
    grp = df_all_eval.groupby("controller")[["wh_per_km", "regen_recovery_fraction", "total_power_deficit_wh"]].agg(["mean", "std"])
    print(grp.to_string())
    print()
    print(f"Versioned artifacts successfully written to:")
    print(f"  {benchmark_csv}")
    print(f"  {curves_csv}")
    print(f"  runs/{exp_name}/")


if __name__ == "__main__":
    main()
