"""
Track B — Candidate B3 checkpoint-quality diagnostic.

PURPOSE: The Track B freeze (audit/long_training_freeze_v2.md, condition
B8) is marked "PASS (qualified)" because the original Cand_B3_100k run
(experiments/driving_candidate_B3_100k.py) only captured end-of-training
evaluation, not per-chunk train/approx_kl, train/explained_variance,
train/value_loss, train/entropy_loss, train/std, rollout/ep_rew_mean, or
rollout/ep_len_mean. This script is diagnostic-only: it re-runs the
IDENTICAL frozen B3 configuration (same weights, same ent_coef, same
100k budget, same seeds) with stable-baselines3's CSV logger attached,
so those curves exist. It is NOT a new reward search and does not
change CAND_WEIGHTS/ENT_COEF/TIMESTEPS -- those are imported directly
from driving_candidate_B3_100k.py so there is no possibility of drift
between "what was gated" and "what is being curve-checked".

Per-chunk metrics land in:
    audit/driving_Cand_B3_100k_training_curves_seed{seed}.csv
(one row per PPO rollout/update, ~100000/2048 ~= 49 rows/seed)

Usage:
    python -m experiments.driving_B3_checkpoint_quality_diagnostic --seed=7
    python -m experiments.driving_B3_checkpoint_quality_diagnostic --seed=21
    python -m experiments.driving_B3_checkpoint_quality_diagnostic --seed=42
    python -m experiments.driving_B3_checkpoint_quality_diagnostic --combine
"""
from __future__ import annotations

import copy
import glob
import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure as configure_sb3_logger
from stable_baselines3.common.monitor import Monitor

from environment.ev_energy_env import EVEnergyEnv
from experiments.driving_candidate_B3_100k import CAND_NAME, CAND_WEIGHTS, ENT_COEF, TIMESTEPS
from training.evaluate_drive_ems import STANDARD_CYCLES, evaluate_all_cycles
from utils.config import load_config
from utils.seed import set_global_seed

SEEDS = [7, 21, 42]
OUT_PREFIX = f"driving_{CAND_NAME}_training_curves"


def run_single_seed(seed: int):
    out_csv = os.path.join("audit", f"{OUT_PREFIX}_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"Seed {seed} curves already captured, skipping.")
        return

    set_global_seed(seed)
    ppo_cfg = load_config("ppo_drive_ems")
    energy_cfg = copy.deepcopy(load_config("energy_management"))
    for k, v in CAND_WEIGHTS.items():
        energy_cfg[k] = v

    raw_env = EVEnergyEnv(
        battery_config=load_config("battery"),
        safety_config=load_config("safety"),
        vehicle_config=load_config("vehicle"),
        drivetrain_config=load_config("drivetrain"),
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
        ent_coef=ENT_COEF,
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=dict(net_arch=ppo_cfg["policy_kwargs"]["net_arch"]),
        target_kl=ppo_cfg.get("target_kl"),
        seed=seed,
        verbose=0,
        use_sde=False,
    )

    log_dir = os.path.join("runs", f"driving_{CAND_NAME}_curvecheck", f"seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)
    sb3_logger = configure_sb3_logger(log_dir, ["csv"])
    model.set_logger(sb3_logger)

    print(f">>> Training {CAND_NAME} (seed {seed}, {TIMESTEPS} steps) with curve logging <<<")
    model.learn(total_timesteps=TIMESTEPS, log_interval=1)

    # Sanity-check consistency with the originally-gated Cand_B3_100k run
    # (same config, same seed -> should reproduce closely).
    df_summary, _ = evaluate_all_cycles(model, is_ppo=True, controller_name=f"{CAND_NAME}_curvecheck_seed{seed}")
    mean_wh = float(df_summary["wh_per_km"].mean())
    mean_regen = float(df_summary["regen_recovery_fraction"].mean()) * 100.0
    print(f"    Sanity check: mean Wh/km={mean_wh:.2f}, mean regen={mean_regen:.1f}%")

    # SB3's CSV logger writes to log_dir/progress.csv
    progress_path = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(progress_path):
        raise FileNotFoundError(f"Expected SB3 progress.csv at {progress_path}, not found")

    df_curves = pd.read_csv(progress_path)
    df_curves["seed"] = seed
    df_curves["candidate"] = CAND_NAME
    df_curves["sanity_check_wh_per_km"] = mean_wh
    df_curves["sanity_check_regen_pct"] = mean_regen
    os.makedirs("audit", exist_ok=True)
    df_curves.to_csv(out_csv, index=False)
    print(f"Saved {len(df_curves)} chunk rows to {out_csv}")


def combine_results():
    frames = []
    for seed in SEEDS:
        path = os.path.join("audit", f"{OUT_PREFIX}_seed{seed}.csv")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    combined_path = os.path.join("audit", f"{OUT_PREFIX}_combined.csv")
    df.to_csv(combined_path, index=False)
    print(f"Combined {len(df)} rows -> {combined_path}")

    cols = [c for c in [
        "seed", "time/total_timesteps", "train/approx_kl", "train/explained_variance",
        "train/value_loss", "train/entropy_loss", "train/std",
        "rollout/ep_rew_mean", "rollout/ep_len_mean",
    ] if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--combine":
        combine_results()
    elif len(sys.argv) > 1 and sys.argv[1].startswith("--seed="):
        run_single_seed(int(sys.argv[1].split("=")[1]))
    else:
        for seed in SEEDS:
            run_single_seed(seed)
        combine_results()
