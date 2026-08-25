"""
Track B — Candidate B2, extended training budget (100,000 steps/seed).

Single-variable isolation experiment: Candidate B2 (w_regen_recovery=1.2,
w_energy_cost=0.2, w_tracking_error=1.5, ent_coef=0.010) already showed the
best result at 50k steps (seed 42: 129.31 Wh/km, 96.0% regen) but seeds 7/21
had not yet discovered regen at 50k (83.9%/90.5%, cross-cycle mean 129.78 vs
gate 129.16). Per master-prompt Track B Step 7 (50k-100k allowed) and
Absolute Rule 8 (change one mechanism at a time), this extends ONLY the
timestep budget for the same candidate config -- reward weights and ent_coef
are untouched -- to test whether seeds 7/21 catch up with more exploration
budget, without conflating a budget effect with a reward-weight effect.

Does NOT overwrite Cand_B1/Cand_B2 50k results (audit/driving_candidates_*.csv,
runs/driving_Cand_B1, runs/driving_Cand_B2). New experiment name: Cand_B2_100k.

Usage:
    python -m experiments.driving_candidate_B2_100k
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment.ev_energy_env import EVEnergyEnv
from training.evaluate_drive_ems import STANDARD_CYCLES, evaluate_all_cycles
from utils.config import load_config
from utils.seed import set_global_seed

SEEDS = [7, 21, 42]
TIMESTEPS = 100_000
CAND_NAME = "Cand_B2_100k"
CAND_WEIGHTS = {"w_regen_recovery": 1.2, "w_energy_cost": 0.2, "w_tracking_error": 1.5}
ENT_COEF = 0.010
RULE_BASED_WH_PER_KM = 129.16


def train_and_eval(seed: int) -> Tuple[List[Dict], Dict]:
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

    model.learn(total_timesteps=TIMESTEPS)

    df_summary, _ = evaluate_all_cycles(model, is_ppo=True, controller_name=f"{CAND_NAME}_seed{seed}")
    df_summary["seed"] = seed
    df_summary["candidate"] = CAND_NAME
    df_summary["ent_coef"] = ENT_COEF
    df_summary["timesteps"] = TIMESTEPS

    ckpt_dir = os.path.join("runs", f"driving_{CAND_NAME}", f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "trained_model.zip"))

    eval_records = df_summary.to_dict("records")
    mean_wh = float(df_summary["wh_per_km"].mean())
    mean_regen = float(df_summary["regen_recovery_fraction"].mean()) * 100.0
    n_beat_rule_based = None  # filled in main() once rule-based per-cycle is available

    summary = {
        "candidate": CAND_NAME,
        "seed": seed,
        "ent_coef": ENT_COEF,
        "timesteps": TIMESTEPS,
        "mean_wh_per_km": mean_wh,
        "mean_regen_recovery_pct": mean_regen,
    }
    return eval_records, summary


def run_single_seed(seed: int):
    """Run + save ONE seed's result to its own file (fits under sandbox
    per-call time limits; combine_results() aggregates afterward)."""
    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)
    per_seed_csv = os.path.join(out_dir, f"driving_{CAND_NAME}_seed{seed}_eval.csv")
    per_seed_summary_csv = os.path.join(out_dir, f"driving_{CAND_NAME}_seed{seed}_summary.csv")

    if os.path.exists(per_seed_csv):
        print(f"Seed {seed} already done, skipping.")
        return

    from baselines.rule_based_ems import RuleBasedEMS
    rule_ctrl = RuleBasedEMS()
    df_rule, _ = evaluate_all_cycles(rule_ctrl, is_ppo=False, controller_name="RuleBasedEMS")
    rule_by_cycle = df_rule.set_index("cycle_id")["wh_per_km"].to_dict()

    print(f">>> Training {CAND_NAME} (seed {seed}, {TIMESTEPS} steps) <<<")
    eval_rows, summ = train_and_eval(seed)
    beat_count = sum(1 for r in eval_rows if r["wh_per_km"] <= rule_by_cycle.get(r["cycle_id"], float("inf")))
    summ["cycles_beating_rule_based"] = beat_count
    summ["cycles_total"] = len(eval_rows)

    pd.DataFrame(eval_rows).to_csv(per_seed_csv, index=False)
    pd.DataFrame([summ]).to_csv(per_seed_summary_csv, index=False)
    d_wh = summ["mean_wh_per_km"] - RULE_BASED_WH_PER_KM
    print(
        f"    Seed {seed}: Mean Wh/km={summ['mean_wh_per_km']:.2f} ({d_wh:+.2f} vs 129.16), "
        f"Regen={summ['mean_regen_recovery_pct']:.1f}%, "
        f"CyclesBeatingRuleBased={beat_count}/{len(eval_rows)}"
    )
    print(f"Saved: {per_seed_csv}\n       {per_seed_summary_csv}")


def combine_results():
    out_dir = "audit"
    benchmark_csv = os.path.join(out_dir, f"driving_{CAND_NAME}_benchmark.csv")
    summary_csv = os.path.join(out_dir, f"driving_{CAND_NAME}_summary.csv")
    eval_frames, summ_frames = [], []
    for seed in SEEDS:
        eval_frames.append(pd.read_csv(os.path.join(out_dir, f"driving_{CAND_NAME}_seed{seed}_eval.csv")))
        summ_frames.append(pd.read_csv(os.path.join(out_dir, f"driving_{CAND_NAME}_seed{seed}_summary.csv")))
    df_eval = pd.concat(eval_frames, ignore_index=True)
    df_summ = pd.concat(summ_frames, ignore_index=True)
    df_eval.to_csv(benchmark_csv, index=False)
    df_summ.to_csv(summary_csv, index=False)
    print(df_summ.to_string(index=False))
    print(f"\nCross-seed mean Wh/km: {df_summ['mean_wh_per_km'].mean():.2f}")
    print(f"Cross-seed mean regen: {df_summ['mean_regen_recovery_pct'].mean():.1f}%")
    print(f"Saved: {benchmark_csv}\n       {summary_csv}")


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
