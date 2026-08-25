"""
Track B: Entropy Isolation & Action Distribution Analysis
=========================================================
1. Inspects existing Stage-Q models (seeds 7, 21, 42) at braking states:
   - Final policy std
   - P(action > 0 | braking)
   - P(action >= 0.5 | braking)
   - Mean action | braking
2. Executes Entropy-Only Isolation (ent_coef = 0.005, reward weights UNCHANGED)
   across seeds [7, 21, 42] for 50,000 steps.
3. Compares braking action distributions, regen recovery %, and Wh/km.

Saves:
  - audit/driving_entropy_isolation.csv
  - audit/driving_entropy_isolation.md
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
from environment.ev_energy_env import EVEnergyEnv
from training.evaluate_drive_ems import STANDARD_CYCLES, evaluate_all_cycles
from training.train_drive_ems import make_drive_ems_env
from utils.config import load_config
from utils.seed import set_global_seed


SEEDS = [7, 21, 42]
TIMESTEPS = 50000
TRAINING_CYCLE = STANDARD_CYCLES["wltp_class3b"]


def analyze_model_at_braking_states(model: PPO, drive_cycle_path: str = TRAINING_CYCLE) -> Dict:
    """Evaluates the model's action distribution specifically during braking opportunity steps."""
    env = make_drive_ems_env(drive_cycle_path=drive_cycle_path, mode="eval")
    obs, _ = env.reset(options={"initial_soc": 0.50, "ambient_temp_c": 25.0})

    actions = []
    braking_actions = []
    propulsion_actions = []

    terminated = truncated = False
    while not (terminated or truncated):
        # Sample stochastic action from policy distribution
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
            dist = model.policy.get_distribution(obs_tensor)
            act = dist.distribution.sample().cpu().numpy().flatten()[0]

        speed = env._drive_cycle.current_speed()
        accel = env._drive_cycle.current_acceleration()
        grade = env._drive_cycle.current_grade()
        forces = env.vehicle.compute(speed_mps=speed, acceleration_mps2=accel, road_grade_rad=grade)

        actions.append(act)
        if forces.p_wheel < 0.0:
            braking_actions.append(act)
        else:
            propulsion_actions.append(act)

        obs, reward, terminated, truncated, info = env.step(np.array([act], dtype=np.float32))

    br_acts = np.array(braking_actions)
    prop_acts = np.array(propulsion_actions)

    with torch.no_grad():
        if hasattr(model.policy, "log_std"):
            policy_std = float(model.policy.log_std.exp().mean().item())
        else:
            policy_std = float("nan")

    return {
        "policy_std": policy_std,
        "n_braking_steps": len(br_acts),
        "mean_action_braking": float(np.mean(br_acts)) if len(br_acts) > 0 else float("nan"),
        "std_action_braking": float(np.std(br_acts)) if len(br_acts) > 0 else float("nan"),
        "p_action_gt_0_braking": float(np.mean(br_acts > 0.0)) if len(br_acts) > 0 else float("nan"),
        "p_action_ge_05_braking": float(np.mean(br_acts >= 0.5)) if len(br_acts) > 0 else float("nan"),
        "mean_action_propulsion": float(np.mean(prop_acts)) if len(prop_acts) > 0 else float("nan"),
        "p_action_lt_0_propulsion": float(np.mean(prop_acts < 0.0)) if len(prop_acts) > 0 else float("nan"),
    }


def train_driving_ppo_with_entropy(seed: int, ent_coef: float, timesteps: int = TIMESTEPS) -> Tuple[PPO, Dict]:
    set_global_seed(seed)
    ppo_cfg = load_config("ppo_drive_ems")
    raw_env = make_drive_ems_env(drive_cycle_path=TRAINING_CYCLE, mode="train")
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

    model.learn(total_timesteps=timesteps)
    braking_stats = analyze_model_at_braking_states(model)
    return model, braking_stats


def main():
    out_dir = "audit"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("TRACK B: ENTROPY A/B ISOLATION EXPERIMENT")
    print("=" * 80)

    # 1. Evaluate existing Stage-Q models (Baseline: ent_coef = 0.0)
    print("\n>>> Analyzing Existing Stage-Q Models (ent_coef = 0.0) <<<")
    baseline_stats = []
    for s in SEEDS:
        ckpt_path = os.path.join("runs", "driving_ppo_stageQ", f"seed_{s}", "trained_model.zip")
        if os.path.exists(ckpt_path):
            m = PPO.load(ckpt_path)
            st = analyze_model_at_braking_states(m)
            st.update({"experiment": "StageQ_Baseline_ent0", "seed": s, "ent_coef": 0.0})
            baseline_stats.append(st)
            print(f"  StageQ Seed {s}: policy_std={st['policy_std']:.3f}, "
                  f"P(a>0|braking)={st['p_action_gt_0_braking']*100:.1f}%, "
                  f"P(a>=0.5|braking)={st['p_action_ge_05_braking']*100:.1f}%, "
                  f"mean_action_braking={st['mean_action_braking']:+.3f}")

    # 2. Train Entropy-Only Isolation models (ent_coef = 0.005, reward unchanged)
    print("\n>>> Training Entropy-Only Models (ent_coef = 0.005, reward weights UNCHANGED) <<<")
    entropy_stats = []
    entropy_eval_rows = []

    for s in SEEDS:
        print(f"  Training Seed {s} (ent_coef = 0.005)...")
        m, st = train_driving_ppo_with_entropy(seed=s, ent_coef=0.005, timesteps=TIMESTEPS)
        st.update({"experiment": "Entropy_Isolated_ent0005", "seed": s, "ent_coef": 0.005})
        entropy_stats.append(st)
        print(f"  Entropy Seed {s}: policy_std={st['policy_std']:.3f}, "
              f"P(a>0|braking)={st['p_action_gt_0_braking']*100:.1f}%, "
              f"P(a>=0.5|braking)={st['p_action_ge_05_braking']*100:.1f}%, "
              f"mean_action_braking={st['mean_action_braking']:+.3f}")

        # Evaluate across 4 standard cycles
        df_summary, _ = evaluate_all_cycles(m, is_ppo=True, controller_name=f"PPO_ent005_seed{s}")
        df_summary["seed"] = s
        df_summary["experiment"] = "Entropy_Isolated_ent0005"
        entropy_eval_rows.extend(df_summary.to_dict("records"))

    df_all_stats = pd.DataFrame(baseline_stats + entropy_stats)
    df_eval = pd.DataFrame(entropy_eval_rows)

    csv_path = os.path.join(out_dir, "driving_entropy_isolation.csv")
    df_all_stats.to_csv(csv_path, index=False)

    md_path = os.path.join(out_dir, "driving_entropy_isolation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Track B: Entropy Isolation & Action Distribution Analysis\n\n")
        f.write("**Objective**: Isolate whether non-zero policy entropy (`ent_coef = 0.005`) alone resolves the regen discovery problem without modifying reward weights.\n\n")
        f.write("---\n\n")
        f.write("## 1. Action Distribution at Braking Opportunity States\n\n")
        f.write("| Experiment | Seed | ent_coef | Policy Std | $P(a > 0 \\mid \\text{braking})$ | $P(a \\ge 0.5 \\mid \\text{braking})$ | Mean Action (Braking) | Mean Action (Propulsion) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in df_all_stats.iterrows():
            f.write(f"| {r['experiment']} | {r['seed']} | {r['ent_coef']} | {r['policy_std']:.4f} | **{r['p_action_gt_0_braking']*100:.1f}%** | {r['p_action_ge_05_braking']*100:.1f}% | {r['mean_action_braking']:+.3f} | {r['mean_action_propulsion']:+.3f} |\n")

        f.write("\n---\n\n")
        f.write("## 2. Multi-Cycle Benchmark with Entropy Isolation (`ent_coef = 0.005`)\n\n")
        f.write("| Seed | UDDS (Wh/km) | HWFET (Wh/km) | US06 (Wh/km) | WLTP 3b (Wh/km) | Mean Wh/km | Regen Recovery (%) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in SEEDS:
            sub = df_eval[df_eval["seed"] == s]
            u = sub[sub["cycle_id"] == "epa_udds"]["wh_per_km"].values[0]
            h = sub[sub["cycle_id"] == "epa_hwfet"]["wh_per_km"].values[0]
            us = sub[sub["cycle_id"] == "epa_us06"]["wh_per_km"].values[0]
            w = sub[sub["cycle_id"] == "wltp_class3b"]["wh_per_km"].values[0]
            m_wh = float(np.mean([u, h, us, w]))
            reg = float(np.mean(sub["regen_recovery_fraction"])) * 100.0
            f.write(f"| Seed {s} | {u:.2f} | {h:.2f} | {us:.2f} | {w:.2f} | **{m_wh:.2f}** | **{reg:.1f}%** |\n")

        f.write(f"\n**Rule-Based EMS Cross-Cycle Baseline**: **129.16 Wh/km** (100% regen)\n")

    print(f"\nArtifacts generated:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
