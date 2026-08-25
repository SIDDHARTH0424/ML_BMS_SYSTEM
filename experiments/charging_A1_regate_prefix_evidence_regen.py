"""
ONE-OFF EVIDENCE-REGENERATION SCRIPT -- DO NOT RUN AGAINST CURRENT agents/train_ppo.py

This is a renamed copy of experiments/charging_A1_regate_via_build_agent.py,
used exactly once to regenerate the original pre-fix (decaying-LR,
FAILING) Candidate A1 evidence after those files were accidentally
overwritten by a post-fix re-run under the same filenames. To produce
that evidence, agents/train_ppo.py::build_agent() was TEMPORARILY
reverted to reintroduce linear_schedule(), this script was run for all
3 seeds (reproducing the original numbers exactly: 2577.60s / 2671.00s
/ 2368.93s), and build_agent() was then restored to the flat-LR fix.

If you run this script now, against the CURRENT (fixed) build_agent(),
it will NOT reproduce the pre-fix numbers -- it will produce the same
PASSING results as
experiments/charging_A1_regate_via_build_agent.py, just saved under
these "_prefix_"-labeled filenames, which would be actively misleading
(implying a FAIL result under a filename that no longer describes what
was run). Do not use this script for anything other than historical
reference. See audit/charging_A1_regate_via_build_agent_report.md and
audit/charging_A1_regate_flat_lr_fixed_report.md for the full story.

--- Original docstring below ---

Track A gate re-validation through the REAL production pipeline.

PURPOSE: audit/charging_stage4_collapse_investigation.md found that the
diagnostic which originally gated Candidate A1
(experiments/run_readiness_diagnostics.py) constructs PPO with a
CONSTANT learning rate, while the actual production long-training
entry point (agents/train_ppo.py::build_agent()) wraps it in
linear_schedule() -- a decaying schedule the diagnostic never tested.
This is option 2 from that investigation's recommendation (and the one
recommended as "scientifically cleaner" if the production LR schedule
is kept): re-run the SAME Candidate A1 gate, SAME evaluation logic,
SAME 3 seeds, SAME 50,000-step budget -- but building the model with
build_agent() (imported directly from agents/train_ppo.py, not
reimplemented) instead of a raw PPO(...) call. This is the only
variable that changes.

Evaluation logic (standard 15-scenario grid + 45C/50C stress grid) is
copied verbatim from
experiments/run_readiness_diagnostics.py::train_and_eval_charging_candidate
to guarantee the gate is evaluated identically to the original.

Usage:
    python -m experiments.charging_A1_regate_via_build_agent --seed=7
    python -m experiments.charging_A1_regate_via_build_agent --seed=21
    python -m experiments.charging_A1_regate_via_build_agent --seed=42
    python -m experiments.charging_A1_regate_via_build_agent --combine
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from agents.train_ppo import build_agent
from environment.battery_env import BatteryChargingEnv
from utils.config import load_config
from utils.seed import set_global_seed

SEEDS = [7, 21, 42]
TIMESTEPS = 50_000
CONFIG_DIR = "configs/final_charging"
BASELINE_TIME_S = 2094.6
GATE_THRESHOLD_S = BASELINE_TIME_S * 1.05  # 2199.3s
MAX_CURRENT_STRESS_PEAK_TEMP_C = 50.75  # from audit/long_training_freeze_HISTORICAL_B2.md Sec 1


def train_and_eval(seed: int):
    set_global_seed(seed)
    battery_cfg = load_config("battery", config_dir=CONFIG_DIR)
    safety_cfg = load_config("safety", config_dir=CONFIG_DIR)
    reward_cfg = load_config("reward", config_dir=CONFIG_DIR)
    sim_cfg = load_config("simulation", config_dir=CONFIG_DIR)
    ppo_cfg = copy.deepcopy(load_config("ppo", config_dir=CONFIG_DIR))
    ppo_cfg["seed"] = seed  # build_agent() reads seed from ppo_cfg, not a separate arg

    raw_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env = Monitor(raw_env)

    tb_dir = os.path.join("runs", "charging_A1_regate_prefix", f"seed_{seed}", "tensorboard")
    model = build_agent(env, ppo_cfg, tensorboard_dir=tb_dir)

    print(f">>> Training Candidate A1 via REAL build_agent() pipeline, seed={seed}, {TIMESTEPS} steps <<<")
    model.learn(total_timesteps=TIMESTEPS)

    # --- Evaluation logic copied verbatim from run_readiness_diagnostics.py ---
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
            info = {}
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

    std_summary = {
        "candidate": "A1_via_build_agent",
        "seed": seed,
        "reached_target_all": all(reached_list),
        "reached_fraction": float(np.mean(reached_list)),
        "mean_charging_time_s": float(np.mean(times)),
        "max_temperature_max_c": float(np.max(peak_temps)),
        "mean_requested_current_a": float(np.mean(mean_req_is)),
        "mean_applied_current_a": float(np.mean(mean_appl_is)),
        "cumulative_q_gen_j": float(np.mean(q_gens)),
    }

    stress_rows = []
    for s0 in [0.10, 0.20, 0.30]:
        for t0 in [45.0, 50.0]:
            obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
            temps, reqs, apps = [eval_env._state.temperature_c], [], []
            tot_q = 0.0
            term = trunc = False
            steps = 0
            info = {}
            while not (term or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(act)
                reqs.append(info["requested_current"])
                apps.append(info["applied_current"])
                temps.append(eval_env._state.temperature_c)
                tot_q += info["q_gen"] * dt
                steps += 1
            stress_rows.append({
                "candidate": "A1_via_build_agent",
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

    ckpt_dir = os.path.join("runs", "charging_A1_regate_prefix", f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "trained_model.zip"))

    return std_summary, stress_rows


def run_single_seed(seed: int):
    std_path = os.path.join("audit", f"charging_A1_regate_prefix_seed{seed}_standard.csv")
    stress_path = os.path.join("audit", f"charging_A1_regate_prefix_seed{seed}_stress.csv")
    if os.path.exists(std_path):
        print(f"Seed {seed} already done, skipping.")
        return
    std_summary, stress_rows = train_and_eval(seed)
    os.makedirs("audit", exist_ok=True)
    pd.DataFrame([std_summary]).to_csv(std_path, index=False)
    pd.DataFrame(stress_rows).to_csv(stress_path, index=False)
    delta_pct = (std_summary["mean_charging_time_s"] - BASELINE_TIME_S) / BASELINE_TIME_S * 100
    print(f"\n  Seed {seed}: {std_summary['mean_charging_time_s']:.2f}s ({delta_pct:+.2f}% vs baseline, "
          f"gate={GATE_THRESHOLD_S:.1f}s), reached={std_summary['reached_fraction']*15:.0f}/15, "
          f"peak_temp={std_summary['max_temperature_max_c']:.2f}C")
    print(f"  Saved: {std_path}\n         {stress_path}")


def combine_and_gate():
    std_frames, stress_frames = [], []
    for seed in SEEDS:
        std_frames.append(pd.read_csv(os.path.join("audit", f"charging_A1_regate_prefix_seed{seed}_standard.csv")))
        stress_frames.append(pd.read_csv(os.path.join("audit", f"charging_A1_regate_prefix_seed{seed}_stress.csv")))
    df_std = pd.concat(std_frames, ignore_index=True)
    df_stress = pd.concat(stress_frames, ignore_index=True)
    df_std.to_csv("audit/charging_A1_regate_prefix_standard_combined.csv", index=False)
    df_stress.to_csv("audit/charging_A1_regate_prefix_stress_combined.csv", index=False)

    print("\n=== STANDARD GRID RESULTS (via real build_agent() pipeline) ===")
    print(df_std.to_string(index=False))

    print("\n=== GATE EVALUATION ===")
    all_pass = True
    # A1: 3/3 seeds reach 95% SoC
    a1 = bool(df_std["reached_target_all"].all())
    print(f"A1 (3/3 seeds reach target, standard grid): {'PASS' if a1 else 'FAIL'}")
    all_pass &= a1

    # A2: every seed <= gate threshold
    a2_rows = df_std[["seed", "mean_charging_time_s"]].copy()
    a2_rows["pass"] = a2_rows["mean_charging_time_s"] <= GATE_THRESHOLD_S
    a2 = bool(a2_rows["pass"].all())
    print(f"A2 (every seed <= {GATE_THRESHOLD_S:.1f}s): {'PASS' if a2 else 'FAIL'}")
    print(a2_rows.to_string(index=False))
    all_pass &= a2

    # A3: measurable stress-dependent derating (normal current > stress current)
    normal_current = df_std["mean_applied_current_a"].mean()
    stress_45 = df_stress[df_stress["ambient_temp_c"] == 45.0]["mean_applied_current_a"].mean()
    stress_50 = df_stress[df_stress["ambient_temp_c"] == 50.0]["mean_applied_current_a"].mean()
    a3 = normal_current > stress_45 > stress_50
    print(f"A3 (derating: normal {normal_current:.1f}A > 45C {stress_45:.1f}A > 50C {stress_50:.1f}A): "
          f"{'PASS' if a3 else 'FAIL'}")
    all_pass &= a3

    # A4: >=2/3 seeds lower peak temp than Max Current (50.75C) at matched 45C stress
    stress_45_by_seed = df_stress[df_stress["ambient_temp_c"] == 45.0].groupby("seed")["peak_temperature_c"].max()
    a4_pass_count = int((stress_45_by_seed < MAX_CURRENT_STRESS_PEAK_TEMP_C).sum())
    a4 = a4_pass_count >= 2
    print(f"A4 (>=2/3 seeds peak temp < {MAX_CURRENT_STRESS_PEAK_TEMP_C}C at 45C stress): "
          f"{'PASS' if a4 else 'FAIL'} ({a4_pass_count}/3)")
    print(stress_45_by_seed.to_string())
    all_pass &= a4

    # A5: normal-temp behavior stays aggressive (mean current not far below 160A ceiling)
    a5 = normal_current >= 140.0  # same qualitative bar as the original A1 report (~155-158A)
    print(f"A5 (normal-temp mean current >= 140A, ceiling=160A): {'PASS' if a5 else 'FAIL'} ({normal_current:.1f}A)")
    all_pass &= a5

    # A6: no catastrophic standard-grid failure
    a6 = bool((df_std["reached_fraction"] >= 0.9).all())
    print(f"A6 (no catastrophic standard-grid failure, all seeds >=90% reached): {'PASS' if a6 else 'FAIL'}")
    all_pass &= a6

    print(f"\n{'ALL GATE CONDITIONS PASS' if all_pass else 'GATE FAILED -- see above'}")
    print("(A7 training-stability requires curve inspection, not covered by this script's summary stats)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--combine":
        combine_and_gate()
    elif len(sys.argv) > 1 and sys.argv[1].startswith("--seed="):
        run_single_seed(int(sys.argv[1].split("=")[1]))
    else:
        for seed in SEEDS:
            run_single_seed(seed)
        combine_and_gate()
