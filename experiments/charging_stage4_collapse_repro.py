"""
Root-cause investigation, part 2: does the real build_agent()/
linear_schedule Stage-4 pipeline (1,000,000 steps, seed 7,
configs/final_charging) actually collapse to near-zero current /
0% target-reached, as reported? And if so, at what point?

Trains via build_agent() (the REAL production model-construction path,
imported directly -- not reimplemented) in resumable chunks, using a
callback that stops training at each checkpoint target while still
passing total_timesteps=1_000_000 to every learn() call so the
linear_schedule's progress_remaining is computed against the FULL
intended 1M-step horizon at every chunk (not reset per-chunk) --
this exactly reproduces what a single uninterrupted
model.learn(total_timesteps=1_000_000) call would do.

Evaluates the standard 15-scenario grid at each checkpoint and appends
to audit/charging_stage4_repro_checkpoints.csv.

Usage (run repeatedly; each call advances to the next un-reached target):
    python -m experiments.charging_stage4_collapse_repro
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from agents.train_ppo import build_agent
from environment.battery_env import BatteryChargingEnv
from utils.config import load_config
from utils.seed import set_global_seed

SEED = 7
FINAL_TARGET = 1_000_000
CHECKPOINT_TARGETS = [100_000, 200_000, 400_000, 600_000, 800_000, 1_000_000]
CONFIG_DIR = "configs/final_charging"
MODEL_PATH = "runs/charging_stage4_repro/model.zip"
CSV_PATH = "audit/charging_stage4_repro_checkpoints.csv"


class StopAtTimestep(BaseCallback):
    """Stops training once num_timesteps reaches `target`, without altering
    total_timesteps passed to learn() -- so the linear_schedule's
    progress_remaining stays anchored to the full 1,000,000-step horizon
    across resumed chunks, exactly matching a single uninterrupted call."""

    def __init__(self, target: int):
        super().__init__()
        self.target = target

    def _on_step(self) -> bool:
        return self.num_timesteps < self.target


def evaluate_standard_grid(model, eval_env, sim_cfg):
    dt = eval_env.dt
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    times, peak_temps, reached_list, mean_currents = [], [], [], []
    for s0 in soc_grid:
        for t0 in temp_grid:
            obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
            temps = [eval_env._state.temperature_c]
            currents = []
            term = trunc = False
            steps = 0
            info = {}
            while not (term or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(act)
                temps.append(eval_env._state.temperature_c)
                currents.append(info.get("applied_current", 0.0))
                steps += 1
            times.append(steps * dt)
            peak_temps.append(max(temps))
            mean_currents.append(float(np.mean(currents)) if currents else 0.0)
            reached_list.append(bool(info.get("target_reached")))
    return {
        "mean_charging_time_s": float(np.mean(times)),
        "reached_fraction": float(np.mean(reached_list)),
        "max_peak_temp_c": float(np.max(peak_temps)),
        "mean_applied_current_a": float(np.mean(mean_currents)),
    }


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs("audit", exist_ok=True)

    battery_cfg = load_config("battery", config_dir=CONFIG_DIR)
    safety_cfg = load_config("safety", config_dir=CONFIG_DIR)
    reward_cfg = load_config("reward", config_dir=CONFIG_DIR)
    sim_cfg = load_config("simulation", config_dir=CONFIG_DIR)
    ppo_cfg = load_config("ppo", config_dir=CONFIG_DIR)

    raw_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env = Monitor(raw_env)

    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, env=env)
        print(f"Resumed from {MODEL_PATH} at num_timesteps={model.num_timesteps}")
    else:
        set_global_seed(SEED)
        model = build_agent(env, ppo_cfg, tensorboard_dir="runs/charging_stage4_repro/tensorboard")
        print(f"Fresh model built via build_agent() (real production path), seed={SEED}")

    existing_rows = []
    if os.path.exists(CSV_PATH):
        existing_rows = pd.read_csv(CSV_PATH).to_dict("records")
    reached_targets = {r["target_timesteps"] for r in existing_rows}

    eval_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)

    for target in CHECKPOINT_TARGETS:
        if target in reached_targets:
            print(f"Checkpoint {target} already evaluated, skipping.")
            continue
        if model.num_timesteps >= target:
            print(f"Model already past {target} ({model.num_timesteps}), evaluating without further training.")
        else:
            print(f"\n>>> Training from {model.num_timesteps} to {target} "
                  f"(total_timesteps={FINAL_TARGET} for schedule purposes) <<<")
            model.learn(
                total_timesteps=FINAL_TARGET,
                callback=StopAtTimestep(target),
                reset_num_timesteps=False,
            )
        model.save(MODEL_PATH)

        result = evaluate_standard_grid(model, eval_env, sim_cfg)
        result["target_timesteps"] = target
        result["actual_num_timesteps"] = model.num_timesteps
        current_lr = model.learning_rate(1.0 - model.num_timesteps / FINAL_TARGET) if callable(model.learning_rate) else model.learning_rate
        result["effective_lr_at_checkpoint"] = float(current_lr)
        print(f"  Checkpoint {target}: {result}")

        existing_rows.append(result)
        pd.DataFrame(existing_rows).to_csv(CSV_PATH, index=False)
        print(f"  Saved to {CSV_PATH}")

        # Stop after ONE checkpoint per invocation to keep each run within
        # sandbox time limits; rerun this script to advance to the next.
        break

    if len(existing_rows) == len(CHECKPOINT_TARGETS):
        print("\nAll checkpoints evaluated.")
        print(pd.DataFrame(existing_rows).to_string(index=False))


if __name__ == "__main__":
    main()
