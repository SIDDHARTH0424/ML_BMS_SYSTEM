"""
Root-cause investigation, part 3: single continuous (properly seeded,
fully deterministic within this one process) run of the REAL
build_agent()/linear_schedule pipeline from 0 to 200,000 steps -- the
exact window where charging_stage4_collapse_repro.py observed collapse
(100k: 2094.8s/100% reached -> 200k: 7200s/0% reached) -- with SB3's
CSV logger attached from step 0, to get full per-chunk
train/approx_kl, train/explained_variance, train/value_loss,
train/entropy_loss, train/std, rollout/ep_rew_mean, rollout/ep_len_mean
curves through the transition.

total_timesteps=1_000_000 is still passed to learn() (via the
StopAtTimestep callback) so linear_schedule's progress_remaining
matches what Stage 4 would actually see at these timesteps -- this is
NOT a fresh 200k-step schedule, it's the first 200k steps of the real
1M-step schedule.

Usage:
    python -m experiments.charging_stage4_collapse_diagnostic
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_sb3_logger
from stable_baselines3.common.monitor import Monitor

from agents.train_ppo import build_agent
from environment.battery_env import BatteryChargingEnv
from utils.config import load_config
from utils.seed import set_global_seed

SEED = 7
FINAL_TARGET = 1_000_000
STOP_AT = 500_000
CONFIG_DIR = "configs/final_charging"
LOG_DIR = "runs/charging_stage4_collapse_diagnostic"
CURVE_CSV = "audit/charging_stage4_collapse_diagnostic_curves.csv"
EVAL_CSV = "audit/charging_stage4_collapse_diagnostic_eval_points.csv"

EVAL_EVERY = 20_000  # evaluate standard grid periodically through the window


class EvalAndStopCallback(BaseCallback):
    def __init__(self, stop_at: int, eval_every: int, eval_fn):
        super().__init__()
        self.stop_at = stop_at
        self.eval_every = eval_every
        self.eval_fn = eval_fn
        self._next_eval = eval_every
        self.eval_rows = []

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            row = self.eval_fn()
            row["at_timesteps"] = self.num_timesteps
            self.eval_rows.append(row)
            print(f"  [eval @ {self.num_timesteps}] {row}")
            self._next_eval += self.eval_every
        return self.num_timesteps < self.stop_at


def evaluate_standard_grid(model, eval_env, sim_cfg):
    dt = eval_env.dt
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    times, reached_list, mean_currents = [], [], []
    for s0 in soc_grid:
        for t0 in temp_grid:
            obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
            currents = []
            term = trunc = False
            steps = 0
            info = {}
            while not (term or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(act)
                currents.append(info.get("applied_current", 0.0))
                steps += 1
            times.append(steps * dt)
            mean_currents.append(float(np.mean(currents)) if currents else 0.0)
            reached_list.append(bool(info.get("target_reached")))
    return {
        "mean_charging_time_s": float(np.mean(times)),
        "reached_fraction": float(np.mean(reached_list)),
        "mean_applied_current_a": float(np.mean(mean_currents)),
    }


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("audit", exist_ok=True)

    set_global_seed(SEED)
    battery_cfg = load_config("battery", config_dir=CONFIG_DIR)
    safety_cfg = load_config("safety", config_dir=CONFIG_DIR)
    reward_cfg = load_config("reward", config_dir=CONFIG_DIR)
    sim_cfg = load_config("simulation", config_dir=CONFIG_DIR)
    ppo_cfg = load_config("ppo", config_dir=CONFIG_DIR)

    raw_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env = Monitor(raw_env)
    eval_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)

    model = build_agent(env, ppo_cfg, tensorboard_dir=os.path.join(LOG_DIR, "tensorboard"))
    sb3_logger = configure_sb3_logger(LOG_DIR, ["csv"])
    model.set_logger(sb3_logger)

    # Curves-only pass: mid-run evaluation (10x full 15-scenario grid) was
    # far more expensive than training itself and timed out the sandbox.
    # The 100k/200k eval numbers are already known from
    # charging_stage4_collapse_repro.py's chunked run; this pass exists
    # purely to capture the per-chunk train/* curves through that window.
    class SimpleStop(BaseCallback):
        def __init__(self, stop_at):
            super().__init__()
            self.stop_at = stop_at

        def _on_step(self) -> bool:
            return self.num_timesteps < self.stop_at

    print(f">>> Single continuous run, seed={SEED}, 0 -> {STOP_AT} steps "
          f"(schedule horizon: {FINAL_TARGET}) <<<")
    model.learn(total_timesteps=FINAL_TARGET, callback=SimpleStop(STOP_AT), log_interval=1)

    progress_path = os.path.join(LOG_DIR, "progress.csv")
    df_curves = pd.read_csv(progress_path)
    df_curves.to_csv(CURVE_CSV, index=False)
    print(f"Saved {len(df_curves)} chunk rows to {CURVE_CSV}")


if __name__ == "__main__":
    main()
