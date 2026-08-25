"""
Root-cause investigation: does agents/train_ppo.py::build_agent()'s
linear_schedule(learning_rate) -- present in the REAL long-training
pipeline but ABSENT from experiments/run_readiness_diagnostics.py's
flat-LR diagnostic that actually gated Candidate A1 -- explain the
reported staged-pipeline degradation (Stage 3 @ 50k steps: 2577.6s,
vs the gated diagnostic's 2114.5s at the same 50k/seed=7 budget)?

This uses build_agent() itself (imported, not reimplemented) so there
is no possibility of a third, subtly-different code path being
introduced by this investigation.

Usage:
    python -m experiments.charging_lr_schedule_repro
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.monitor import Monitor

from agents.train_ppo import build_agent
from environment.battery_env import BatteryChargingEnv
from utils.config import load_config
from utils.seed import set_global_seed

SEED = 7
TIMESTEPS = 50_000  # Stage 3's budget, same as the gated Candidate A1 seed-7 run
CONFIG_DIR = "configs/final_charging"


def evaluate_standard_grid(model, eval_env, sim_cfg):
    dt = eval_env.dt
    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    times, peak_temps, reached_list = [], [], []
    for s0 in soc_grid:
        for t0 in temp_grid:
            obs, _ = eval_env.reset(options={"initial_soc": s0, "ambient_temp_c": t0})
            temps = [eval_env._state.temperature_c]
            term = trunc = False
            steps = 0
            info = {}
            while not (term or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(act)
                temps.append(eval_env._state.temperature_c)
                steps += 1
            times.append(steps * dt)
            peak_temps.append(max(temps))
            reached_list.append(bool(info.get("target_reached")))
    return {
        "mean_charging_time_s": float(np.mean(times)),
        "reached_target_all": all(reached_list),
        "reached_fraction": float(np.mean(reached_list)),
        "max_peak_temp_c": float(np.max(peak_temps)),
    }


def main():
    set_global_seed(SEED)
    battery_cfg = load_config("battery", config_dir=CONFIG_DIR)
    safety_cfg = load_config("safety", config_dir=CONFIG_DIR)
    reward_cfg = load_config("reward", config_dir=CONFIG_DIR)
    sim_cfg = load_config("simulation", config_dir=CONFIG_DIR)
    ppo_cfg = load_config("ppo", config_dir=CONFIG_DIR)

    raw_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env = Monitor(raw_env)

    print(f"learning_rate in ppo_cfg: {ppo_cfg['learning_rate']} (type: {type(ppo_cfg['learning_rate'])})")
    model = build_agent(env, ppo_cfg, tensorboard_dir="/tmp/lr_repro_tb")
    print(f"model.learning_rate after build_agent(): {model.learning_rate}")
    print(f"    -> at progress_remaining=1.0 (start): {model.learning_rate(1.0) if callable(model.learning_rate) else model.learning_rate}")
    print(f"    -> at progress_remaining=0.5 (mid):   {model.learning_rate(0.5) if callable(model.learning_rate) else model.learning_rate}")
    print(f"    -> at progress_remaining=0.01 (end):  {model.learning_rate(0.01) if callable(model.learning_rate) else model.learning_rate}")

    print(f"\n>>> Training via REAL build_agent()/linear_schedule pipeline, seed={SEED}, {TIMESTEPS} steps <<<")
    model.learn(total_timesteps=TIMESTEPS)

    eval_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval", enforce_safety=True)
    result = evaluate_standard_grid(model, eval_env, sim_cfg)

    print("\n=== RESULT (build_agent()/linear_schedule pipeline, 50k steps, seed 7) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== COMPARISON ===")
    print(f"  Gated Candidate A1 diagnostic (flat LR, 50k, seed 7): 2114.53s, 15/15 reached")
    print(f"  This run (linear_schedule LR, 50k, seed 7):           {result['mean_charging_time_s']:.2f}s, "
          f"{result['reached_fraction']*15:.0f}/15 reached")
    delta_pct = (result['mean_charging_time_s'] - 2114.53) / 2114.53 * 100
    print(f"  Delta: {delta_pct:+.1f}%")


if __name__ == "__main__":
    main()
