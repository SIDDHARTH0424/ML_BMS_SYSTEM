"""
Evaluation framework: runs PPO and all baseline controllers through the
identical fixed evaluation grid (SoC x ambient temp), computes the shared
metric set, and produces comparison tables + plots.

Usage:
    python -m training.evaluate --model runs/run_004/trained_model.zip --run-name run_004
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines.adaptive import AdaptiveController
from baselines.cc import ConstantCurrentController, MaxCurrentController
from baselines.cccv import CCCVController
from environment.battery_env import BatteryChargingEnv
from environment.ecm_model import BatteryECM
from safety.safety_layer import safety_layer
from utils.config import load_config
from utils.metrics import summarize_episode
from utils.plotting import plot_comparison_bar, plot_profile


def _run_baseline_episode(controller, ecm: BatteryECM, safety_cfg: Dict, reward_cfg: Dict,
                           sim_cfg: Dict, initial_soc: float, ambient_temp: float) -> Dict:
    """Run one baseline controller through one scenario, logging the same
    per-step arrays the PPO path logs, so metrics.summarize_episode applies uniformly."""
    dt = float(sim_cfg.get("dt_seconds", ecm.dt))
    max_steps = int(sim_cfg["max_episode_steps"])
    target_soc = float(sim_cfg["target_soc"])
    v_max = ecm.v_max
    t_max = ecm.t_max_c

    state = ecm.reset_state(initial_soc=initial_soc, ambient_temp_c=ambient_temp)
    controller.reset()
    prev_current = 0.0

    log = {"soc": [], "temperature_c": [], "current_a": [], "voltage_v": [],
           "safety_intervention": [], "input_energy_wh": [], "stored_energy_wh": []}

    for _ in range(max_steps):
        v = ecm.terminal_voltage(state, prev_current)
        obs = {"soc": state.soc, "terminal_voltage": v, "temperature_c": state.temperature_c,
               "previous_current_a": prev_current, "ambient_temp_c": ambient_temp}
        requested = controller.act(obs)
        # Safety ceiling estimate uses i_max (worst case), NOT the actual
        # request or prev_current — matches battery_env.py's fix for the
        # voltage-taper circularity (see that file for the full explanation).
        # This keeps baselines evaluated under the identical safety
        # semantics PPO trains under.
        v_for_ceiling = ecm.terminal_voltage(state, ecm.i_max_a)
        applied, safety_info = safety_layer(requested, state, safety_cfg, estimated_voltage=v_for_ceiling)

        state = ecm.step(state, applied, ambient_temp)
        terminal_v = ecm.terminal_voltage(state, applied)

        log["soc"].append(state.soc)
        log["temperature_c"].append(state.temperature_c)
        log["current_a"].append(applied)
        log["voltage_v"].append(terminal_v)
        log["safety_intervention"].append(safety_info.intervened)
        # v3 fix: input_energy_wh = charger's true input power at the
        # terminals; stored_energy_wh = the OCV-referenced portion that
        # actually raises stored energy. No term is counted twice (see
        # utils/metrics.py energy_efficiency docstring for the bug this fixes).
        log["input_energy_wh"].append(applied * terminal_v * dt / 3600.0)
        log["stored_energy_wh"].append(applied * ecm.ocv(state.soc) * dt / 3600.0)

        prev_current = applied

        if state.soc >= target_soc or terminal_v >= v_max or state.temperature_c >= t_max:
            break

    return log


def _run_ppo_episode(model, env: BatteryChargingEnv, initial_soc: float, ambient_temp: float) -> Dict:
    obs, _ = env.reset(options={"initial_soc": initial_soc, "ambient_temp_c": ambient_temp})
    dt = env.dt
    log = {"soc": [], "temperature_c": [], "current_a": [], "voltage_v": [],
           "safety_intervention": [], "input_energy_wh": [], "stored_energy_wh": []}

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        v = info["terminal_voltage"]
        applied = info["applied_current_a"]
        log["soc"].append(env._state.soc)
        log["temperature_c"].append(env._state.temperature_c)
        log["current_a"].append(applied)
        log["voltage_v"].append(v)
        log["safety_intervention"].append(info["safety_intervention"]["type"] != "none")
        log["input_energy_wh"].append(applied * v * dt / 3600.0)
        log["stored_energy_wh"].append(applied * env.ecm.ocv(env._state.soc) * dt / 3600.0)

    return log


def run_evaluation(model_path: str, run_name: str):
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    eval_cfg = load_config("evaluation")

    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]

    ecm = BatteryECM(battery_cfg)
    dt = float(sim_cfg.get("dt_seconds", ecm.dt))

    controllers = {
        "cc": ConstantCurrentController(eval_cfg["cc"]),
        "cccv": CCCVController(eval_cfg["cccv"]),
        "adaptive": AdaptiveController(eval_cfg["adaptive"]),
        "max_current": MaxCurrentController(battery_cfg),
    }

    ppo_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval",
                                  enforce_safety=True)
    # Ablation: same trained policy, safety layer left in monitoring-only mode
    # (interventions are still logged; the episode still hard-terminates on
    # overvoltage/overtemperature) so we can quantify what the safety layer
    # is actually contributing rather than simulating an unbounded battery.
    ppo_env_no_safety = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval",
                                            enforce_safety=False)
    model = PPO.load(model_path)

    out_dir = os.path.join("runs", run_name, "evaluation")
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    def _save_trajectory(controller_name: str, soc0: float, temp0: float, log: Dict) -> None:
        """Full per-step state trajectory (SoC, voltage, temperature, current) for one episode."""
        pd.DataFrame({
            "step": range(len(log["soc"])),
            "time_s": [i * dt for i in range(len(log["soc"]))],
            "soc": log["soc"],
            "voltage_v": log["voltage_v"],
            "temperature_c": log["temperature_c"],
            "current_a": log["current_a"],
            "safety_intervention": log["safety_intervention"],
        }).to_csv(
            os.path.join(traj_dir, f"{controller_name}_soc{soc0:.2f}_temp{temp0:.0f}.csv"),
            index=False,
        )

    all_rows: List[Dict] = []
    profiles: Dict[str, Dict] = {}  # keyed by controller -> one representative scenario's log

    for soc in soc_grid:
        for temp in temp_grid:
            for name, controller in controllers.items():
                log = _run_baseline_episode(controller, ecm, safety_cfg, reward_cfg, sim_cfg, soc, temp)
                metrics = summarize_episode(log, dt, target_soc=sim_cfg["target_soc"], initial_soc=soc)
                metrics.update({"controller": name, "initial_soc": soc, "ambient_temp_c": temp})
                all_rows.append(metrics)
                _save_trajectory(name, soc, temp, log)
                if soc == soc_grid[0] and temp == temp_grid[0]:
                    profiles[name] = log

            log = _run_ppo_episode(model, ppo_env, soc, temp)
            metrics = summarize_episode(log, dt, target_soc=sim_cfg["target_soc"], initial_soc=soc)
            metrics.update({"controller": "ppo", "initial_soc": soc, "ambient_temp_c": temp})
            all_rows.append(metrics)
            _save_trajectory("ppo", soc, temp, log)
            if soc == soc_grid[0] and temp == temp_grid[0]:
                profiles["ppo"] = log

            # Ablation: PPO with the safety layer in monitoring-only mode
            log_no_safety = _run_ppo_episode(model, ppo_env_no_safety, soc, temp)
            metrics_no_safety = summarize_episode(log_no_safety, dt, target_soc=sim_cfg["target_soc"], initial_soc=soc)
            metrics_no_safety.update({"controller": "ppo_no_safety", "initial_soc": soc, "ambient_temp_c": temp})
            all_rows.append(metrics_no_safety)
            _save_trajectory("ppo_no_safety", soc, temp, log_no_safety)
            if soc == soc_grid[0] and temp == temp_grid[0]:
                profiles["ppo_no_safety"] = log_no_safety

    df = pd.DataFrame(all_rows)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "raw_metrics.csv"), index=False)

    summary = df.groupby("controller").agg(["mean", "std"])
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

    # Comparison bar plots per metric
    metric_names = [m for m in eval_cfg["metrics"]]
    for metric in metric_names:
        means = df.groupby("controller")[metric].mean()
        stds = df.groupby("controller")[metric].std().fillna(0.0)
        plot_comparison_bar(
            labels=list(means.index),
            means=list(means.values),
            stds=list(stds.values),
            ylabel=metric,
            title=f"{metric} by controller",
            out_path=os.path.join(out_dir, "plots", f"{metric}.png"),
        )

    # Representative episode profiles (first eval scenario)
    for series_name, ylabel in [("soc", "SoC"), ("voltage_v", "Voltage (V)"),
                                 ("temperature_c", "Temperature (C)"), ("current_a", "Current (A)")]:
        series = {}
        for ctrl_name, log in profiles.items():
            n = len(log[series_name])
            series[ctrl_name] = log[series_name]
        # pad to common time axis using each controller's own step count
        max_len = max(len(v) for v in series.values())
        time_axis = [i * dt for i in range(max_len)]
        padded = {k: (v + [v[-1]] * (max_len - len(v))) for k, v in series.items()}
        plot_profile(time_axis, padded, ylabel=ylabel,
                     title=f"{ylabel} profile (scenario: SoC={soc_grid[0]}, T={temp_grid[0]}C)",
                     out_path=os.path.join(out_dir, "plots", f"profile_{series_name}.png"))

    print(f"Evaluation complete. Results written to {out_dir}")
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO vs baseline controllers.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model .zip")
    parser.add_argument("--run-name", type=str, required=True, help="Run name for output directory")
    args = parser.parse_args()
    run_evaluation(args.model, args.run_name)


if __name__ == "__main__":
    main()