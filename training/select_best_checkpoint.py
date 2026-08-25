"""
Sweep every checkpoint saved during Stage 4 training and report which one
actually performs best — PPO is not guaranteed to improve monotonically
over very long runs (policy can find a good solution early and drift away
from it later with no LR annealing to protect it). Don't assume the final
checkpoint (trained_model.zip) is the best one; check.

v2 (audit fix, see audit/ISSUES.md ISSUE-008): the previous version scored
checkpoints by mean_final_soc alone, which cannot distinguish a checkpoint
that reaches 95% SoC safely and quickly from one that reaches 95% while
running hot, intervening on safety constantly, or taking far longer — and
could select a checkpoint with a marginally higher final SoC (e.g. 96% vs
95%) over one with substantially better thermal/safety behavior. This
version computes the full evaluation metric set (utils.metrics.
summarize_episode — the same function training/evaluate.py uses) per
checkpoint per scenario, then selects using an explicit lexicographic
policy:

    1. target_reached_rate (higher is better — did it actually finish)
    2. safety_interventions (lower is better — fewer hard constraint hits)
    3. charging_time_s (lower is better — faster, among safe/complete runs)
    4. peak_temperature_c (lower is better — thermal stress)
    5. energy_efficiency (higher is better — tie-break)

Each criterion is only used to break ties left by the previous one (values
compared with a small tolerance so near-identical checkpoints don't get
arbitrarily ordered by noise).

Usage:
    python -m training.select_best_checkpoint --run-name run_002
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from environment.battery_env import BatteryChargingEnv
from training.evaluate import _run_ppo_episode
from utils.config import load_config
from utils.metrics import aggregate_runs, summarize_episode

# Lexicographic selection order. `lower_is_better=False` means higher wins.
# `tolerance` treats values within this absolute difference as a tie for
# this criterion, falling through to the next one (avoids letting float
# noise decide the outcome on an otherwise-equal criterion).
_SELECTION_CRITERIA = [
    ("target_reached_rate", False, 1e-6),
    ("safety_interventions", True, 0.5),
    ("charging_time_s", True, 1.0),
    ("peak_temperature_c", True, 0.1),
    ("energy_efficiency", False, 1e-3),
]


def _score_checkpoint(model, env: BatteryChargingEnv, scenarios, dt: float, target_soc: float) -> dict:
    """Run one checkpoint through every scenario and return the aggregated
    (mean-across-scenarios) full metric set, using the exact same per-step
    logging and summarize_episode() that training/evaluate.py uses for
    final reported results — so checkpoint selection and final evaluation
    can never silently disagree on what a metric means."""
    per_scenario_metrics = []
    for soc0, temp0 in scenarios:
        log = _run_ppo_episode(model, env, soc0, temp0)
        per_scenario_metrics.append(
            summarize_episode(log, dt, target_soc=target_soc, initial_soc=soc0)
        )

    agg = aggregate_runs(per_scenario_metrics)
    # target_reached is boolean per scenario; report as a rate in [0,1]
    # rather than mean/std of a bool, which aggregate_runs would otherwise
    # compute correctly but under a less intuitive name.
    target_reached_rate = float(np.mean([m["target_reached"] for m in per_scenario_metrics]))

    flat = {"target_reached_rate": target_reached_rate}
    for key, stats in agg.items():
        if key == "target_reached":
            continue
        flat[key] = stats["mean"]
        flat[f"{key}_std"] = stats["std"]
        flat[f"{key}_valid_runs"] = stats["valid_runs"]
        flat[f"{key}_failed_runs"] = stats["failed_runs"]
    return flat


def _select_best(results: list[tuple[str, dict]]) -> tuple[str, dict]:
    """Apply the lexicographic policy in _SELECTION_CRITERIA. Returns the
    winning (path, metrics) pair."""
    candidates = list(results)
    for key, lower_is_better, tol in _SELECTION_CRITERIA:
        if len(candidates) == 1:
            break
        values = [m[key] for _, m in candidates]
        best_val = min(values) if lower_is_better else max(values)
        candidates = [
            (p, m) for p, m in candidates
            if abs(m[key] - best_val) <= tol
        ]
    # If still tied after every criterion, keep the first (stable: sorted
    # checkpoint path order, i.e. earliest checkpoint) rather than an
    # arbitrary max() comparison on a dict.
    return candidates[0]


def sweep_checkpoints(run_name: str, n_scenarios: int = 4):
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval")
    dt = float(sim_cfg.get("dt_seconds", env.ecm.dt))
    target_soc = float(sim_cfg["target_soc"])

    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    scenarios = [(soc_grid[i % len(soc_grid)], temp_grid[i % len(temp_grid)])
                 for i in range(n_scenarios)]

    run_dir = os.path.join("runs", run_name)
    checkpoint_paths = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.zip")))
    final_path = os.path.join(run_dir, "trained_model.zip")
    if os.path.isfile(final_path):
        checkpoint_paths.append(final_path)

    if not checkpoint_paths:
        print(f"No checkpoints found under {run_dir}/checkpoints/ (and no trained_model.zip).")
        return

    results = []
    for path in checkpoint_paths:
        model = PPO.load(path)
        metrics = _score_checkpoint(model, env, scenarios, dt, target_soc)
        results.append((path, metrics))
        print(
            f"{os.path.basename(path):28s} "
            f"target_reached_rate={metrics['target_reached_rate']:.2f}  "
            f"safety_interventions={metrics['safety_interventions']:6.2f}  "
            f"charging_time_s={metrics['charging_time_s']:8.1f}  "
            f"peak_temp_c={metrics['peak_temperature_c']:6.2f}  "
            f"energy_eff={metrics['energy_efficiency']:.3f}  "
            f"final_soc={metrics['final_soc']:.4f}"
        )

    best_path, best_metrics = _select_best(results)

    print(f"\nBest checkpoint (lexicographic: target_reached_rate > "
          f"safety_interventions > charging_time_s > peak_temperature_c > "
          f"energy_efficiency): {best_path}")
    for key in ("target_reached_rate", "safety_interventions", "charging_time_s",
                "peak_temperature_c", "energy_efficiency", "final_soc"):
        print(f"  {key}: {best_metrics[key]:.4f}")
    print("Re-run training/evaluate.py with --model pointing at this checkpoint "
          "for the full evaluation grid + plots.")

    table = pd.DataFrame(
        [{"checkpoint": os.path.basename(p), **m} for p, m in results]
    )
    table_path = os.path.join(run_dir, "checkpoint_selection.csv")
    if os.path.isdir(run_dir):
        table.to_csv(table_path, index=False)
        print(f"Full checkpoint comparison table written to {table_path}")

    return best_path


def main():
    parser = argparse.ArgumentParser(description="Find the best-performing PPO checkpoint in a run.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--n-scenarios", type=int, default=4,
                         help="Number of quick scenarios to test per checkpoint (default 4, for speed).")
    args = parser.parse_args()
    sweep_checkpoints(args.run_name, n_scenarios=args.n_scenarios)


if __name__ == "__main__":
    main()
