"""
Policy sensitivity analysis: does the trained PPO policy's output actually
depend on the environment's state, or has it converged to something close
to a constant-current strategy regardless of SoC/voltage/temperature?

Two complementary tests:

1. PARTIAL-DEPENDENCE SWEEPS — hold every observation dimension fixed at a
   baseline value except one, sweep that one across its full [0,1] range,
   and record the policy's raw (pre-clip) mean output and the resulting
   current. This isolates each state variable's individual effect on the
   policy, uncontaminated by what actually happens during a real episode.

2. REAL-TRAJECTORY CORRELATION — using the per-step trajectory CSVs already
   produced by training/evaluate.py (runs/<run_name>/evaluation/trajectories/),
   compute the correlation and total variation of applied current against
   SoC and temperature as they actually evolve during real episodes. This
   catches state-dependence that a pure sweep could miss (e.g. if the
   policy only reacts to *combinations* of state variables) and is the more
   ecologically valid test of what the policy actually does in practice.

Usage:
    python -m training.policy_sensitivity_analysis --model runs/run_006/trained_model.zip --run-name run_006
"""

from __future__ import annotations

import argparse
import glob
import os
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from utils.config import load_config


OBS_DIMS = ["soc", "voltage_norm", "temperature_norm", "prev_current_norm", "ambient_temp_norm", "state_based_safe_fraction"]


def _baseline_obs() -> np.ndarray:
    """A representative mid-charge, moderate-condition observation to hold
    fixed while sweeping each dimension in turn."""
    # soc=0.5, voltage_norm=0.5, temp_norm mapped from ~25C, prev_current_norm=0.5, ambient mapped from ~25C
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    t_range = battery_cfg["t_max_c"]  # t_min_ref is 0 in the env
    temp_norm_25c = 25.0 / t_range

    # safe_current_fraction at this baseline state (soc=0.5, temp=25C) — well
    # within safe bounds (soc_taper_start=0.9, t_derate_start=45C), so the
    # safety layer permits full current here: multiplier = 1.0.
    from safety.safety_layer import state_based_current_multiplier
    baseline_state = SimpleNamespace(soc=0.5, temperature_c=25.0)
    safe_frac_baseline = state_based_current_multiplier(baseline_state, safety_cfg)

    return np.array([0.5, 0.5, temp_norm_25c, 0.5, temp_norm_25c, safe_frac_baseline], dtype=np.float32)


def run_partial_dependence(model: PPO, i_max_a: float, n_points: int = 21) -> pd.DataFrame:
    baseline = _baseline_obs()
    rows: List[Dict] = []

    for dim_idx, dim_name in enumerate(OBS_DIMS):
        sweep_values = np.linspace(0.0, 1.0, n_points)
        for val in sweep_values:
            obs = baseline.copy()
            obs[dim_idx] = val
            obs_t = torch.as_tensor(obs).float().unsqueeze(0)
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_t)
                raw_mean = float(dist.distribution.mean.item())
            clipped = float(np.clip(raw_mean, -1.0, 1.0))
            action_val = (clipped + 1.0) / 2.0
            current_a = action_val * i_max_a
            rows.append({
                "swept_dim": dim_name,
                "swept_value": val,
                "raw_policy_mean": raw_mean,
                "clipped_action": clipped,
                "current_a": current_a,
            })

    return pd.DataFrame(rows)


def summarize_partial_dependence(df: pd.DataFrame) -> pd.DataFrame:
    """For each swept dimension, report BOTH the raw (pre-clip) policy
    response and the clipped/converted current — plus what fraction of the
    sweep was saturated at the action boundary.

    This distinguishes two very different situations that both show
    range_a == 0 in the clipped current alone: (a) the policy genuinely does
    not respond to this input at all, vs (b) the policy DOES respond
    internally (raw_mean varies) but the response is entirely absorbed by
    clipping to [-1, 1] before it ever reaches the environment. Only (a) is
    evidence of "no sensitivity" — (b) means the underlying network has
    learned something the clipped action space is hiding from view.
    """
    grouped = df.groupby("swept_dim")
    summary = grouped["current_a"].agg(["min", "max"])
    summary["range_a"] = summary["max"] - summary["min"]
    denom = df["current_a"].max()
    summary["range_fraction_of_i_max"] = summary["range_a"] / denom if denom > 0 else 0.0

    raw = grouped["raw_policy_mean"].agg(["min", "max"])
    summary["raw_policy_mean_min"] = raw["min"]
    summary["raw_policy_mean_max"] = raw["max"]
    summary["raw_policy_mean_range"] = raw["max"] - raw["min"]

    summary["fraction_saturated"] = grouped["raw_policy_mean"].apply(
        lambda s: float(((s <= -1.0) | (s >= 1.0)).mean())
    )

    return summary.sort_values("raw_policy_mean_range", ascending=False)


def analyze_real_trajectories(run_name: str) -> pd.DataFrame:
    """Correlate applied current against SoC and temperature within actual
    logged PPO episodes (both safety-enforced and no-safety variants)."""
    traj_dir = os.path.join("runs", run_name, "evaluation", "trajectories")
    ppo_files = sorted(glob.glob(os.path.join(traj_dir, "ppo_*.csv"))) + \
        sorted(glob.glob(os.path.join(traj_dir, "ppo_no_safety_*.csv")))

    if not ppo_files:
        print(f"No PPO trajectory files found under {traj_dir}. "
              f"Run training/evaluate.py first with this run-name.")
        return pd.DataFrame()

    rows = []
    for path in ppo_files:
        df = pd.read_csv(path)
        variant = "ppo_no_safety" if "no_safety" in os.path.basename(path) else "ppo"
        current_std = df["current_a"].std()
        current_range = df["current_a"].max() - df["current_a"].min()
        soc_corr = df["current_a"].corr(df["soc"]) if current_std > 1e-9 else float("nan")
        temp_corr = df["current_a"].corr(df["temperature_c"]) if current_std > 1e-9 else float("nan")
        rows.append({
            "file": os.path.basename(path),
            "variant": variant,
            "current_std_a": current_std,
            "current_range_a": current_range,
            "corr_current_vs_soc": soc_corr,
            "corr_current_vs_temp": temp_corr,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze whether the PPO policy is state-adaptive or near-constant.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model .zip")
    parser.add_argument("--run-name", type=str, required=True,
                         help="Run name whose evaluation/trajectories/ folder to analyze (must have run training.evaluate first)")
    args = parser.parse_args()

    battery_cfg = load_config("battery")
    i_max_a = float(battery_cfg["i_max_a"])

    print("=" * 70)
    print("PART 1: Partial-dependence sweep (isolated effect of each state variable)")
    print("=" * 70)
    model = PPO.load(args.model)
    pd_df = run_partial_dependence(model, i_max_a)
    pd_summary = summarize_partial_dependence(pd_df)
    print(pd_summary.to_string())
    print()

    out_dir = os.path.join("runs", args.run_name, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    pd_df.to_csv(os.path.join(out_dir, "sensitivity_partial_dependence.csv"), index=False)
    pd_summary.to_csv(os.path.join(out_dir, "sensitivity_partial_dependence_summary.csv"))

    print("=" * 70)
    print("PART 2: Real-trajectory correlation (actual observed behavior)")
    print("=" * 70)
    traj_df = analyze_real_trajectories(args.run_name)
    if not traj_df.empty:
        print(traj_df.to_string(index=False))
        traj_df.to_csv(os.path.join(out_dir, "sensitivity_trajectory_correlation.csv"), index=False)
    print()

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("NOTE: 'ppo' (safety-enforced) trajectories are excluded from the")
    print("verdict below — the safety layer mechanically tapers current near")
    print("SoC/voltage limits regardless of whether the underlying policy is")
    print("adaptive, so their variation isn't diagnostic of the policy itself.")
    print("The verdict uses only the partial-dependence sweep (pure policy,")
    print("no safety layer involved) and the ppo_no_safety trajectories")
    print("(policy's own behavior, safety layer in monitoring-only mode).")
    print()
    print("IMPORTANT: a dimension can show clipped current_a range == 0 while")
    print("the underlying raw_policy_mean still varies substantially — the")
    print("action is symmetric [-1,1] and gets hard-clipped before reaching")
    print("the environment, so a raw mean moving from e.g. 1.05 to 2.3 is")
    print("entirely invisible in the clipped current. The verdict below checks")
    print("raw_policy_mean_range, not clipped current, for exactly this reason.")
    print("Per-dimension fraction_saturated (in the summary table above) shows")
    print("how much of each sweep was pinned at the clip boundary.")
    print()

    # Use the RAW (pre-clip) policy response, not the clipped current, as the
    # primary flatness signal — clipping can hide genuine internal
    # sensitivity (see note above). But raw variation alone isn't enough:
    # a raw mean moving from 3.9 to 4.8 varies internally, yet EVERY value
    # in that range clips to the identical action (both are >1.0), so it
    # has ZERO practical behavioral effect despite a nonzero raw range —
    # this was a real gap in an earlier version of this verdict, which
    # would have called that case "(A) state-adaptive" purely because
    # raw_policy_mean_range > 0.05, without checking whether the range
    # straddles the clip boundary or sits entirely on one side of it.
    max_raw_range = pd_summary["raw_policy_mean_range"].max()
    max_range_fraction = pd_summary["range_fraction_of_i_max"].max()
    no_safety_rows = traj_df[traj_df["variant"] == "ppo_no_safety"] if not traj_df.empty else pd.DataFrame()
    mean_current_std_no_safety = no_safety_rows["current_std_a"].mean() if not no_safety_rows.empty else float("nan")

    policy_is_flat = max_raw_range < 0.05 and (
        no_safety_rows.empty or mean_current_std_no_safety < 0.05 * i_max_a
    )
    # Practically saturated: every dimension's sweep is ~always clipped
    # (fraction_saturated ~1.0 everywhere), regardless of how much the raw
    # mean itself moves within the saturated region.
    practically_saturated = (pd_summary["fraction_saturated"] >= 0.95).all()

    if policy_is_flat:
        print(f"Raw policy output varies by less than 0.05 (pre-clip units) across")
        print("every state dimension in the partial-dependence sweep — this is a")
        print("genuine lack of internal sensitivity, not a clipping artifact — and")
        print("the safety-layer-free (ppo_no_safety) trajectories show negligible")
        print(f"current variation (mean std = {mean_current_std_no_safety:.2f}A).")
        print("=> Evidence supports (B): PPO discovered an optimized near-constant-current")
        print("   strategy under the v1 reward/environment, not a state-adaptive policy.")
    elif practically_saturated:
        print(f"Raw policy output DOES vary with state (largest raw range: {max_raw_range:.3f}),")
        print("but every swept dimension is saturated (fraction_saturated >= 0.95) across")
        print("essentially the ENTIRE sweep — the raw mean is moving, but staying on one")
        print("side of the clip boundary throughout, so every value in that range produces")
        print("the IDENTICAL clipped action. This has zero practical behavioral effect.")
        print("=> Evidence supports (B), practically: the network technically responds")
        print("   internally, but that response is invisible to the environment. This is")
        print("   NOT genuine state-adaptive behavior despite nonzero raw variation.")
    else:
        print(f"Policy output varies meaningfully with state (largest RAW response")
        print(f"range: {max_raw_range:.3f}; largest clipped-current sweep range:")
        print(f"{max_range_fraction*100:.1f}% of i_max; ppo_no_safety trajectory current")
        print(f"std: {mean_current_std_no_safety:.1f}A).")
        print("=> Evidence supports (A): PPO learned a state-adaptive charging policy.")
    print()
    print("For reference, 'ppo' (safety-enforced) trajectory stats are in")
    print(f"{out_dir}/sensitivity_trajectory_correlation.csv — expect these to show")
    print("more variation than ppo_no_safety even under a flat/constant policy,")
    print("purely from the safety layer's own tapering behavior.")
    print()
    print(f"Full data written to {out_dir}/sensitivity_*.csv")


if __name__ == "__main__":
    main()