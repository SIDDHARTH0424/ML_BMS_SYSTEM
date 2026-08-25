"""
Generate report-ready graphs and a results summary table for a trained run.

Reads the CSVs already produced by `training/evaluate.py` and
`training/policy_sensitivity_analysis.py` for a given --run-name (it does
NOT re-run evaluation or training — run those two scripts against your
checkpoint first if you haven't already):

    python -m training.evaluate --model runs\\run_010\\checkpoints\\ppo_bms_75000_steps.zip --run-name run_010
    python -m training.policy_sensitivity_analysis --model runs\\run_010\\checkpoints\\ppo_bms_75000_steps.zip --run-name run_010
    python -m training.generate_report --run-name run_010

Produces, under runs/<run_name>/report/:
    01_comparison_final_soc.png
    02_comparison_charging_time.png
    03_comparison_safety_interventions.png
    04_comparison_energy_efficiency.png
    05_trajectory_profiles.png       (SoC / Voltage / Temperature / Current, one representative scenario)
    06_sensitivity_raw_response.png  (partial-dependence: raw policy response per observation dimension)
    results_summary.md               (the same table format used in the project's results/discussion doc)
    results_summary.csv
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CONTROLLER_ORDER = ["adaptive", "cc", "cccv", "max_current", "ppo", "ppo_no_safety"]
CONTROLLER_LABELS = {
    "adaptive": "Adaptive",
    "cc": "CC",
    "cccv": "CCCV",
    "max_current": "Max-Current",
    "ppo": "PPO",
    "ppo_no_safety": "PPO (no safety)",
}
CONTROLLER_COLORS = {
    "adaptive": "#8c8c8c",
    "cc": "#4C72B0",
    "cccv": "#55A868",
    "max_current": "#C44E52",
    "ppo": "#8172B2",
    "ppo_no_safety": "#CCB974",
}


def _bar_chart(means: pd.Series, stds: pd.Series, ylabel: str, title: str, out_path: str) -> None:
    order = [c for c in CONTROLLER_ORDER if c in means.index]
    labels = [CONTROLLER_LABELS[c] for c in order]
    colors = [CONTROLLER_COLORS[c] for c in order]
    vals = [means[c] for c in order]
    errs = [stds[c] if c in stds.index and pd.notna(stds[c]) else 0.0 for c in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_comparison_charts(raw_metrics_path: str, out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(raw_metrics_path)
    grouped_mean = df.groupby("controller").mean(numeric_only=True)
    grouped_std = df.groupby("controller").std(numeric_only=True)

    _bar_chart(grouped_mean["final_soc"], grouped_std["final_soc"],
               "Final SoC", "Final State of Charge by Controller",
               os.path.join(out_dir, "01_comparison_final_soc.png"))

    _bar_chart(grouped_mean["charging_time_s"], grouped_std["charging_time_s"],
               "Charging Time (s)", "Charging Time by Controller",
               os.path.join(out_dir, "02_comparison_charging_time.png"))

    _bar_chart(grouped_mean["safety_interventions"], grouped_std["safety_interventions"],
               "Safety Interventions (count)", "Safety-Layer Interventions by Controller",
               os.path.join(out_dir, "03_comparison_safety_interventions.png"))

    if "energy_efficiency" in grouped_mean.columns:
        _bar_chart(grouped_mean["energy_efficiency"], grouped_std["energy_efficiency"],
                   "Energy Efficiency (fraction)", "Energy Efficiency by Controller",
                   os.path.join(out_dir, "04_comparison_energy_efficiency.png"))

    return grouped_mean


def make_trajectory_profile(traj_dir: str, out_dir: str, soc0: float = 0.10, temp0: float = 15.0) -> None:
    """One representative scenario, all controllers overlaid, 4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("soc", "SoC", axes[0, 0]), ("voltage_v", "Voltage (V)", axes[0, 1]),
              ("temperature_c", "Temperature (C)", axes[1, 0]), ("current_a", "Current (A)", axes[1, 1])]

    found_any = False
    for controller in CONTROLLER_ORDER:
        fname = f"{controller}_soc{soc0:.2f}_temp{temp0:.0f}.csv"
        path = os.path.join(traj_dir, fname)
        if not os.path.isfile(path):
            continue
        found_any = True
        tdf = pd.read_csv(path)
        for col, ylabel, ax in panels:
            ax.plot(tdf["time_s"], tdf[col], label=CONTROLLER_LABELS[controller],
                    color=CONTROLLER_COLORS[controller], linewidth=1.5)

    if not found_any:
        plt.close(fig)
        print(f"No trajectory files found for soc0={soc0}, temp0={temp0} in {traj_dir} -- skipping profile plot.")
        return

    for col, ylabel, ax in panels:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Charging Profiles -- initial SoC={soc0:.0%}, ambient={temp0:.0f}C")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_trajectory_profiles.png"), dpi=150)
    plt.close(fig)


def make_sensitivity_chart(sensitivity_summary_path: str, out_dir: str) -> None:
    if not os.path.isfile(sensitivity_summary_path):
        print(f"No partial-dependence summary found at {sensitivity_summary_path} -- skipping sensitivity chart.")
        return
    # This file is the raw per-sweep-point CSV; recompute the summary here
    # rather than relying on the console-only summary table.
    df = pd.read_csv(sensitivity_summary_path)
    summary = df.groupby("swept_dim")["raw_policy_mean"].agg(lambda s: s.max() - s.min())
    summary = summary.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(summary.index, summary.values, color="#8172B2")
    ax.set_xlabel("Raw policy response range (pre-clip)")
    ax.set_title("Policy Sensitivity by Observation Dimension")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_sensitivity_raw_response.png"), dpi=150)
    plt.close(fig)


def write_results_summary(grouped_mean: pd.DataFrame, out_dir: str, run_name: str, checkpoint_label: str) -> None:
    cols = [c for c in ["final_soc", "charging_time_s", "safety_interventions", "target_reached"]
            if c in grouped_mean.columns]
    table = grouped_mean[cols].reindex([c for c in CONTROLLER_ORDER if c in grouped_mean.index])
    table.index = [CONTROLLER_LABELS[c] for c in table.index]

    table.to_csv(os.path.join(out_dir, "results_summary.csv"))

    lines = [f"# Results Summary -- {run_name} ({checkpoint_label})", ""]
    header = "| Controller | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    lines += [header, sep]
    for idx, row in table.iterrows():
        vals = " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols)
        lines.append(f"| {idx} | {vals} |")

    with open(os.path.join(out_dir, "results_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate report graphs and a results summary for a trained run.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint-label", type=str, default=None,
                         help="Label for the report title, e.g. '75k checkpoint'. Defaults to --run-name only.")
    parser.add_argument("--scenario-soc", type=float, default=0.10,
                         help="Initial SoC of the representative scenario used for the trajectory profile plot.")
    parser.add_argument("--scenario-temp", type=float, default=15.0,
                         help="Ambient temperature of the representative scenario used for the trajectory profile plot.")
    args = parser.parse_args()

    eval_dir = os.path.join("runs", args.run_name, "evaluation")
    raw_metrics_path = os.path.join(eval_dir, "raw_metrics.csv")
    traj_dir = os.path.join(eval_dir, "trajectories")
    sensitivity_path = os.path.join(eval_dir, "sensitivity_partial_dependence.csv")

    if not os.path.isfile(raw_metrics_path):
        print(f"ERROR: {raw_metrics_path} not found. Run training/evaluate.py against your "
              f"checkpoint first, with --run-name {args.run_name}.")
        return

    out_dir = os.path.join("runs", args.run_name, "report")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_label = args.checkpoint_label or args.run_name

    grouped_mean = make_comparison_charts(raw_metrics_path, out_dir)
    make_trajectory_profile(traj_dir, out_dir, soc0=args.scenario_soc, temp0=args.scenario_temp)
    make_sensitivity_chart(sensitivity_path, out_dir)
    write_results_summary(grouped_mean, out_dir, args.run_name, checkpoint_label)

    print(f"Report generated in {out_dir}/:")
    for fname in sorted(os.listdir(out_dir)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
