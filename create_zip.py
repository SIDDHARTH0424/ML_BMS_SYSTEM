"""
Standalone script to bundle all key RL-BMS code, configs, and a run's evaluation & training logs
into a single ZIP file (`rl_bms_<run_name>_bundle.zip`).

Includes:
  1. Priority Environment & Safety Code (battery_env.py, safety_layer.py, ecm_model.py)
  2. Training & Analysis Code (train_ppo.py, evaluate.py, policy_sensitivity_analysis.py)
  3. Config Files (reward.yaml, simulation.yaml, ppo.yaml, battery.yaml, safety.yaml, evaluation.yaml)
  4. Selected Run's Results (raw_metrics.csv, summary_metrics.csv, reward_components.csv, sensitivity CSVs)
  5. Selected Run's TensorBoard Event Logs & Extracted TensorBoard Scalars CSV

Usage:
    python create_zip.py --run-name run_009
"""

import argparse
import glob
import os
import zipfile
import pandas as pd

def parse_tensorboard_scalars(tb_dir: str) -> pd.DataFrame:
    """Attempt to parse TensorBoard event files into a clean pandas DataFrame."""
    records = []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_files = sorted(glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents*"), recursive=True))
        for ef in event_files:
            run_tag = os.path.basename(os.path.dirname(ef))
            try:
                ea = EventAccumulator(ef)
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    for event in ea.Scalars(tag):
                        records.append({
                            "tb_subfolder": run_tag,
                            "tag": tag,
                            "step": event.step,
                            "value": event.value,
                            "wall_time": event.wall_time,
                        })
            except Exception as ex:
                print(f"[Warning] Could not parse TB file {ef}: {ex}")
    except ImportError:
        print("[Info] tensorboard package not installed; skipping scalar extraction to CSV.")

    return pd.DataFrame(records)

def get_latest_run(base_dir: str) -> str:
    """Find the latest run folder under runs/ directory."""
    runs_dir = os.path.join(base_dir, "runs")
    if not os.path.exists(runs_dir):
        return "run_009"
    run_folders = [f for f in os.listdir(runs_dir) if f.startswith("run_") and os.path.isdir(os.path.join(runs_dir, f))]
    if not run_folders:
        return "run_009"
    # Sort numerically by suffix if possible
    try:
        run_folders.sort(key=lambda x: int(x.split("_")[1]))
    except (IndexError, ValueError):
        run_folders.sort()
    return run_folders[-1]

def create_bundle_zip(run_name: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, f"rl_bms_{run_name}_bundle.zip")

    # Files to include (relative_path_in_repo, archive_path)
    file_manifest = [
        # 1. Environment & Safety Code (Highest Priority)
        ("environment/battery_env.py", "environment/battery_env.py"),
        ("safety/safety_layer.py", "safety/safety_layer.py"),
        ("environment/ecm_model.py", "environment/ecm_model.py"),

        # 2. Configs (Highest Priority & Useful)
        ("configs/reward.yaml", "configs/reward.yaml"),
        ("configs/simulation.yaml", "configs/simulation.yaml"),
        ("configs/ppo.yaml", "configs/ppo.yaml"),
        ("configs/battery.yaml", "configs/battery.yaml"),
        ("configs/safety.yaml", "configs/safety.yaml"),
        ("configs/evaluation.yaml", "configs/evaluation.yaml"),

        # 3. Training & Evaluation Code
        ("training/evaluate.py", "training/evaluate.py"),
        ("training/policy_sensitivity_analysis.py", "training/policy_sensitivity_analysis.py"),
        ("agents/train_ppo.py", "agents/train_ppo.py"),

        # 4. Run-Specific Config Snapshots
        (f"runs/{run_name}/config/battery.yaml", f"runs/{run_name}/config/battery.yaml"),
        (f"runs/{run_name}/config/evaluation.yaml", f"runs/{run_name}/config/evaluation.yaml"),
        (f"runs/{run_name}/config/ppo.yaml", f"runs/{run_name}/config/ppo.yaml"),
        (f"runs/{run_name}/config/reward.yaml", f"runs/{run_name}/config/reward.yaml"),
        (f"runs/{run_name}/config/safety.yaml", f"runs/{run_name}/config/safety.yaml"),
        (f"runs/{run_name}/config/simulation.yaml", f"runs/{run_name}/config/simulation.yaml"),

        # 5. Run Logs & Metrics
        (f"runs/{run_name}/evaluation/raw_metrics.csv", f"runs/{run_name}/evaluation/raw_metrics.csv"),
        (f"runs/{run_name}/evaluation/summary_metrics.csv", f"runs/{run_name}/evaluation/summary_metrics.csv"),
        (f"runs/{run_name}/evaluation/sensitivity_partial_dependence.csv", f"runs/{run_name}/evaluation/sensitivity_partial_dependence.csv"),
        (f"runs/{run_name}/evaluation/sensitivity_partial_dependence_summary.csv", f"runs/{run_name}/evaluation/sensitivity_partial_dependence_summary.csv"),
        (f"runs/{run_name}/evaluation/sensitivity_trajectory_correlation.csv", f"runs/{run_name}/evaluation/sensitivity_trajectory_correlation.csv"),
        (f"runs/{run_name}/reward_components.csv", f"runs/{run_name}/reward_components.csv"),
    ]

    # Dynamically find Tensorboard event files & evaluation trajectory files for the run
    tb_dir = os.path.join(base_dir, "runs", run_name, "tensorboard")
    tb_files = glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents*"), recursive=True)
    for tf in tb_files:
        rel = os.path.relpath(tf, base_dir)
        file_manifest.append((rel, rel))

    traj_dir = os.path.join(base_dir, "runs", run_name, "evaluation", "trajectories")
    traj_files = glob.glob(os.path.join(traj_dir, "*.csv"))
    for tf in traj_files:
        rel = os.path.relpath(tf, base_dir)
        file_manifest.append((rel, rel))

    # Also parse Tensorboard events into CSV if available
    tb_csv_path = os.path.join(base_dir, "runs", run_name, "tensorboard_scalars.csv")
    tb_df = parse_tensorboard_scalars(tb_dir)
    if not tb_df.empty:
        tb_df.to_csv(tb_csv_path, index=False)
        file_manifest.append((os.path.relpath(tb_csv_path, base_dir), f"runs/{run_name}/tensorboard_scalars.csv"))

    # Write ZIP
    print(f"Creating ZIP archive at {zip_path}...")
    written_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for local_rel, arc_name in file_manifest:
            abs_p = os.path.join(base_dir, local_rel)
            if os.path.exists(abs_p):
                zf.write(abs_p, arc_name)
                written_count += 1
                print(f"  [+] Added: {arc_name}")
            else:
                print(f"  [-] Skipped (file not found): {local_rel}")

    print(f"\nSuccessfully archived {written_count} files into '{os.path.basename(zip_path)}'.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_run = get_latest_run(base_dir)
    
    parser = argparse.ArgumentParser(description="Bundle RL-BMS code, configs, and run results into a zip.")
    parser.add_argument("--run-name", type=str, default=default_run, help=f"Name of the run directory to bundle (default: {default_run})")
    args = parser.parse_args()
    
    create_bundle_zip(args.run_name)

if __name__ == "__main__":
    main()
