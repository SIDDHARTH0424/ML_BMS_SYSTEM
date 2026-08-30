"""
Aggregate research results for RL-BMS-Driving.
Computes per-seed statistics, cross-cycle means, and multi-seed statistics.
"""

import os
import pandas as pd
import numpy as np

RESULTS_DIR = "audit/final_research"

def load_summary_csv(path):
    """Load a summary CSV and return DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"Warning: {path} not found")
        return pd.DataFrame()

def compute_per_seed_statistics():
    """Compute statistics across seeds for each controller and cycle."""
    seeds = [7, 21, 42]
    controllers = ["RuleBasedEMS", "PPO_seed7", "PPO_seed21", "PPO_seed42"]
    cycles = ["epa_udds", "epa_hwfet", "epa_us06", "wltp_class3b"]

    metrics = [
        "distance_km", "mean_speed_kmh", "discharge_energy_wh", "regen_energy_wh",
        "net_energy_wh", "wh_per_km", "regen_recovery_fraction", "min_soc",
        "final_soc", "delta_soc", "max_temperature_c", "avg_temperature_c",
        "total_power_deficit_wh", "safety_interventions", "safety_intervention_rate"
    ]

    # For each controller, cycle, and metric, collect values across seeds
    stats_data = []

    for controller in controllers:
        for cycle in cycles:
            values = {metric: [] for metric in metrics}

            for seed in seeds:
                if controller == "RuleBasedEMS":
                    # Rule-Based has no seed variation, use the same file for all seeds
                    summary_path = os.path.join(RESULTS_DIR, "driving_rule_based_summary.csv")
                else:
                    # For PPO controllers, use the specific seed file
                    summary_path = os.path.join(RESULTS_DIR, f"driving_ppo_seed{seed}_summary.csv")

                df = load_summary_csv(summary_path)
                if not df.empty:
                    cycle_row = df[df["cycle_id"] == cycle]
                    if not cycle_row.empty:
                        for metric in metrics:
                            if metric in cycle_row.columns:
                                values[metric].append(cycle_row.iloc[0][metric])

            # Compute statistics for this controller-cycle combination
            for metric in metrics:
                if values[metric]:  # Only if we have data
                    vals = np.array(values[metric])
                    stats_data.append({
                        "controller": controller,
                        "cycle_id": cycle,
                        "metric": metric,
                        "mean": np.mean(vals),
                        "std": np.std(vals),
                        "min": np.min(vals),
                        "max": np.max(vals),
                        "seed_7": values[metric][0] if len(values[metric]) > 0 else None,
                        "seed_21": values[metric][1] if len(values[metric]) > 1 else None,
                        "seed_42": values[metric][2] if len(values[metric]) > 2 else None,
                    })

    return pd.DataFrame(stats_data)

def compute_cross_cycle_means():
    """Compute cross-cycle means for each controller and seed."""
    seeds = [7, 21, 42]
    controllers = ["RuleBasedEMS", "PPO_seed7", "PPO_seed21", "PPO_seed42"]
    cycles = ["epa_udds", "epa_hwfet", "epa_us06", "wltp_class3b"]

    metrics = [
        "distance_km", "mean_speed_kmh", "discharge_energy_wh", "regen_energy_wh",
        "net_energy_wh", "wh_per_km", "regen_recovery_fraction", "min_soc",
        "final_soc", "delta_soc", "max_temperature_c", "avg_temperature_c",
        "total_power_deficit_wh", "safety_interventions", "safety_intervention_rate"
    ]

    # For each controller, seed, and metric, compute mean across cycles
    cross_cycle_data = []

    for controller in controllers:
        for seed in seeds:
            if controller == "RuleBasedEMS":
                # Rule-Based has no seed variation
                controller_seed = None
                summary_path = os.path.join(RESULTS_DIR, "driving_rule_based_summary.csv")
            else:
                controller_seed = seed
                summary_path = os.path.join(RESULTS_DIR, f"driving_ppo_seed{seed}_summary.csv")

            df = load_summary_csv(summary_path)
            if not df.empty:
                # Filter for the standard cycles
                cycle_df = df[df["cycle_id"].isin(cycles)]
                if not cycle_df.empty:
                    for metric in metrics:
                        if metric in cycle_df.columns:
                            vals = cycle_df[metric].values
                            if len(vals) > 0:
                                cross_cycle_data.append({
                                    "controller": controller,
                                    "seed": controller_seed if controller_seed else "N/A",
                                    "metric": metric,
                                    "cross_cycle_mean": np.mean(vals),
                                    "cross_cycle_std": np.std(vals),
                                    "epa_udds": cycle_df[cycle_df["cycle_id"] == "epa_udds"][metric].iloc[0] if "epa_udds" in cycle_df["cycle_id"].values else None,
                                    "epa_hwfet": cycle_df[cycle_df["cycle_id"] == "epa_hwfet"][metric].iloc[0] if "epa_hwfet" in cycle_df["cycle_id"].values else None,
                                    "epa_us06": cycle_df[cycle_df["cycle_id"] == "epa_us06"][metric].iloc[0] if "epa_us06" in cycle_df["cycle_id"].values else None,
                                    "wltp_class3b": cycle_df[cycle_df["cycle_id"] == "wltp_class3b"][metric].iloc[0] if "wltp_class3b" in cycle_df["cycle_id"].values else None,
                                })

    return pd.DataFrame(cross_cycle_data)

def compute_multi_seed_statistics():
    """Compute multi-seed statistics for PPO controllers (seeds 7, 21, 42)."""
    controllers = ["PPO_seed7", "PPO_seed21", "PPO_seed42"]
    cycles = ["epa_udds", "epa_hwfet", "epa_us06", "wltp_class3b"]

    metrics = [
        "distance_km", "mean_speed_kmh", "discharge_energy_wh", "regen_energy_wh",
        "net_energy_wh", "wh_per_km", "regen_recovery_fraction", "min_soc",
        "final_soc", "delta_soc", "max_temperature_c", "avg_temperature_c",
        "total_power_deficit_wh", "safety_interventions", "safety_intervention_rate"
    ]

    # For each PPO controller type (aggregating across seeds), cycle, and metric
    multi_seed_data = []

    # Group controllers by type (PPO) and compute statistics across seeds
    ppo_controllers = ["PPO_seed7", "PPO_seed21", "PPO_seed42"]

    for cycle in cycles:
        for metric in metrics:
            values = []
            seed_values = {}

            for controller in ppo_controllers:
                # Extract seed from controller name (e.g., PPO_seed7 -> 7)
                seed = controller.split("_")[-1].replace("seed", "")
                summary_path = os.path.join(RESULTS_DIR, f"driving_ppo_seed{seed}_summary.csv")
                df = load_summary_csv(summary_path)
                if not df.empty:
                    cycle_row = df[df["cycle_id"] == cycle]
                    if not cycle_row.empty and metric in cycle_row.columns:
                        val = cycle_row.iloc[0][metric]
                        values.append(val)
                        seed_values[f"seed_{seed}"] = val

            if values:  # Only if we have data from all seeds
                vals = np.array(values)
                multi_seed_data.append({
                    "controller_type": "PPO",
                    "cycle_id": cycle,
                    "metric": metric,
                    "multi_seed_mean": np.mean(vals),
                    "multi_seed_std": np.std(vals),
                    "multi_seed_min": np.min(vals),
                    "multi_seed_max": np.max(vals),
                    **seed_values
                })

    return pd.DataFrame(multi_seed_data)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Computing per-seed statistics...")
    per_seed_stats = compute_per_seed_statistics()
    if not per_seed_stats.empty:
        per_seed_stats.to_csv(os.path.join(RESULTS_DIR, "per_seed_statistics.csv"), index=False)
        print(f"Per-seed statistics saved to {os.path.join(RESULTS_DIR, 'per_seed_statistics.csv')}")

    print("Computing cross-cycle means...")
    cross_cycle_means = compute_cross_cycle_means()
    if not cross_cycle_means.empty:
        cross_cycle_means.to_csv(os.path.join(RESULTS_DIR, "cross_cycle_means.csv"), index=False)
        print(f"Cross-cycle means saved to {os.path.join(RESULTS_DIR, 'cross_cycle_means.csv')}")

    print("Computing multi-seed statistics...")
    multi_seed_stats = compute_multi_seed_statistics()
    if not multi_seed_stats.empty:
        multi_seed_stats.to_csv(os.path.join(RESULTS_DIR, "multi_seed_statistics.csv"), index=False)
        print(f"Multi-seed statistics saved to {os.path.join(RESULTS_DIR, 'multi_seed_statistics.csv')}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Per-seed statistics: {len(per_seed_stats)} rows")
    print(f"Cross-cycle means: {len(cross_cycle_means)} rows")
    print(f"Multi-seed statistics: {len(multi_seed_stats)} rows")

if __name__ == "__main__":
    main()