"""
Passive Cooling Validation for RL-BMS Driving ECM
Validates that the battery ECM demonstrates physically realistic passive cooling
when current ≈ 0 A, vehicle stopped, and battery temperature > ambient.
"""

import sys
import os
# Add the project root to the Python path so we can import from environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from environment.ecm_model import BatteryECM
from dataclasses import dataclass
from typing import Dict
import yaml
from pathlib import Path


@dataclass
class ValidationConfig:
    """Configuration for cooling validation test."""
    nominal_capacity_ah: float = 121.0  # Ah
    r0_ohm: float = 0.05  # Ohms
    r1_ohm: float = 0.1   # Ohms
    c1_farad: float = 2000.0  # Farads
    mass_kg: float = 200.0  # kg
    specific_heat_j_per_kgk: float = 900.0  # J/(kg·K)
    convection_h: float = 5.0  # W/(m²·K)
    surface_area_m2: float = 2.0  # m²
    v_max: float = 4.2  # V per cell * cells in series (simplified)
    v_min: float = 2.5  # V per cell * cells in series (simplified)
    t_max_c: float = 60.0  # °C
    i_max_a: float = 160.0  # A
    soh_degradation_per_ah: float = 0.0  # Disabled for validation
    dt_seconds: float = 1.0  # 1 second timestep
    integration_method: str = "euler"


def create_battery_config() -> Dict:
    """Create battery configuration dictionary for ECM."""
    config = ValidationConfig()
    return {
        "nominal_capacity_ah": config.nominal_capacity_ah,
        "r0_ohm": config.r0_ohm,
        "r1_ohm": config.r1_ohm,
        "c1_farad": config.c1_farad,
        "mass_kg": config.mass_kg,
        "specific_heat_j_per_kgk": config.specific_heat_j_per_kgk,
        "convection_h": config.convection_h,
        "surface_area_m2": config.surface_area_m2,
        "v_max": config.v_max,
        "v_min": config.v_min,
        "t_max_c": config.t_max_c,
        "i_max_a": config.i_max_a,
        "soh_degradation_per_ah_throughput": config.soh_degradation_per_ah,
        "dt_seconds": config.dt_seconds,
        "integration_method": config.integration_method,
        "ocv_soc_points": {
            "soc": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "ocv_v": [3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2]  # Simplified NMC OCV
        }
    }


def validate_passive_cooling() -> Dict:
    """
    Validate passive cooling behavior of the ECM.

    Test condition:
    - battery temperature > ambient
    - current ≈ 0 A
    - vehicle stopped
    - ambient temperature < battery temperature

    Returns:
        Dictionary with validation results and trajectory data
    """
    print("Starting passive cooling validation...")

    # Create battery configuration
    battery_config = create_battery_config()

    # Initialize ECM
    ecm = BatteryECM(battery_config)

    # Test conditions
    initial_temp_c = 50.0  # Battery starts at 50°C
    ambient_temp_c = 25.0  # Ambient at 25°C
    initial_soc = 0.5      # 50% SoC

    # Initial state
    state = ecm.reset_state(initial_soc=initial_soc, ambient_temp_c=ambient_temp_c)
    state.temperature_c = initial_temp_c  # Set initial temperature above ambient

    # Validation parameters
    max_time_s = 3600  # 1 hour max simulation
    dt = battery_config["dt_seconds"]

    # Storage for trajectory
    trajectory = {
        "time_s": [],
        "temperature_c": [],
        "ambient_temp_c": [],
        "current_a": [],
        "heat_generation_w": [],
        "soc": [],
        "terminal_voltage_v": []
    }

    print(f"Initial conditions:")
    print(f"  Battery temperature: {initial_temp_c}°C")
    print(f"  Ambient temperature: {ambient_temp_c}°C")
    print(f"  SoC: {initial_soc}")
    print(f"  Current: 0 A (passive cooling)")
    print(f"  Simulating for {max_time_s} seconds...")

    # Simulation loop
    for step in range(int(max_time_s / dt)):
        # Step the ECM with zero current (passive cooling)
        state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient_temp_c)

        # Calculate heat generation at zero current
        heat_gen = ecm.heat_generation_w(state, 0.0)
        terminal_voltage = ecm.terminal_voltage(state, 0.0)

        # Store trajectory data
        trajectory["time_s"].append(step * dt)
        trajectory["temperature_c"].append(state.temperature_c)
        trajectory["ambient_temp_c"].append(ambient_temp_c)
        trajectory["current_a"].append(0.0)
        trajectory["heat_generation_w"].append(heat_gen)
        trajectory["soc"].append(state.soc)
        trajectory["terminal_voltage_v"].append(terminal_voltage)

        # Progress indicator
        if step % 600 == 0 and step > 0:  # Every 10 minutes
            print(f"  Time: {step * dt:.0f}s, Temp: {state.temperature_c:.2f}°C")

    print("Simulation complete.")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(trajectory)

    # Validation checks
    validation_results = {
        "initial_temperature_c": initial_temp_c,
        "ambient_temperature_c": ambient_temp_c,
        "final_temperature_c": df["temperature_c"].iloc[-1],
        "minimum_temperature_c": df["temperature_c"].min(),
        "maximum_temperature_c": df["temperature_c"].max(),
        "cooling_duration_s": df["time_s"].iloc[-1],
        "temperature_trend": "cooling" if df["temperature_c"].iloc[-1] < df["temperature_c"].iloc[0] else "heating",
        "finite_value_result": True if np.isfinite(df["temperature_c"]).all() else False,
        "no_artificial_oscillations": True,  # Will check below
        "no_impossible_increases": True,     # Will check below
        "toward_ambient": True if df["temperature_c"].iloc[-1] < df["temperature_c"].iloc[0] else False,
        "validation_passed": True  # Will determine based on checks
    }

    # Check for artificial oscillations (more than 5 significant reversals in trend)
    temp_diff = np.diff(df["temperature_c"])
    sign_changes = np.sum(np.diff(np.sign(temp_diff)) != 0)
    validation_results["sign_changes_in_derivative"] = sign_changes
    validation_results["no_artificial_oscillations"] = sign_changes < 10  # Allow some noise

    # Check for impossible temperature increases (when cooling should occur)
    # Since we start hot and ambient is cooler, temperature should monotonically decrease
    # or at least not show sustained increases
    increasing_periods = np.sum(temp_diff > 0.01)  # More than 0.01°C increase per step
    validation_results["increasing_periods_count"] = increasing_periods
    validation_results["no_impossible_increases"] = increasing_periods < len(temp_diff) * 0.1  # Less than 10% of steps

    # Overall validation
    validation_results["validation_passed"] = (
        validation_results["finite_value_result"] and
        validation_results["no_artificial_oscillations"] and
        validation_results["no_impossible_increases"] and
        validation_results["toward_ambient"] and
        validation_results["minimum_temperature_c"] >= ambient_temp_c - 1.0  # Allow 1°C overshoot due to discretization
    )

    # Print results
    print("\n=== COOLING VALIDATION RESULTS ===")
    print(f"Initial temperature: {validation_results['initial_temperature_c']:.2f}°C")
    print(f"Ambient temperature: {validation_results['ambient_temperature_c']:.2f}°C")
    print(f"Final temperature: {validation_results['final_temperature_c']:.2f}°C")
    print(f"Minimum temperature: {validation_results['minimum_temperature_c']:.2f}°C")
    print(f"Maximum temperature: {validation_results['maximum_temperature_c']:.2f}°C")
    print(f"Cooling duration: {validation_results['cooling_duration_s']:.0f} s ({validation_results['cooling_duration_s']/3600:.2f} hrs)")
    print(f"Temperature trend: {validation_results['temperature_trend']}")
    print(f"Finite values: {validation_results['finite_value_result']}")
    print(f"No artificial oscillations: {validation_results['no_artificial_oscillations']} (sign changes: {validation_results['sign_changes_in_derivative']})")
    print(f"No impossible increases: {validation_results['no_impossible_increases']} ({validation_results['increasing_periods_count']} increasing periods)")
    print(f"Toward ambient: {validation_results['toward_ambient']}")
    print(f"Validation PASSED: {validation_results['validation_passed']}")

    return validation_results, df


def save_results(validation_results: Dict, df: pd.DataFrame, output_dir: Path):
    """Save validation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / "cooling_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved trajectory data to: {csv_path}")

    # Save markdown report
    md_path = output_dir / "cooling_validation.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Passive Cooling Validation Report\n\n")
        f.write("## Test Conditions\n")
        f.write("- Battery temperature > ambient\n")
        f.write("- Current ~ 0 A\n")
        f.write("- Vehicle stopped\n")
        f.write("- Ambient temperature < battery temperature\n\n")

        f.write("## Results\n")
        f.write(f"- **Initial temperature**: {validation_results['initial_temperature_c']:.2f}°C\n")
        f.write(f"- **Ambient temperature**: {validation_results['ambient_temperature_c']:.2f}°C\n")
        f.write(f"- **Final temperature**: {validation_results['final_temperature_c']:.2f}°C\n")
        f.write(f"- **Minimum temperature**: {validation_results['minimum_temperature_c']:.2f}°C\n")
        f.write(f"- **Maximum temperature**: {validation_results['maximum_temperature_c']:.2f}°C\n")
        f.write(f"- **Cooling duration**: {validation_results['cooling_duration_s']:.0f} s ({validation_results['cooling_duration_s']/3600:.2f} hours)\n")
        f.write(f"- **Temperature trend**: {validation_results['temperature_trend']}\n")
        f.write(f"- **Finite value result**: {validation_results['finite_value_result']}\n")
        f.write(f"- **No artificial oscillations**: {validation_results['no_artificial_oscillations']}\n")
        f.write(f"- **No impossible temperature increases**: {validation_results['no_impossible_increases']}\n")
        f.write(f"- **Temperature moving toward ambient**: {validation_results['toward_ambient']}\n\n")

        f.write(f"## Validation Result\n")
        f.write(f"**PASSED: {validation_results['validation_passed']}**\n\n")

        if validation_results['validation_passed']:
            f.write("The ECM demonstrates physically realistic passive cooling behavior:\n")
            f.write("- Temperature remains finite throughout simulation\n")
            f.write("- Temperature cools after load removal (zero current)\n")
            f.write("- Temperature moves toward ambient temperature\n")
            f.write("- No numerical divergence or artificial oscillations detected\n")
            f.write("- No impossible temperature increases during cooling phase\n")
        else:
            f.write("The ECM cooling behavior requires investigation:\n")
            if not validation_results['finite_value_result']:
                f.write("- Non-finite temperature values detected\n")
            if not validation_results['no_artificial_oscillations']:
                f.write("- Excessive artificial oscillations in temperature curve\n")
            if not validation_results['no_impossible_increases']:
                f.write("- Impossible temperature increases detected during cooling\n")
            if not validation_results['toward_ambient']:
                f.write("- Temperature not moving toward ambient as expected\n")
            if validation_results['minimum_temperature_c'] < validation_results['ambient_temperature_c'] - 1.0:
                f.write(f"- Temperature dropped significantly below ambient (more than 1.0°C)\n")

    print(f"Saved markdown report to: {md_path}")

    # Create and save plot
    plt.figure(figsize=(12, 8))

    # Temperature plot
    plt.subplot(2, 1, 1)
    plt.plot(df['time_s'] / 60, df['temperature_c'], 'b-', linewidth=2, label='Battery Temperature')
    plt.axhline(y=validation_results['ambient_temperature_c'], color='r', linestyle='--',
                label=f"Ambient Temperature ({validation_results['ambient_temperature_c']:.1f}°C)")
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title('Passive Cooling Validation: Battery Temperature vs Time')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Heat generation plot
    plt.subplot(2, 1, 2)
    plt.plot(df['time_s'] / 60, df['heat_generation_w'], 'g-', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Heat Generation (W)')
    plt.title('Heat Generation at Zero Current')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "cooling_validation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to: {plot_path}")


def main():
    """Main validation function."""
    print("RL-BMS Driving - Passive Cooling Validation")
    print("=" * 50)

    # Run validation
    validation_results, df = validate_passive_cooling()

    # Save results
    output_dir = Path("audit/driving_thermal_cooling_validation")
    save_results(validation_results, df, output_dir)

    print("\n" + "=" * 50)
    print("Validation complete!")


if __name__ == "__main__":
    main()