# RL-BMS & EV Driving Energy Management (RL-BMS-Driving)

A scientifically grounded, physics-informed Reinforcement Learning framework for **EV Battery Charging Control** and **Driving Energy Management**, targeting the **Tata Nexon EV Long Range** platform (45 kWh usable, 121 Ah pack, 160 A DC Fast Charge, 300–420 V operating window).

---

## 1. Project Overview & Dual Research Tracks

The project investigates whether physics-constrained Reinforcement Learning (PPO) provides a measurable advantage over rule-based controllers under well-defined, safety-governed objectives.

```
                               rl-bms-Driving Framework
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
Track A: Charging BMS                                            Track B: Driving EMS
├── 1RC Thevenin ECM & Lumped Thermal                            ├── Longitudinal Vehicle Dynamics (Nexon EV)
├── Shared Supervisory Safety Layer (Taper/Derate)               ├── Drivetrain Efficiency & Regenerative Braking
├── PPO Fast-Charging Policy                                     ├── Bidirectional Safety Layer (Charge/Discharge)
├── Baselines: Max Current, CC (1C), CCCV, Adaptive              ├── Standard Drive Cycles (UDDS, HWFET, US06, WLTP)
└── Evaluated on 15 Standard Scenarios                           └── Rule-Based EMS vs Driving PPO Benchmark
```

---

## 2. Track A: Battery Charging BMS

### Battery Model (1RC Equivalent Circuit Model)
- **Cell Chemistry / Specs**: Large-format NMC cells in a 121 Ah pack configuration.
- **Electrical Dynamics**: $V_t(t) = \text{OCV}(\text{SoC}) + I(t) R_0 + V_{rc}(t)$, with polarization RC branch dynamics $\frac{dV_{rc}}{dt} = \frac{I}{C_1} - \frac{V_{rc}}{R_1 C_1}$.
- **Thermal Dynamics**: Joule heating $\dot{Q}_{\text{gen}} = I^2 R_0 + \frac{V_{rc}^2}{R_1}$, lumped convective dissipation $\dot{Q}_{\text{loss}} = h A (T - T_{\text{amb}})$.

### Safety Supervisory Layer
- Rule-based supervisor applied identically across all controllers:
  - Current ceiling enforcement ($I \le I_{\text{max}} = 160\text{ A}$)
  - Progressive thermal derating ($45^\circ\text{C}$ ramp to $55^\circ\text{C}$ hard cutoff)
  - Progressive voltage tapering ($415\text{ V}$ ramp to $420\text{ V}$ hard ceiling)
  - High-SoC saturation taper ($90\%$ SoC ramp to $95\%$ target)

### Verified Findings (`run_001` & Diagnostics)
- Baseline `runs/run_001/` (`CHARGING_PPO_BASELINE_1M`) is strictly preserved as historical evidence.
- PPO converges to behavior identical to Max Current charging under the 15-scenario grid because the progress reward ($+0.0918$/step per 40A) structurally dominates the thermal cost ($-0.0141$/step per 40A) by a 6.5:1 to 27:1 ratio.
- Action sensitivity tests confirm RL retains full control authority across 91% of charging steps; the safety supervisor only binds during the final 9% (SoC taper).

---

## 3. Track B: Driving Energy Management (EMS)

### Vehicle & Drivetrain Dynamics (Tata Nexon EV)
- **Longitudinal Dynamics**: Aerodynamic drag ($C_d = 0.32$, $A_f = 2.42\text{ m}^2$), rolling resistance ($C_{rr} = 0.012$), grade resistance, and inertial force ($m = 1400\text{ kg}$).
- **Drivetrain & Regen**: Motor peak power $106.4\text{ kW}$, max regen power $25.0\text{ kW}$, combined motor/inverter efficiency mapping (90% nominal).
- **Bidirectional Safety**: Constrains motor draw during low-voltage/low-SoC conditions and limits regenerative braking during cold battery or high-voltage states.

### Standard Drive Cycles Repository (`data/drive_cycles/standard/`)
All driving cycles are official regulatory test schedules sampled at 1.0 Hz matching the simulation timestep:
- **EPA UDDS** (`data/drive_cycles/standard/epa_udds/cycle.csv`): 1,372 s, 10.42 km, urban stop-and-go.
- **EPA HWFET** (`data/drive_cycles/standard/epa_hwfet/cycle.csv`): 765 s, 16.70 km, highway cruising.
- **EPA US06** (`data/drive_cycles/standard/epa_us06/cycle.csv`): 596 s, 12.28 km, high acceleration / aggressive.
- **WLTP Class 3b** (`data/drive_cycles/standard/wltp_class3b/cycle.csv`): 1,800 s, 23.44 km, international mixed.

---

## 4. How to Run Tests & Experiments

### 1. Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run complete unit test suite (193 tests)
python -m pytest tests/ -v
```

### 2. Track A: Charging Diagnostics & Baselines
```bash
# Run Action-Sensitivity test (control authority)
python -m experiments.action_sensitivity_test

# Run Safety-Dominance quantification (N >= 5,000 steps)
python -m experiments.safety_dominance_test

# Run Baseline Reward Decomposition & Thermal Engagement test
python -m experiments.baseline_reward_comparison

# Run 3-Seed Charging A/B Diagnostic (seeds 7, 21, 42)
python -m experiments.diagnostic_ab
```

### 3. Track B: Driving EMS Evaluation & PPO Training
```bash
# Validate all standard drive cycles
python -m experiments.validate_drive_cycles

# Benchmark Rule-Based EMS across all cycles
python -m training.evaluate_drive_ems --controller rule_based --all-cycles

# Run 3-Seed Driving PPO Diagnostic & Benchmark (seeds 7, 21, 42)
python -m experiments.diagnostic_driving_ppo
```

---

## 5. Audit & Research Reports

Complete empirical reports and verified datasets are located in `audit/`:
- `audit/run001_diagnosis.md`: Preserved baseline diagnosis for run_001.
- `audit/charging_reward_balance.md`: Empirical Stage 2 reward component decomposition.
- `audit/action_sensitivity_report.md`: Control authority verification across 11 battery states.
- `audit/safety_dominance_report.md`: Safety ceiling active percentage (8.99%) analysis.
- `audit/diagnostic_ab_results.md`: Complete 3-seed A/B diagnostic findings.
- `audit/real_drive_cycle_validation.md`: Kinematic and mathematical validation of standard drive schedules.
- `audit/driving_reward_balance.md`: Empirical driving reward balance from multi-cycle execution.
- `audit/final_project_validation.md`: Final synthesis and scientific claims audit.

---

## 6. Known Limitations (Phase 1)

1. **Open-Loop Prescribed Speed**: The driving EMS follows the prescribed vehicle velocity trace. If electrical power is constrained, a power deficit is recorded and penalized; closed-loop vehicle speed adaptation is out of scope for Phase 1.
2. **Lumped Convective Thermal Model**: Both charging and driving tracks model battery heat dissipation via lumped convective approximation rather than an active liquid coolant loop.
3. **SoH Tracking**: Battery degradation is tracked via coulomb-throughput accumulation for logging only; SoH reward optimization remains deliberately disabled.

## Final frozen training profiles

The repository contains explicit frozen configuration bundles so long training cannot accidentally use the development configs.

Charging (Track A):

```powershell
python -m training.train --run-name charging_final_1m --config-dir configs/final_charging --stages 1 2 3 4
```

Driving (Track B):

```powershell
python -m training.train_drive_ems --train --run-name driving_final_1m --config-dir configs/final_driving --drive-cycle data/drive_cycles/standard/wltp_class3b/cycle.csv --timesteps 1000000 --seed 7
```

Use a separate run name/seed for each final driving seed. The final profiles are frozen copies of the validated Candidate A1/B3 configurations.
