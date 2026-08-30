# RL-BMS & EV Driving Energy Management (RL-BMS-Driving)

A scientifically grounded, physics-informed Reinforcement Learning (PPO) framework for **EV Battery Charging Control** and **Driving Energy Management (EMS)**, targeting the **Tata Nexon EV Long Range** platform (45 kWh usable, 121 Ah pack, 160 A DC Fast Charge, 300–420 V operating window).

---

## 1. Project Overview & Architecture

The framework investigates whether physics-constrained Reinforcement Learning (PPO) provides a measurable advantage over rule-based controllers under well-defined, safety-governed automotive objectives.

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
└── Evaluated on 15 Standard Scenarios                           ├── 9-State Thermal Safety Controller
                                                                 └── Interactive Pygame Engineering Visualizer
```

---

## 2. Key Features

- **Authoritative Battery Equivalent Circuit Model (ECM)**:
  - 1RC Thevenin electrical dynamics: $V_t(t) = \text{OCV}(\text{SoC}) + I(t) R_0 + V_{rc}(t)$ with polarization RC branch dynamics.
  - Lumped thermal dissipation with ambient coupling: $\dot{Q}_{\text{gen}} = I^2 R_0 + \frac{V_{rc}^2}{R_1}$, $\dot{Q}_{\text{loss}} = h A (T - T_{\text{amb}})$.
- **Bidirectional Automotive Safety Layer**:
  - Enforces continuous current ceilings ($I \le 160\text{ A}$), progressive thermal derating ($45^\circ\text{C} \to 55^\circ\text{C}$ cutoff), and voltage bounds ($300\text{ V} \le V_t \le 420\text{ V}$).
  - Dynamically calculates power deficit and mechanical friction braking intervention.
- **Authoritative 9-State Thermal State Machine**:
  - `OPTIMAL` $\to$ `ELEVATED_THERMAL` $\to$ `DERATING_ACTIVE` $\to$ `CRITICAL` $\to$ `STOP_REQUESTED` $\to$ `DECELERATING` $\to$ `STOPPED` $\to$ `COOLING` $\to$ `SAFE_TO_RESUME`.
  - State-relative hysteresis prevention and speed recommendation calculations.
- **Professional Pygame Interactive Simulator**:
  - 24px unified grid layout, Tesla/Grafana-style numeric unit de-emphasis, colorblind-accessible icon badges, and live oscilloscope telemetry traces.
  - Seamless dual-mode operation: **Research Benchmark Mode** (pure frozen standard cycles) vs. **Demo Mode** (interactive safety stops & ECM cooling).
- **Comprehensive Test Suite**:
  - **261 / 261 automated unit and integration tests** passing with 100% coverage across dynamics, rewards, safety layers, and visualizer components.

---

## 3. Standard Regulatory Drive Cycles

The environment includes official 1.0 Hz regulatory driving test schedules in `data/drive_cycles/standard/`:

| Cycle | Description | Duration | Distance | Target Profile |
| :--- | :--- | :--- | :--- | :--- |
| **EPA UDDS** | Urban Dynamometer Driving Schedule | 1,372 s | 10.42 km | City stop-and-go |
| **EPA HWFET** | Highway Fuel Economy Test | 765 s | 16.70 km | Smooth highway cruising |
| **EPA US06** | Supplemental FTP (Aggressive) | 596 s | 12.28 km | High acceleration & high power demand |
| **WLTP Class 3b** | Worldwide Harmonised Light Vehicle Test | 1,800 s | 23.44 km | Dynamic multi-phase mixed driving |

---

## 4. Quick Start & Execution

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/SIDDHARTH0424/ML_BMS_SYSTEM.git
cd ML_BMS_SYSTEM

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Interactive Pygame Simulator
```powershell
python -m app.interactive_ev_simulator
```

**Simulator Controls**:
- `PLAY / PAUSE`: Toggle live simulation execution.
- `STEP`: Advance single simulation step ($1.0\text{ s}$).
- `RESET`: Reset episode to initial conditions.
- `DRIVING / CHARGING`: Switch between Driving EMS and DC Fast-Charging modes.
- `RESEARCH / DEMO`: Toggle pure benchmark evaluation vs. interactive thermal safety stop demo.
- `PPO / BASELINE`: Toggle between pre-trained PPO neural policy and baseline rule-based controller.
- `UDDS / HWFET / US06 / WLTP`: Cycle through standard driving schedules.
- `STOP / RESUME`: Trigger vehicle safety deceleration and resume once cooled below $42.0^\circ\text{C}$.
- `Ambient [+] / [-]`: Adjust ambient temperature ($0^\circ\text{C} \to 45^\circ\text{C}$).
- `Sim Speed [+] / [-]`: Adjust playback speed multiplier ($0.5\times \to 10\times$).

---

## 5. Verification & Testing

### Run Complete Test Suite (261 Tests)
```powershell
pytest tests/ -v
```

### Run Master 10-Phase Project Verification
```powershell
python scripts/verify_project.py
```

### Run Multi-Cycle Benchmark Evaluation
```powershell
# Run benchmark sweep across all 4 drive cycles with multi-seed PPO models
python run_final_evaluation.py

# Aggregate statistical results
python aggregate_results.py
```

---

## 6. Pre-Trained Models (`final_models/`)

Pre-trained, fully verified model weights are included directly in the repository:

- **Driving EMS (Track B)**:
  - `final_models/driving_B3_100k_seed7/ppo_driving_100000_steps.zip`
  - `final_models/driving_B3_100k_seed21/ppo_driving_100000_steps.zip`
  - `final_models/driving_B3_100k_seed42/ppo_driving_100000_steps.zip`
- **Charging BMS (Track A)**:
  - `final_models/charging_A1_50k_seed7/trained_model.zip`
  - `final_models/charging_A1_50k_seed21/trained_model.zip`
  - `final_models/charging_A1_50k_seed42/trained_model.zip`

---

## 7. Audit Reports & Research Documentation

Detailed empirical reports, verification proofs, and scientific analysis documents are located in:
- `COMPLIANCE.md`: Comprehensive audit compliance matrix.
- `FINAL_OUTPUT_SUMMARY.md`: Master Claude Code Task executive summary.
- `FINAL_VERIFICATION_SUMMARY.md`: Full 10-phase verification logs.
- `results_and_discussion.md`: Quantitative multi-cycle benchmark analysis.
- `audit/FINAL_VERIFICATION/FINAL_VERIFICATION_REPORT.md`: Authoritative verification report.

---

## 8. License

Distributed under the MIT License. See `LICENSE` for more information.
