# Driving Reward Balance Audit

**Controller**: `PPO_ppo_driving_100000_steps.zip`  
**Evaluation Type**: Standardized drive-cycle evaluation of a simulated Tata Nexon EV Long Range  
**Cycles Evaluated**: EPA UDDS (Urban), EPA HWFET (Highway), EPA US06 (Aggressive), WLTP Class 3b (Mixed)  
**Total Evaluated Steps**: 4,529  

---

## 1. Empirical Reward Component Distributions

| Component | Mean | Std | Min | Max | % Contribution |
|---|---|---|---|---|---|
| `tracking_error` | 0.000042 | 0.000642 | 0.000000 | 0.026948 | **0.05%** |
| `energy_cost` | 0.067054 | 0.080217 | 0.000000 | 0.505781 | **82.69%** |
| `regen_recovery` | 0.013991 | 0.041557 | 0.000000 | 0.570314 | **17.25%** |
| `thermal_stress` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **0.00%** |
| `safety_penalty` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **0.00%** |

---

## 2. Total Reward Statistics

- **Mean per-step reward**: -0.013184
- **Std**: 0.034003
- **Min**: -0.151734, **Max**: 0.244956

## 3. Findings

- **Active Terms**: Under nominal standardized drive cycles, the practical reward signal is dominated by energy consumption and regenerative recovery.
- **Inactive Terms**: Thermal stress, safety penalties, and tracking error remain inactive because nominal power demands are within physical and safety envelopes.
- **Regenerative Incentive**: Capturing available regenerative energy strictly increases per-step reward over letting kinetic energy dissipate into friction braking.
