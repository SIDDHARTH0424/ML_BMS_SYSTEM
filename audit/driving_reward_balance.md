# Driving Reward Balance Audit

**Controller**: `PPO_ppo_driving_100000_steps`  
**Evaluation Type**: Standardized drive-cycle evaluation of a simulated Tata Nexon EV Long Range  
**Cycles Evaluated**: EPA UDDS (Urban), EPA HWFET (Highway), EPA US06 (Aggressive), WLTP Class 3b (Mixed)  
**Total Evaluated Steps**: 4,529  

---

## 1. Empirical Reward Component Distributions

| Component | Mean | Std | Min | Max | % Contribution |
|---|---|---|---|---|---|
| `tracking_error` | 0.000259 | 0.003358 | 0.000000 | 0.099442 | **0.32%** |
| `energy_cost` | 0.066901 | 0.080269 | 0.000000 | 0.505781 | **82.17%** |
| `regen_recovery` | 0.014263 | 0.041565 | 0.000000 | 0.570314 | **17.52%** |
| `thermal_stress` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **0.00%** |
| `safety_penalty` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **0.00%** |

---

## 2. Total Reward Statistics

- **Mean per-step reward**: 0.006200
- **Std**: 0.062891
- **Min**: -0.149164, **Max**: 0.771639

## 3. Findings

- **Active Terms**: Under nominal standardized drive cycles, the practical reward signal is dominated by energy consumption and regenerative recovery.
- **Inactive Terms**: Thermal stress, safety penalties, and tracking error remain inactive because nominal power demands are within physical and safety envelopes.
- **Regenerative Incentive**: Capturing available regenerative energy strictly increases per-step reward over letting kinetic energy dissipate into friction braking.
