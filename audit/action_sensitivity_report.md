# Action-Sensitivity Audit Report

**Source Script**: [`experiments/action_sensitivity_test.py`](file:///c:/Users/siddh/Downloads/rl-bms-latest/rl-bms-Driving/experiments/action_sensitivity_test.py)  
**Raw Data**: [`audit/action_sensitivity_results.csv`](file:///c:/Users/siddh/Downloads/rl-bms-latest/rl-bms-Driving/audit/action_sensitivity_results.csv)  
**Date**: 2026-08-16

---

## Methodology

Five fixed raw actions (`-1.0, -0.5, 0.0, 0.5, 1.0`) were evaluated at 11 representative
battery states spanning the reachable charging distribution. For each (state, action) pair,
the following were recorded:
- `raw_action`, `clipped_action`
- `requested_current` (A), `safe_current_ceiling` (A), `applied_current` (A)
- `next_soc`, `next_temperature` (°C)
- `q_gen` (W), `reward`

States were chosen to cover:
- Bulk charging (low/mid SoC, normal temperature)
- High SoC with SoC-taper active
- High temperature with temperature derating active
- Both taper and derating simultaneously active

---

## Results

### Per-State Control Authority

| State | SoC | Temp (°C) | Safe Ceiling (A) | Ceiling Active? | Applied Range (A) | Unique Applied | Collapsed? |
|---|---|---|---|---|---|---|---|
| low_soc_low_temp | 0.15 | 20.0 | 160.0 | No | 0 – 160 | 5/5 | No |
| low_soc_mid_temp | 0.15 | 30.0 | 160.0 | No | 0 – 160 | 5/5 | No |
| mid_soc_low_temp | 0.50 | 20.0 | 160.0 | No | 0 – 160 | 5/5 | No |
| mid_soc_mid_temp | 0.50 | 30.0 | 160.0 | No | 0 – 160 | 5/5 | No |
| mid_soc_high_temp | 0.50 | 47.0 | 128.0 | **Yes** | 0 – 128 | 5/5 | No |
| high_soc_low_temp | 0.85 | 20.0 | 160.0 | No | 0 – 160 | 5/5 | No |
| high_soc_mid_temp | 0.85 | 30.0 | 160.0 | No | 0 – 160 | 5/5 | No |
| taper_zone | 0.92 | 25.0 | 128.0 | **Yes** | 0 – 128 | 5/5 | No |
| deep_taper | 0.96 | 25.0 | 64.0 | **Yes** | 0 – 64 | **3/5** | Partial |
| high_temp_derate | 0.50 | 52.0 | 48.0 | **Yes** | 0 – 48 | **3/5** | Partial |
| both_active | 0.93 | 48.0 | 112.0 | **Yes** | 0 – 112 | **4/5** | Partial |

### Summary Statistics

| Metric | Value |
|---|---|
| Total test states | 11 |
| States with complete collapse (all actions → same applied current) | **0 / 11** |
| States with partial collapse (fewer unique applied values than actions) | **3 / 11** |
| States with full control authority (all 5 actions → 5 distinct applied currents) | **8 / 11** |
| States where safety ceiling is active (ceiling < i_max) | **5 / 11** |

---

## Answers to Key Research Questions

### 1. Does action magnitude actually influence applied current?

**Yes.** In 8 of 11 states, all 5 action values produce distinct applied currents spanning the full
range from 0A to the safe ceiling (128–160A in bulk, 48–112A near limits). Even in the 3 partially
collapsed states (deep_taper, high_temp_derate, both_active), 3–4 of 5 actions still produce distinct
currents — collapse occurs only for the highest 2–3 actions that exceed the safety ceiling.

### 2. In what fraction of states does the safety layer erase action differences?

**0%** of states show complete erasure. The safety layer engages (ceiling < i_max) in 5/11 states,
but even when active, lower actions (below the ceiling) still produce distinct applied currents.
The safety layer only collapses actions that *exceed* the ceiling to the ceiling value.

### 3. In what fraction of states is the action clipped?

Action clipping (raw action outside [-1, 1]) is not observed in these tests since we use fixed
actions within the valid range. The *safety layer* clips applied current for actions exceeding
the ceiling in 5/11 states, but this is physical constraint enforcement, not action-space saturation.

### 4. What fraction of states are fully controlled by the safety supervisor?

**0%** of states are "fully controlled" in the sense that the safety supervisor determines the
outcome regardless of the agent's action. Even in the most constrained state (high_temp_derate:
SoC=0.50, T=52°C, ceiling=48A), the agent still has 3 distinct applied-current levels available.

---

## Interpretation

The charging environment provides **adequate control authority** for RL across the full reachable
state space. The safety layer acts as a ceiling constraint, not a total override — it limits the
maximum achievable current in thermally stressed or high-SoC states, but the agent retains the
ability to request currents below the ceiling and receive proportional applied-current responses.

The fact that PPO converges to Max Current is therefore **not caused by lack of control authority**.
It is caused by the reward formulation making maximum current the optimal action in every state
(progress reward dominates all penalties).
