# Run_001 Diagnosis Report

**Run ID**: `run_001`  
**Label**: `CHARGING_PPO_BASELINE_1M`  
**Status**: Preserved — DO NOT delete, overwrite, or modify.

---

## Summary

Run_001 completed a 1,000,000-step PPO training run using the charging
environment (`environment/battery_env.py`) with the reward formulation in
`configs/reward.yaml` (thermal_enabled=false) and the shared safety layer
in `configs/safety.yaml`.

The trained model and all artifacts (checkpoints, TensorBoard logs,
reward_components.csv, evaluation metrics) are preserved in `runs/run_001/`.

---

## Required Findings

### 1. PPO completed 1M steps

Confirmed. Stage 4 ran to completion (1,000,000 timesteps). Model saved as
`runs/run_001/trained_model.zip` (`trained_model_stage4.zip`).

### 2. PPO reached the charging target

Confirmed. Across all 15 evaluation scenarios (5 initial SoCs × 3 ambient
temperatures), `target_reached = True` and `target_shortfall = 0.0` for
every scenario.

### 3. PPO matched Max Current

Confirmed. The trained PPO policy is **behaviorally indistinguishable** from
the trivial Max Current controller across all evaluation metrics:

| Metric | PPO (mean ± std) | Max Current (mean ± std) |
|---|---|---|
| charging_time_s | 2094.6 ± 199.5 | 2094.6 ± 199.5 |
| peak_temperature_c | 32.26 ± 8.47 | 32.26 ± 8.47 |
| final_soc | 0.9501 ± 0.0001 | 0.9501 ± 0.0001 |
| safety_interventions | 188.4 ± 0.5 | 188.4 ± 0.5 |
| energy_efficiency | 0.9807 ± 0.0001 | 0.9807 ± 0.0001 |
| target_reached | 1.000 | 1.000 |

PPO ≈ Max Current under the current charging problem formulation.

### 4. Thermal reward was zero in the Stage-2 log

Confirmed. Analysis of `runs/run_001/reward_components.csv` (24,576 logged
steps from Stage 2):

- `thermal_reward`: mean=0.000000, std=0.000000, min=0.000000, max=0.000000
- Non-zero count: 0 out of 24,576 steps (0.0%)

The thermal reward infrastructure was correctly configured (`thermal_enabled:
false` in the production reward.yaml for run_001), so zero thermal reward is
expected. However, even with the existing state-aware thermal reward
formulation enabled, the 40°C reference temperature (`thermal_reference_temp_c`)
matches the `temperature_penalty_start_c` threshold, and typical charging
trajectories starting from 15–35°C ambient never exceed 40°C, so the thermal
term would have remained zero regardless.

### 5. Thermal-aware optimization was NOT demonstrated

Confirmed. Since thermal_reward = 0 throughout all logged training steps, the
1M-step run did NOT meaningfully train a thermal-aware policy. The existing
thermal reward infrastructure is not the issue — the thermal objective simply
remained inactive under the current 40°C reference.

### 6. Next version changes the objective/reward formulation

The correct interpretation of run_001 is:

> "PPO successfully trained and reached the charging target, but under the
> current environment, reward formulation, action interface, and safety
> constraints, the learned policy converged to behavior indistinguishable
> from the Max Current baseline."

This is a diagnostic baseline. The next phase changes the objective/reward
formulation (lowering the thermal optimization reference to engage the
state-aware thermal reward) rather than simply increasing PPO training time
or modifying the PPO algorithm.

---

## Reward Balance (Stage-2 Log)

From `runs/run_001/reward_components.csv` (24,576 steps):

| Component | Mean | Std | Min | Max | % Contribution |
|---|---|---|---|---|---|
| progress | 0.2003 | 0.1298 | 0.0000 | 0.3673 | 40.07% |
| smoothness_penalty | 0.1949 | 0.1457 | 0.0000 | 0.5000 | 38.98% |
| time_penalty | 0.0500 | 0.0000 | 0.0500 | 0.0500 | 10.00% |
| safety_penalty | 0.0309 | 0.2186 | 0.0000 | 2.4906 | 6.19% |
| target_reached_bonus | 0.0122 | 0.7812 | 0.0000 | 50.0000 | 2.44% |
| overrequest_penalty | 0.0116 | 0.0835 | 0.0000 | 0.9962 | 2.32% |
| temp_penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| thermal_reward | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| terminal_shortfall | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| overvoltage_penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| overtemperature_penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% |

**Key observations:**
- The reward is dominated by just two components: `progress` (40%) and
  `smoothness_penalty` (39%).
- Safety-related penalties fire on only 2.7% of steps (664 out of 24,576).
- Temperature never exceeds `temperature_penalty_start_c` (40°C), so
  `temp_penalty` is always zero.
- No thermal, voltage, or temperature-related terminations occurred.

---

## What This Means

The current charging problem under these conditions is **effectively trivial**
for RL: maximum-current charging + safety layer tapering + progress reward
is sufficient to produce the best achievable behavior. PPO correctly found
this optimum — the problem is that the optimum is indistinguishable from
the simplest possible controller.

The fix is to make the optimization problem non-trivial by activating and
lowering the thermal reward reference, creating a meaningful fast-charging
vs. thermal-management tradeoff that a constant max-current strategy cannot
optimally resolve.
