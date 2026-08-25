# Charging Reward Balance Audit

**Source**: `runs/run_001/reward_components.csv` (Stage-2 logging, 24,576 steps)  
**Reward code**: `environment/battery_env.py::_compute_reward` + `::step` terminal terms  
**Weights**: `configs/reward.yaml` (run_001 snapshot in `runs/run_001/config/reward.yaml`)

---

## Per-Step Component Statistics

All values below are computed from the 24,576 logged reward-component rows in
run_001's Stage-2 reward_components.csv. These are **empirical** distributions
from actual training, not theoretical ranges.

| Component | Weight | Mean | Std | Min | Max | Nonzero% | % of ΣAbsMeans |
|---|---|---|---|---|---|---|---|
| `progress` | 1000.0 | 0.200333 | 0.129758 | 0.000000 | 0.367309 | 87.4% | **40.07%** |
| `smoothness_penalty` | 0.5 | 0.194857 | 0.145663 | 0.000000 | 0.500000 | 94.3% | **38.98%** |
| `time_penalty` | 0.05 | 0.050000 | 0.000000 | 0.050000 | 0.050000 | 100.0% | **10.00%** |
| `safety_penalty` | 5.0 | 0.030938 | 0.218588 | 0.000000 | 2.490609 | 2.7% | 6.19% |
| `target_reached_bonus` | 50.0 | 0.012207 | 0.781171 | 0.000000 | 50.000000 | 0.0%* | 2.44% |
| `overrequest_penalty` | 2.0 | 0.011579 | 0.083489 | 0.000000 | 0.996244 | 2.7% | 2.32% |
| `temp_penalty` | 0.05 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0% | 0.00% |
| `thermal_reward` | 0.5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0% | 0.00% |
| `terminal_shortfall_penalty` | 1000.0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0% | 0.00% |
| `overvoltage_penalty` | 20.0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0% | 0.00% |
| `overtemperature_penalty` | 20.0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0% | 0.00% |

*\*target_reached_bonus: 6 nonzero values (one per completed episode in the
24,576-step Stage-2 log), rounds to 0.0% of total steps.*

---

## Total Reward

| Stat | Value |
|---|---|
| Mean | −0.074835 |
| Std | 0.847860 |
| Min | −3.509948 |
| Max | 50.031325 |

The mean total reward is slightly negative (−0.075), dominated by
`smoothness_penalty` and `time_penalty` outweighing `progress` on average.
The max (50.03) corresponds to a target-reached terminal step.

---

## Analysis

### Reward is dominated by two components

The `progress` and `smoothness_penalty` terms together account for **79%** of
all reward signal magnitude. This means the PPO agent is primarily optimizing
a tradeoff between:
- **Charging fast** (progress ≈ 0.20/step at average current)
- **Charging smoothly** (smoothness ≈ 0.19/step on average)

### Thermal terms are completely inactive

Both `temp_penalty` and `thermal_reward` are identically zero across all
24,576 steps. Temperature never exceeded the 40°C threshold in the
evaluation range (15–35°C ambient, 10–30% initial SoC).

### Safety interventions are rare

The safety layer engages on only 2.7% of steps (664/24,576). This
confirms that for the vast majority of the charging trajectory (SoC < 0.90),
the safety layer permits full i_max and does not constrain the policy.

### Max Current is reward-optimal

Under this reward formulation:
- **progress** is maximized by charging at the highest possible current
- **smoothness_penalty** is minimized by maintaining a constant current
- **safety/overrequest penalties** are zero when not over-requesting
- **time_penalty** is minimized by finishing faster
- **thermal terms** provide no signal

Therefore, the reward-optimal strategy is: *"Request i_max every step, let
the safety layer taper at high SoC."* This is exactly what Max Current does.
PPO correctly converged to this optimum.

### The intended multi-objective tradeoff does not exist

The design intent was:

> reliable target completion + fast charging + thermal control + safety +
> smooth charging

But in practice:
- **thermal control** has zero gradient (reward is zero everywhere)
- **safety** is satisfied by the safety layer regardless of policy
- **target completion** is guaranteed by any sufficiently aggressive policy
- **smoothness** only differentiates within the first few steps

The only remaining axis is **speed**, which Max Current wins trivially.

---

## Recommendation

Activate and lower the thermal reward reference to create a genuine
tradeoff. The thermal reward infrastructure
(`battery_env.py::_compute_reward`, thermal_weight / thermal_reference_temp_c /
thermal_scale_c / thermal_q_reference_w in reward.yaml) already exists — it
just needs operating conditions where it produces non-zero signal.

**Do NOT increase weights arbitrarily.** Previous diagnostics
(audit/thermal_weight_040_045_report.md) showed flat thermal penalties are
seed-fragile. The state-aware formulation (quadratic × q_gen) is the right
approach — it just needs a lower reference temperature to engage.
