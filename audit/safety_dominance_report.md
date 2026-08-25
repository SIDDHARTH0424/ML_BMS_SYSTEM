# Safety-Layer Dominance Audit Report

**Source Script**: [`experiments/safety_dominance_test.py`](file:///c:/Users/siddh/Downloads/rl-bms-latest/rl-bms-Driving/experiments/safety_dominance_test.py)  
**Raw Data**: [`audit/safety_dominance_results.csv`](file:///c:/Users/siddh/Downloads/rl-bms-latest/rl-bms-Driving/audit/safety_dominance_results.csv)  
**Date**: 2026-08-16  
**Sample Size**: N = 31,419 organically reachable steps across 15 evaluation scenarios (5 initial SoCs × 3 ambient temperatures)

---

## Methodology

Two controllers were evaluated:
1. **Max Current Baseline**: Requests 160A (i_max) every step.
2. **PPO Baseline (run_001)**: Trained 1M-step model from `runs/run_001/trained_model.zip`.

For each step, the safety layer's clamping behavior was recorded:
- Whether the safe ceiling was active (below i_max)
- Whether the request exceeded the ceiling
- Whether the applied current equaled the ceiling
- Requested and applied currents

---

## Results

### Max Current Baseline (N = 31,419 steps)

| Metric | Value |
|---|---|
| $P(\text{safety ceiling active})$ | **8.99%** |
| $P(\text{requested} > \text{safe ceiling})$ | **8.99%** |
| $P(\text{requested} \le \text{ceiling} \mid \text{ceiling active})$ | **0.00%** |
| $P(\text{applied} == \text{safe ceiling})$ | **100.00%** |
| Mean requested current | **160.00 A** |
| Mean applied current | **155.99 A** |

### PPO Baseline — run_001 (N = 31,419 steps)

| Metric | Value |
|---|---|
| $P(\text{safety ceiling active})$ | **8.99%** |
| $P(\text{requested} > \text{safe ceiling})$ | **8.99%** |
| $P(\text{requested} \le \text{ceiling} \mid \text{ceiling active})$ | **0.00%** |
| $P(\text{applied} == \text{safe ceiling})$ | **100.00%** |
| Mean requested current | **160.00 A** |
| Mean applied current | **155.99 A** |

---

## Interpretation

### How much actual control authority does PPO have?

1. **For 91.01% of the charging trajectory**, the safety ceiling equals i_max (160A) — the safety
   layer is not binding and does not constrain the policy. In this region, PPO has full control
   authority over applied current (0A to 160A).

2. **For 8.99% of the trajectory** (exclusively in the high-SoC taper zone, SoC > 0.90), the safety
   ceiling drops below i_max. In this region, both Max Current and PPO request above the ceiling,
   so the safety supervisor determines the applied current entirely.

3. **PPO's behavioral identity with Max Current is NOT caused by safety-layer dominance.** The
   safety layer only constrains the final 8.99% of steps. For the vast majority of the episode,
   PPO voluntarily chooses to request maximum current because the reward formulation makes this
   optimal.

### Key Research Question

> "How much of the actual charging trajectory is controlled by PPO vs. the safety supervisor?"

**Answer**: PPO has full decision authority over 91% of the trajectory. The safety supervisor
controls only the final 9% (high-SoC taper). PPO's convergence to Max Current is a reward
formulation issue, not a control authority issue.

---

## Implications for Reward Redesign

Since the safety layer is only active for 8.99% of steps:
- **Bulk charging (91%)**: PPO must be incentivized by the reward to reduce current below i_max.
  Currently, progress reward (+0.358/step) dominates thermal penalty (-0.013/step), so max current
  is always reward-optimal.
- **Taper zone (9%)**: The safety layer handles derating correctly regardless of the policy.
  No reward redesign is needed here.

The redesigned reward must create a meaningful tradeoff **in the bulk charging region** where
the agent has full control authority.
