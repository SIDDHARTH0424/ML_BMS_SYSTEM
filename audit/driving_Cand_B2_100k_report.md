# Track B — Candidate B2, Extended Budget (100,000 steps/seed)

**Purpose**: At 50k steps, Candidate B2 (`w_regen_recovery=1.2, w_energy_cost=0.2,
w_tracking_error=1.5, ent_coef=0.010`) was the best-performing driving
candidate but did not clear the strict gate: cross-seed mean 129.78 Wh/km
(vs 129.16 gate) and regen recovery 83.9-96.0% (seed 42 only cleared 85%).
Seeds 7 and 21 appeared to still be exploring the regen action space at
50k steps.

Per master-prompt Track B Step 7 (50k-100k timesteps authorized) and
Absolute Rule 8 (isolate one mechanism at a time), this experiment changes
**only the timestep budget** (50k -> 100k) for the identical B2 reward
weights and `ent_coef` — no reward-weight change is conflated with the
budget change.

**Seeds**: 7, 21, 42. **Timesteps**: 100,000/seed. **Training cycle**: WLTP
Class 3b (same as original B1/B2 runs). **Evaluation**: EPA UDDS, EPA
HWFET, EPA US06, WLTP Class 3b (same 4 standard cycles).

Outputs: `audit/driving_Cand_B2_100k_benchmark.csv` (per-cycle),
`audit/driving_Cand_B2_100k_summary.csv` (per-seed), checkpoints in
`runs/driving_Cand_B2_100k/seed_{7,21,42}/`. Does not overwrite the
original Cand_B1/Cand_B2 50k artifacts.

## Results vs Rule-Based EMS (129.16 Wh/km cross-cycle mean, 100% regen)

| Seed | Cross-cycle Wh/km | Δ vs 129.16 | Regen Recovery | Cycles Beating Rule-Based | Safety Interventions | Max \|Δcycle\| |
|---|---|---|---|---|---|---|
| 7  | **129.05** | **-0.11** | **98.9%** | **4/4** | 0 | +0.7% (WLTP) |
| 21 | 129.46 | +0.30 | 94.9% | 0/4 | 0 | +0.7% (UDDS/HWFET) |
| 42 | 129.21 | +0.05 | 97.9% | 1/4 | 0 | +0.2% (all cycles) |
| **Mean** | **129.24** | **+0.08** | **97.2%** | — | 0 | — |

Comparison to the 50k-step B2 run: cross-seed mean improved from
129.78 -> 129.24 Wh/km (gap to gate shrank from +0.62 to +0.08 Wh/km,
~87% closed); mean regen recovery improved from 90.1% -> 97.2%; zero
safety interventions and zero power-deficit anomalies at any seed/cycle
(same as before). No individual cycle differs from Rule-Based by more
than 1.0 Wh/km (<1%) at any seed — condition B6 (no cycle >10% worse)
passes comfortably for all seeds, including the two that miss the
overall gate.

## Gate Evaluation (Track B Step 8, requires ≥2/3 seeds)

| Condition | Seed 7 | Seed 21 | Seed 42 | Seeds passing | Verdict (need ≥2/3) |
|---|---|---|---|---|---|
| 1. Cross-cycle Wh/km ≤ 129.16 | PASS | FAIL (+0.30) | FAIL (+0.05) | 1/3 | **FAIL** |
| 2. PPO ≤ Rule-Based on ≥3/4 cycles | PASS (4/4) | FAIL (0/4) | FAIL (1/4) | 1/3 | **FAIL** |
| 3. Regen recovery > 85% | PASS | PASS | PASS | 3/3 | PASS |
| 4. No safety violations | PASS | PASS | PASS | 3/3 | PASS |
| 5. Power deficit not materially worse | PASS | PASS | PASS | 3/3 | PASS |
| 6. No cycle >10% worse | PASS | PASS | PASS | 3/3 | PASS |
| 7. Final SOC / max temp valid | PASS | PASS | PASS | 3/3 | PASS |
| 8. Training stable | PASS (no NaN/Inf observed) | PASS | PASS | 3/3 | PASS |

**Conditions 1 and 2 are the binding constraints** and only 1/3 seeds
clear them. Per Absolute Rule 9 ("do not call a single-seed improvement
a success") and Step 8's explicit "do not call one good cycle a
success," this candidate/budget combination does **not** pass the
strict gate, despite the large, real improvement.

## Conclusion

**`Track B = NOT_READY`** (unchanged classification), but materially
closer than the 50k-step result. The remaining gap for seeds 21/42 is
small (+0.30 and +0.05 Wh/km, i.e. ≤0.25%) and driven entirely by
slightly lower regen recovery than seed 7 (94.9%/97.9% vs 98.9%), not by
any safety, power-delivery, or thermal issue — all secondary gates pass
cleanly for all three seeds. 100,000 steps is the top of the budget
range explicitly authorized in Track B Step 7; going further is a
deviation from the stated plan and is not applied here without
explicit sign-off.

**Per Absolute Rule 10, the 129.16 Wh/km / ≥2/3-seed gate is not
loosened to declare success.**
