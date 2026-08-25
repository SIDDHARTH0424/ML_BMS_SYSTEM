# RL-BMS-Driving: Final Project Validation Report (v2)

**Project**: rl-bms-Driving
**Vehicle**: Tata Nexon EV Long Range (simulated)
**Timestamp**: 2026-08-17T00:00:00Z
**Overall Project Status**: `NOT_READY_FOR_LONG_TRAINING` (revised — see note below; this document originally read READY_FOR_LONG_TRAINING)

> [!WARNING]
> **Track A status revised after this document was originally
> written.** `audit/charging_stage4_collapse_investigation.md` found
> that the real long-training entry point (`agents/train_ppo.py::build_agent()`)
> used a decaying learning-rate schedule the Track A short-gate
> diagnostic never tested (constant LR only) — confirmed, reproduced,
> and corroborated by drift precursors observed through 450k of
> 1,000,000 steps. **That mismatch has since been fixed and
> re-verified** — the 3-seed short gate passes again through the real
> pipeline (`audit/charging_A1_regate_flat_lr_fixed_report.md`) — but
> **Track A remains `NOT_READY`**: no full 1,000,000-step run has been
> executed against the fixed pipeline yet. Track B is unaffected.
> Section 7/9 below reflect the ORIGINAL (now superseded) READY verdict
> for Track A and are retained for evidence continuity — read
> `audit/charging_stage4_collapse_investigation.md` for the current
> status.

**This document supersedes the original `audit/final_project_validation.md`**,
which was written before Candidates A1 (Track A) and B3 (Track B)
existed and reported historical Experiment C / Stage-Q results as the
primary final results — creating an internal contradiction (header
said `READY_FOR_LONG_TRAINING`, body sections concluded
`NEEDS_CHARGING_OBJECTIVE_REVISION` / `NEEDS_DRIVING_REWARD_REVISION` /
`NOT_READY_FOR_LONG_TRAINING`). The original is preserved unmodified
at `audit/final_project_validation_HISTORICAL_pre_A1_B3.md` for
evidence continuity; it should not be read as current status.

---

## Section 1 — Software Validation

### 1.1 Compilation Check
```
python -m compileall . -q
Exit code: 0 (0 syntax errors across all Python modules)
```

### 1.2 Full Test Suite
```
213 passed, 0 failed, 0 skipped, 0 errors
```
Source: `audit/long_training_freeze_tests_v2.txt` (post config_dir-fix
regression) — reconfirmed identical count after the Part 2 driving
config-snapshot fix in this session.

### 1.3–1.4 Ambient Instrumentation / Sampler Validation

Unchanged from the original report (Sections 1.3–1.4) — these validate
environment/test infrastructure, not any specific candidate, and
remain accurate. See
`audit/final_project_validation_HISTORICAL_pre_A1_B3.md` §1.3–1.4 for
the full ambient-logging chain and the N=2000 Exp-C sampler validation
(p_stress = 0.2490 vs target 0.25).

---

## Section 2 — Track A: Final Candidate (A1) Results

**Full detail**: `audit/charging_final_candidate_evaluation.md` (authoritative).

Historical Experiment C (the pre-candidate diagnostic run) showed
genuine thermal derating but failed the ±5% charging-time gate for
2/3 seeds (Seed 7: +20.6%, Seed 21: +5.1%). Candidate A1
(`thermal_reference_temp_c=35.0, thermal_weight=2.5, thermal_scale_c=20.0`)
was derived from the analytical reward-landscape screen
(`audit/charging_tradeoff_analysis.md`) specifically to fix this.

| Seed | Charging Time (s) | Δ vs 2094.6s Baseline | vs 2199.3s Gate | Peak Temp, 45°C Stress (°C) |
|---|---|---|---|---|
| 7 | 2114.53 | +0.95% | PASS | 49.29 |
| 21 | 2095.00 | +0.02% | PASS | 49.60 |
| 42 | 2095.13 | +0.03% | PASS | 49.60 |

Max Current baseline peak temp at 45°C stress: 50.75°C — A1 beats it
by 1.15–1.46°C on all 3 seeds while staying within the time gate.
Applied current derates monotonically with ambient stress (156A → 108A
→ 57A), confirming the thermal behavior is retained, not merely a
faster-but-unsafe configuration.

**Track A gate: 7/7 conditions PASS** on this short-gate diagnostic
(does not by itself establish the real long-training pipeline behaves
the same way — see the warning at the top of this document and
`audit/charging_stage4_collapse_investigation.md`). See
`audit/charging_final_candidate_evaluation.md` §4 for the full
per-condition table.

---

## Section 3 — Track B: Final Candidate (B3) Results

**Full detail**: `audit/driving_power_deficit_and_full_gate.md` (authoritative).

Historical Stage-Q (pre-candidate diagnostic) showed PPO 1.9–6.3%
worse than Rule-Based EMS (129.16 Wh/km) with 0–82% regen recovery
across seeds, well short of the 85% gate. Candidates B1/B2 (reward
reweighting) and the B2-at-100k budget extension closed most of the
gap but still missed on 2/3 seeds. Candidate B3
(`w_regen_recovery=1.4` — the only change from B2, isolated per
Absolute Rule 8 — at 100,000 steps/seed) is the first configuration to
clear the gate:

| Seed | Cross-cycle Wh/km | vs 129.16 Gate | Regen Recovery | Cycles Beating Rule-Based | Max Power Deficit |
|---|---|---|---|---|---|
| 7 | 128.53 | **-0.63 (PASS)** | 99.96% | 4/4 | 1.31% of discharge energy |
| 21 | 129.34 | +0.18 (FAIL) | 96.67% | 0/4 | 0.00% |
| 42 | 128.92 | **-0.24 (PASS)** | 98.87% | 4/4 | 0.40% |

**Track B gate: 8/8 conditions PASS** at the required ≥2/3-seed
threshold (seeds 7 and 42 pass every condition; seed 21 passes 6/8,
missing only the two primary Wh/km conditions by a small margin, with
zero safety violations). See
`audit/driving_power_deficit_and_full_gate.md` for the full
per-condition table and the power-deficit tolerance definition/check.

**Disclosed limitation**: seed 21 does not reach Wh/km parity with
Rule-Based EMS. This is reported plainly, not smoothed into the
aggregate — one of three seeds trains stably and safely but does not
match Rule-Based efficiency.

---

## Section 4 — Reward Analysis

Unchanged from the original report — the reward-gradient and
empirical-distribution analysis that motivated the A1/B1/B2/B3
candidate searches. See
`audit/final_project_validation_HISTORICAL_pre_A1_B3.md` §5 for the
full charging reward-gradient table, driving reward empirical
distribution (`audit/driving_reward_distribution_final.csv`), regen
ordering test (100% pass rate, R(+1.0)>R(0.0) at all braking steps),
and action-authority verification. These are environment/reward-
structure facts, not candidate-specific results, and remain valid for
A1/B3.

---

## Section 5 — Training Stability

### 5.1 Track A (Candidate A1)
No NaN/Inf across all 3 seeds; standard-grid and stress-grid runs
both completed without incident (`audit/charging_final_candidate_evaluation.md`).

### 5.2 Track B (Candidate B3)
No NaN/Inf/crashes across 6 total 100k-step runs (Candidates B2@100k
and B3@100k, 3 seeds each). Per-chunk approx_kl/explained_variance
curves were not captured for these two runs (only end-of-training
evaluation) — disclosed as a weaker form of stability evidence than
Track A's explicit curve logging, but no anomalies were observed in
any resulting SOC/temperature/energy value.

---

## Section 6 — Limitations

Unchanged from the original report (n=3 seeds, standardized cycles
only, open-loop driving architecture, SoH reward off) — see
`audit/final_project_validation_HISTORICAL_pre_A1_B3.md` §7. One
addition:

7. **Track B seed variance**: Candidate B3 passes the ≥2/3-seed gate,
   but seed 21 does not reach Wh/km parity with Rule-Based EMS. Long
   training's per-seed checkpoint evaluation should watch whether this
   variance narrows or persists over a full run.

---

## Section 7 — Track A Final Status

**`NOT_READY_FOR_LONG_TRAINING`** (revised — originally read
READY_FOR_LONG_TRAINING at 7/7 gate conditions for Candidate A1, per
`audit/charging_final_candidate_evaluation.md`). Subsequent
investigation found that short-gate result does not transfer to the
actual long-training entry point: see
`audit/charging_stage4_collapse_investigation.md` for the confirmed
learning-rate-schedule mismatch and independently observed drift
precursors.

## Section 8 — Track B Final Status

**`READY_FOR_LONG_TRAINING`** — 8/8 gate conditions pass at the
required ≥2/3-seed threshold for Candidate B3. Full evidence:
`audit/driving_power_deficit_and_full_gate.md`. Unaffected by the
Track A investigation.

## Section 9 — Overall Project Status

**`NOT_READY_FOR_LONG_TRAINING`**

| Track | Gate | Status |
|---|---|---|
| Track A — Charging BMS | Charging time within ±5% of baseline, 7/7 conditions **of the short-gate script only — not validated against the real long-training pipeline** | **NOT READY** |
| Track B — Driving EMS | Cross-cycle Wh/km, regen, safety, deficit — 8/8 conditions | **READY** |

Track A does not currently pass its gate — the short-gate procedure
diverges from the real long-training entry point (§ warning at top of
document; full evidence in `audit/charging_stage4_collapse_investigation.md`).
Track B independently passes. Frozen
configurations: `configs/final_charging/`, `configs/final_driving/`
(also mirrored at `configs/reward_final.yaml` /
`configs/ppo_final.yaml` / `configs/energy_management_final.yaml` /
`configs/ppo_drive_ems_final.yaml`). Reproducibility freeze:
`audit/long_training_freeze_v2.md`.
