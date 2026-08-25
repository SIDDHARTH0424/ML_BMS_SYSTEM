# Master Long-Training Readiness & Configuration Freeze Report (v2)

**Timestamp**: 2026-08-16T14:30:00Z
**Python**: 3.12 (sandbox re-verification)
**Torch**: 2.13.0
**Stable-Baselines3**: 2.9.0
**Test Suite**: 213 passed / 0 failed / 0 errors (`audit/long_training_freeze_tests.txt`)
**Compileall**: 0 errors

> [!WARNING]
> **Track A status revised below — no longer unconditionally READY.**
> A later investigation (`audit/charging_stage4_collapse_investigation.md`)
> found that the actual long-training entry point
> (`agents/train_ppo.py::build_agent()`) used a decaying learning-rate
> schedule that the short-gate diagnostic below never tested (it used
> a constant LR) — confirmed, reproduced, and corroborated by drift
> precursors observed through 450k of 1,000,000 steps. **That specific
> mismatch has since been fixed** (`build_agent()` now uses a constant
> LR) **and re-verified — the 3-seed short gate now passes again
> through the real production pipeline** (`audit/charging_A1_regate_flat_lr_fixed_report.md`).
> **Status is still `NOT_READY`**: the fix closes the short-budget
> mismatch, but no full 1,000,000-step run has been executed against
> the fixed pipeline with checkpoint evaluation. See
> `audit/charging_stage4_collapse_investigation.md` for the full
> history and current authoritative status — this section is retained
> unedited below as the historical basis for the original READY
> verdict.

Supersedes the Track B verdict in the original freeze report — now
renamed `audit/long_training_freeze_HISTORICAL_B2.md` to make clear it
no longer reflects current status (Track A verdict there is unchanged
and reconfirmed here). **This document (v2) is the sole authoritative
freeze document for the project** for the claims made as of its
original timestamp; Track A's status has since been revised — see the
warning above.

---

## 1. Executive Summary

### Track A (Charging BMS) — `READY_FOR_LONG_TRAINING` (as of this document's original writing — SUPERSEDED, see warning above)
Candidate A1 (`T_ref=35.0°C, w_th=2.5, T_scale=20.0°C, w_time=0.05`)
passed all 7/7 gate conditions **of the short-gate diagnostic script**
— which, it was later discovered, does not use the same PPO
construction as the actual long-training entry point (see
`audit/charging_stage4_collapse_investigation.md`). See
`audit/long_training_freeze_HISTORICAL_B2.md` §1 for the original
short-gate detail; nothing about Track A's short-gate evidence itself
was rerun or altered in this pass.

### Track B (Driving EMS) — `READY_FOR_LONG_TRAINING` (updated)

Two controlled, single-variable follow-on experiments were run after the
original 50k-step Candidate B2 result (129.78 Wh/km mean, 1/7 conditions
failing):

1. **Candidate B2 @ 100k steps** (budget-only change, reward weights
   unchanged): 129.24 Wh/km mean, 97.2% regen. Closed ~87% of the gap
   but still only 1/3 seeds cleared the binding Wh/km conditions.
   (`audit/driving_Cand_B2_100k_report.md`)

2. **Candidate B3 @ 100k steps** (single further change: `w_regen_recovery`
   1.2 → 1.4, everything else identical to Candidate B2 @ 100k):
   **128.93 Wh/km mean, 98.5% regen. Seeds 7 and 42 (2/3) pass every
   gate condition.** Seed 21 passes 6/8 conditions (misses only the
   Wh/km and per-cycle conditions, by +0.18 Wh/km / 0 cycles beaten —
   still zero safety violations, negligible power deficit, no cycle
   >1.5% off). (`audit/driving_power_deficit_and_full_gate.md`,
   `audit/driving_B3_checkpoint_quality_diagnostic_report.md`)

| Metric | Seed 7 | Seed 21 | Seed 42 |
|---|---|---|---|
| Cross-cycle Wh/km | **128.53** | 129.34 | **128.92** |
| vs 129.16 gate | **-0.63** | +0.18 | **-0.24** |
| Regen recovery | **99.96%** | 96.67% | **98.87%** |
| Cycles beating Rule-Based | **4/4** | 0/4 | **4/4** |
| Safety interventions | 0 | 0 | 0 |
| Max power deficit (any cycle) | 38.15 Wh (WLTP, ~1.3% of trip energy) | 0.00 Wh | 11.63 Wh (WLTP, ~0.4%) |

**Gate Evaluation (Track B Step 8, need ≥2/3 seeds per condition)**

| Condition | Seeds passing | Verdict |
|---|---|---|
| 1. Cross-cycle Wh/km ≤ 129.16 | 2/3 (7, 42) | **PASS** |
| 2. PPO ≤ Rule-Based on ≥3/4 cycles | 2/3 (7, 42) | **PASS** |
| 3. Regen recovery > 85% | 3/3 | PASS |
| 4. No safety violations | 3/3 | PASS |
| 5. Power deficit not materially worse | 3/3 (all <1.5% of trip energy) | PASS |
| 6. No cycle >10% worse | 3/3 (worst case -1.4%) | PASS |
| 7. Final SOC / max temp valid | 3/3 | PASS |
| 8. Training stable (no NaN/Inf, sane outputs) | 3/3 (full per-chunk approx_kl/explained_variance/value_loss/entropy_loss curves captured for all 3 seeds — `audit/driving_B3_checkpoint_quality_diagnostic_report.md`; zero NaN/Inf, one isolated approx_kl exceedance of 0.0219 disclosed, explained_variance 0.86-0.90 by final chunk) | **PASS** |

**8/8 conditions clear the ≥2/3-seed bar. Track B passes the strict
short-training gate.**

**Note on condition 8 (resolved)**: the original candidate-search
scripts did not log per-chunk curves — only end-of-training
evaluation. This gap is now closed: a diagnostic-only re-run of the
identical, unchanged B3 config with SB3's CSV logger attached
(`audit/driving_B3_checkpoint_quality_diagnostic_report.md`)
reproduced the exact same gated evaluation numbers for all 3 seeds
(confirming it's genuinely the same training process) and captured
full `train/approx_kl`, `train/explained_variance`, `train/value_loss`,
`train/entropy_loss`, `train/std`, `rollout/ep_rew_mean`, and
`rollout/ep_len_mean` curves. Result: zero NaN/Inf, one isolated
approx_kl exceedance (0.0219 vs ~0.02 tolerance, 1/147 chunks,
disclosed rather than smoothed over), explained_variance climbing to
0.86-0.90 by the final chunk for all seeds, and monotonic learning
progress (+48 to +59 reward points from first to last chunk).
Condition 8 is now an unqualified PASS with evidence on par with
Track A's.

---

## 2. Frozen Configuration Hashes

| File | MD5 |
|---|---|
| `configs/reward_final.yaml` | `f368cf85f71a05ce1f32fbc1caa27f3a` |
| `configs/simulation_final.yaml` | `114225f9478394f4f406de1d947811c5` |
| `configs/ppo_final.yaml` | `5d077488d852ed970a2ba7be22b16077` |
| `configs/energy_management_final.yaml` | `86c3861fc5dd0ea8fd17f3454e787682` |
| `configs/ppo_drive_ems_final.yaml` | `01f85e9f55fc236bfd7c1bdf75dc0dd4` |

`energy_management_final.yaml` differs from the dev `energy_management.yaml`
in exactly one field: `w_regen_recovery: 0.5 -> 1.4`. All other reward
weights, thermal parameters, and episode settings are unchanged.
`ppo_drive_ems_final.yaml` differs from the dev `ppo_drive_ems.yaml` in
exactly one field: `ent_coef: 0.0 -> 0.010`.

---

## 3. Overall Project Status

**`NOT_READY_FOR_LONG_TRAINING`** (revised — see warning at top of
document) — Track A's short-gate procedure does not match the actual
long-training entry point's PPO construction, and independently
observed drift precursors mean the 1,000,000-step claim is not
currently supported. Track B is unaffected and remains READY.

| Track | Gate | Status |
|---|---|---|
| Track A — Charging BMS | Charging time within ±5% of baseline, 7/7 conditions **of the short-gate script — not validated against the actual long-training pipeline** | **NOT READY** — see `audit/charging_stage4_collapse_investigation.md` |
| Track B — Driving EMS | Cross-cycle Wh/km, regen, safety — 8/8 conditions (≥2/3 seed) | **READY** |

---

## 4. Caveats Carried Forward (unchanged from original report)

- n=3 seeds: no formal statistical significance claimed for either track.
- Standardized drive cycles (UDDS/HWFET/US06/WLTP 3b) only — not real-world
  road validation.
- Open-loop driving architecture: prescribed speed trace, power deficit
  logged rather than the vehicle slowing down.
- SoH/battery-degradation reward is off; no battery-life claim is made.
- Track B seed 21, while not part of the passing 2/3, showed no failure
  mode beyond slightly slower regen-policy convergence — same reward
  ordering, zero safety issues. Long-training checkpoint evaluation
  (Part 9) should watch whether this seed-level variance persists or
  narrows over a full run.

## 5. Post-Freeze Fixes (this pass)

Verified against the "Final Improvements Before Long Training" checklist:
- `configs/final_charging/` and `configs/final_driving/` config-dir
  loading confirmed via actual runs (not just file presence) —
  `audit/config_dir_checkout_review.md`.
- Driving-side effective-config snapshot added (was missing;
  charging side already had it) — `training/train_drive_ems.py::train_long()`.
- Formal per-condition gate docs: `audit/charging_final_candidate_evaluation.md`
  (Track A, 7/7), `audit/driving_power_deficit_and_full_gate.md`
  (Track B, 8/8, including a defined power-deficit tolerance).
- `audit/final_project_validation.md` rewritten to reflect A1/B3 as
  final (was internally contradictory, citing stale Exp-C/Stage-Q
  data as primary); original preserved as
  `audit/final_project_validation_HISTORICAL_pre_A1_B3.md`.
- Discovered and fixed a real bug: `tests/test_multistage_run_dir.py`
  was silently writing real run directories into the project's
  `runs/` instead of an isolated tmp dir — `audit/test_isolation_bug_fix.md`.
- Final regression after all fixes: 213/213 passed, 0 compile errors
  (`audit/long_training_freeze_tests.txt`).

## 6. Final Package Checksums
> [!NOTE]
> Self-reference caveat: the checksums above describe the package
> build immediately prior to this note being added. Since this file
> ships inside the source archive, a checksum of "the archive
> containing this exact checksum" is not self-consistent by
> construction (adding the checksum changes the archive). Treat the
> checksums the assistant reports in the final chat message of the
> session that produced a given delivery as authoritative for that
> delivery; this file documents the freeze rationale, not a live
> self-hash.


| Archive | MD5 |
|---|---|
| `rl-bms-Driving-source-clean-final.zip` (234 files, 0 bad) | `462c05a2ad1204ca8f32ca3609608da3` |
| `rl-bms-Driving-results-final.zip` (173 files) | `425af495cb49431586464e985ba65686` |

**The configuration is frozen for long training.**

## 7. Authorization

Per Part 8: long training (1,000,000 timesteps) may proceed for BOTH
tracks using the frozen configurations above. Checkpoint evaluation at
50k/100k/200k/400k/600k/800k/1M per Part 9 remains required before
selecting a final model — passing the short gate authorizes starting
the long run, not the eventual checkpoint choice.

## 6. Final Training Entrypoint Hardening

The long-training entrypoints now accept an explicit `--config-dir` so the frozen A1/B3 configurations cannot be accidentally replaced by development YAML files.

Frozen bundles:
- `configs/final_charging/` → canonical `ppo.yaml`, `reward.yaml`, `simulation.yaml`, plus battery/safety configs.
- `configs/final_driving/` → canonical `ppo_drive_ems.yaml`, `energy_management.yaml`, plus battery/safety/vehicle/drivetrain configs.

Charging long training:
`python -m training.train --run-name charging_final_1m --config-dir configs/final_charging --stages 1 2 3 4`

Driving long training:
`python -m training.train_drive_ems --train --run-name driving_final_1m --config-dir configs/final_driving --drive-cycle data/drive_cycles/standard/wltp_class3b/cycle.csv --timesteps 1000000 --seed 7`

The training commands themselves are ready, but long training must still be launched separately and monitored at the required checkpoints.
