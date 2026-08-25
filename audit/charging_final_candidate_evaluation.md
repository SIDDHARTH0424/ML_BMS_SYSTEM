# Track A — Final Candidate Evaluation (Candidate A1)

> [!WARNING]
> **Superseded — do not read the 7/7-PASS conclusion below as current
> status without the update in this notice.** This document reports
> the short-gate diagnostic (`experiments/run_readiness_diagnostics.py`)
> results only. A later investigation
> (`audit/charging_stage4_collapse_investigation.md`) found that
> diagnostic uses a **constant learning rate**, while the actual
> long-training entry point (`agents/train_ppo.py::build_agent()`)
> used a **decaying** LR schedule the diagnostic never tested —
> confirmed and reproduced exactly (50k-step, seed-7 replay through
> the real pipeline gave 2577.60s at the time, matching the
> externally-reported Stage 3 result to the decimal, vs 2114.53s
> here), with a full 3-seed re-gate confirming FAIL
> (`audit/charging_A1_regate_via_build_agent_report.md`). **That
> mismatch has since been fixed** (`build_agent()` now uses a constant
> LR) **and the 3-seed gate re-run against the fix now PASSES again**
> (2115.53s / 2095.00s / 2095.27s — within ~1s of this document's
> original numbers per seed; `audit/charging_A1_regate_flat_lr_fixed_report.md`).
> **Track A remains `NOT_READY`**, not because of this short-gate
> result (which now holds again) but because the full 1,000,000-step
> long run has not yet been executed and evaluated against the fixed
> pipeline — separately, a properly-controlled run under the *old*
> decaying-LR pipeline found genuine drift precursors through 450k of
> 1,000,000 steps that have not been re-checked under the fix.
> Everything below this notice is retained unmodified
> as the historical basis for the original (now superseded) READY
> verdict.

**This is the authoritative Track A evidence document.** It reports
only the actual Candidate A1 3-seed results — not historical Exp C or
earlier candidates (A2/A3), which failed and are documented separately
in `audit/charging_candidates_metrics.csv` / `audit/charging_candidates_stress_eval.csv`
for traceability only.

**Configuration**: `thermal_reference_temp_c=35.0, thermal_weight=2.5,
thermal_scale_c=20.0`, PPO otherwise unchanged from dev
(`ent_coef=0.01, target_kl=0.01, use_sde=false, squash_output=false`).
Frozen at `configs/reward_final.yaml` / `configs/final_charging/`.

**Baseline**: Max Current controller, standard charging time = 2094.6s.
**Gate threshold**: 2094.6 × 1.05 = **2199.3s**.

## 1. Standard 15-Scenario Grid — Charging Time

| Seed | Charging Time (s) | Δ vs Baseline (s) | Δ vs Baseline (%) | vs 2199.3s Gate | Reached Target (15/15) |
|---|---|---|---|---|---|
| 7 | 2114.53 | +19.93 | **+0.95%** | PASS (84.8s margin) | Yes |
| 21 | 2095.00 | +0.40 | **+0.02%** | PASS (104.3s margin) | Yes |
| 42 | 2095.13 | +0.53 | **+0.03%** | PASS (104.2s margin) | Yes |
| **Mean ± SD** | **2101.55 ± 11.24** | **+6.95 ± 11.24** | **+0.33 ± 0.53%** | **PASS, all 3 seeds** | **45/45 (100%)** |

No delta is rounded or hidden — all three are comfortably inside the
gate, with the largest (seed 7) using only 43% of the allowed 5%
margin.

## 2. Thermal Behavior — Standard Grid

| Seed | Peak Temp (°C) | Mean Requested Current (A) | Mean Applied Current (A) | Cumulative Q_gen (J) |
|---|---|---|---|---|
| 7 | 42.76 | 155.3 | 154.5 | 2.362e6 |
| 21 | 43.07 | 157.6 | 155.9 | 2.382e6 |
| 42 | 43.06 | 158.2 | 155.9 | 2.382e6 |

At normal ambient (15–35°C), A1 requests current close to the 160A
ceiling (mean 155–158A) — confirms the candidate does **not**
unnecessarily derate under cool conditions (gate condition A5).

## 3. Thermal Behavior — Stress Grid (45°C / 50°C ambient)

| Seed | Ambient | Peak Temp (°C) | Mean Applied Current (A) | Cumulative Q_gen (J) | Reached Target |
|---|---|---|---|---|---|
| 7 | 45°C | 49.29 | 103.8 | 1.653e6 | 3/3 SoC scenarios |
| 7 | 50°C | 51.76 | 55.2 | 0.868e6 | 3/3 |
| 21 | 45°C | 49.60 | 109.9 | 1.714e6 | 3/3 |
| 21 | 50°C | 51.91 | 58.3 | 0.899e6 | 3/3 |
| 42 | 45°C | 49.60 | 109.9 | 1.714e6 | 3/3 |
| 42 | 50°C | 51.91 | 58.3 | 0.899e6 | 3/3 |

**Max Current baseline** at 45°C stress: peak temperature **50.75°C**
(from `audit/long_training_freeze_HISTORICAL_B2.md` §1, historical comparison run).

A1 vs Max Current at matched 45°C stress: **peak temp 49.29–49.60°C vs
50.75°C — 1.15–1.46°C lower for all 3 seeds (3/3)**, exceeding gate
condition A4's "≥2/3 seeds" requirement.

Applied current drops monotonically from ~156A (normal) → ~108A (45°C)
→ ~57A (50°C) — thermal derating is present and scales with ambient
stress, not merely present-or-absent (gate condition A3).

## 4. Gate Evaluation (Track A, 7 conditions)

| # | Condition | Result |
|---|---|---|
| A1 | 3/3 seeds reach 95% SoC, standard grid | PASS (45/45 scenarios) |
| A2 | Every seed ≤ 2199.3s | PASS (max 2114.5s, seed 7) |
| A3 | Measurable stress-dependent derating | PASS (156A→108A→57A) |
| A4 | ≥2/3 seeds lower peak temp than Max Current at stress | PASS (3/3) |
| A5 | Normal-temp behavior stays aggressive | PASS (mean 155–158A vs 160A ceiling) |
| A6 | No catastrophic standard-grid failure | PASS |
| A7 | Training numerically stable (no NaN/Inf) | PASS (no anomalies across all 3 seeds' logged runs) |

**7/7 PASS on the short-gate diagnostic. Track A = `READY_FOR_LONG_TRAINING` — SUPERSEDED, see warning at top of document. Current status: `NOT_READY` per `audit/charging_stage4_collapse_investigation.md`.**

This conclusion rests on both the time gate *and* the retained thermal
behavior — A1 is not simply "fast," it demonstrably still derates
under stress while staying within the time budget (rules out passing
by ignoring thermal safety, and rules out passing only because it
charges too slowly, per Absolute Rules 11). **This remains true of the
short-gate diagnostic's own behavior** — what it does not establish is
that the real long-training entry point (different LR schedule)
behaves the same way at scale; see the investigation doc.
