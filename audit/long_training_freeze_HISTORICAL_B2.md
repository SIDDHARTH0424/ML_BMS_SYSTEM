# Master Long-Training Readiness & Configuration Freeze Report

> [!WARNING]
> **SUPERSEDED.** This is the v1 freeze report, written when only
> Candidate B2 existed for Track B (which did NOT pass the gate).
> Track B was later re-run as Candidate B3, which DOES pass.
> **Current authoritative status: `audit/long_training_freeze_v2.md`.**
> This document is retained as historical evidence of the B2-only
> state, not as current project status.


**Timestamp**: 2026-08-16T13:55:00Z  
**OS**: Windows 11  
**Python**: 3.12.10 (venv)  
**Torch**: 2.13.0+cpu  
**Stable-Baselines3**: 2.9.0  
**Test Suite**: 213 passed / 0 failed  

---

## 1. Executive Summary of Diagnostic Readiness

### Track A (Charging BMS) — `READY_FOR_LONG_TRAINING`
- **Candidate A1** ($T_{\text{ref}} = 35.0^\circ\text{C}, w_{\text{th}} = 2.5, T_{\text{scale}} = 20.0^\circ\text{C}, w_{\text{time}} = 0.05$) successfully resolved the charging-time tradeoff:
  - Standard Grid Time Gate: Baseline = 2094.6s, Maximum allowed = 2199.3s (+5.0%).
  - Seed 7: **2114.5s (+0.95%)** — **PASS**
  - Seed 21: **2095.0s (+0.02%)** — **PASS**
  - Seed 42: **2095.1s (+0.03%)** — **PASS**
  - Standard Target Completion: **100% (15/15 scenarios across all 3 seeds)**.
  - Stress Derating: Monotonically decreases current with temperature ($155\text{--}158\text{A}$ normal $\rightarrow 102\text{--}135\text{A}$ at 45°C $\rightarrow 57\text{--}104\text{A}$ at 50°C).
  - Stress Temperature Control: Peak temperature at 45°C stress is $49.0\text{--}49.8^\circ\text{C}$ vs $50.75^\circ\text{C}$ Max Current baseline.
  - Target Completion on Stress Grid: **100% (6/6 scenarios across all 3 seeds)**.
  - **All 7/7 Track A Readiness Gate Criteria PASSED.**

### Track B (Driving EMS) — `NEEDS_DRIVING_REWARD_REVISION`
- **Candidate B2** ($w_{\text{regen}} = 1.2, w_{\text{energy}} = 0.2, w_{\text{track}} = 1.5, \text{ent\_coef} = 0.010$) dramatically improved regen discovery across all 3 seeds:
  - Seed 42: **129.31 Wh/km** (vs Rule-Based 129.16 Wh/km: **+0.15 Wh/km (+0.11%)**), **96.0% regen recovery**, beats Rule-Based on EPA US06 ($179.55\text{ vs } 179.65\text{ Wh/km}$).
  - Seed 21: **129.72 Wh/km** (+0.56 Wh/km), **90.5% regen recovery**.
  - Seed 7: **130.31 Wh/km** (+1.15 Wh/km), **83.9% regen recovery**.
- **Gate Evaluation**: In short 50k diagnostic training, Candidate B2 closes 92% of the gap, but the cross-cycle mean across seeds ($129.78\text{ Wh/km}$) is slightly above the strict threshold ($129.16\text{ Wh/km}$). Therefore, per strict non-loosening rules, Track B remains `NEEDS_DRIVING_REWARD_REVISION` / `READY_FOR_LONG_TRAINING_CANDIDATE`.

---

## 2. Frozen Configuration Hashes

| File | MD5 Fingerprint |
|---|---|
| `configs/reward_final.yaml` | `37812984cfb7d03a116a4be5a21074a3` |
| `configs/simulation_final.yaml` | `f5999ae24268e7ecda03b41c09b0dd47` |
| `configs/ppo_final.yaml` | `5c73c2423bb0d927c3d2dc590c685bf5` |
| `configs/energy_management_final.yaml` | `2657e504c3114fa9960ffbb56eef4b89` |
| `configs/ppo_drive_ems_final.yaml` | `ec8ee6a51d4512b9d5c80e1b6f5cf212` |

---

## 3. Master Multi-Seed Consistency Summary

### Track A (Charging BMS — Candidate A1 vs Baselines)

| Controller / Seed | Target Reached | Standard Time (s) | $\Delta$ Time vs Baseline (%) | Peak Temp (°C) | Stress 45°C Req (A) | Stress 50°C Req (A) |
|---|---|---|---|---|---|---|
| Max Current Baseline | 100% | 2094.6 | 0.0% | 43.07 | 160.0 | 160.0 |
| run_001 (1M PPO) | 100% | 2094.6 | 0.0% | 43.07 | 160.0 | 160.0 |
| Candidate A1 (Seed 7) | **100%** | **2114.5** | **+0.95%** | 42.76 | **104.2** | **56.9** |
| Candidate A1 (Seed 21) | **100%** | **2095.0** | **+0.02%** | 43.07 | **130.3** | **86.5** |
| Candidate A1 (Seed 42) | **100%** | **2095.1** | **+0.03%** | 43.06 | **134.6** | **104.0** |
| **Candidate A1 Mean ± SD** | **100%** | **2101.5 ± 11.2** | **+0.33 ± 0.53%** | **42.96 ± 0.18** | **123.0 ± 16.4** | **82.5 ± 23.8** |

### Track B (Driving EMS — Candidate B2 vs Rule-Based)

| Controller / Seed | UDDS (Wh/km) | HWFET (Wh/km) | US06 (Wh/km) | WLTP 3b (Wh/km) | Mean Wh/km | Regen Recovery (%) | Power Deficit (Wh) |
|---|---|---|---|---|---|---|---|
| Rule-Based EMS Baseline | **86.74** | **130.87** | **179.65** | **119.37** | **129.16** | **100.0%** | **0.00** |
| Candidate B2 (Seed 42) | 87.05 | 130.96 | **179.55** | 119.68 | **129.31** | **96.0%** | 1.42 ± 1.41 |
| Candidate B2 (Seed 21) | 87.80 | 131.22 | 179.85 | 120.03 | **129.72** | **90.5%** | 0.00 ± 0.00 |
| Candidate B2 (Seed 7) | 88.67 | 131.15 | 179.86 | 121.54 | **130.31** | **83.9%** | 0.00 ± 0.00 |
| **Candidate B2 Mean ± SD** | **87.84 ± 0.81** | **131.11 ± 0.14** | **179.75 ± 0.18** | **120.42 ± 0.99** | **129.78 ± 0.50** | **90.1 ± 6.1%** | **0.47 ± 0.72** |
