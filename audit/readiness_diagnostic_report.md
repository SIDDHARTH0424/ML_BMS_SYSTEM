# Final Long-Training Readiness Diagnostic Report

**Evaluation Timestamp**: 2026-08-16T19:23:56.662981  

---

## 1. Track A: Charging BMS Candidate Results

Strict Time Gate: $\le 2199.3\text{s}$ ($+5.0\%$ vs $2094.6\text{s}$ baseline)

| Candidate | Seed | Target Reached (15/15) | Mean Charging Time (s) | $\Delta$ Time vs Baseline (%) | Peak Temp (°C) | Mean Req Current (A) | Time Gate Status |
|---|---|---|---|---|---|---|---|
| Cand_A1 | 7 | True | 2114.5 | +0.95% | 42.76 | 155.3 | **PASS** |
| Cand_A1 | 21 | True | 2095.0 | +0.02% | 43.07 | 157.6 | **PASS** |
| Cand_A1 | 42 | True | 2095.1 | +0.03% | 43.06 | 158.2 | **PASS** |
| Cand_A2 | 7 | True | 2397.3 | +14.45% | 40.83 | 137.8 | **FAIL** |
| Cand_A2 | 21 | True | 2129.4 | +1.66% | 42.84 | 153.4 | **PASS** |
| Cand_A2 | 42 | True | 2198.2 | +4.95% | 42.47 | 149.4 | **PASS** |

---

## 2. Track B: Driving EMS Candidate Results

Strict Efficiency Gate: Cross-Cycle Mean $\le 129.16\text{ Wh/km}$ and Regen Recovery $> 85\%$

| Candidate | Seed | ent_coef | Mean Wh/km | $\Delta$ vs Rule-Based (129.16) | Regen Recovery (%) | Gate Status |
|---|---|---|---|---|---|---|
| Cand_B1 | 7 | 0.005 | 131.35 | +2.19 | 70.6% | **FAIL** |
| Cand_B1 | 21 | 0.005 | 129.75 | +0.59 | 90.3% | **FAIL** |
| Cand_B1 | 42 | 0.005 | 129.76 | +0.60 | 90.0% | **FAIL** |
| Cand_B2 | 7 | 0.01 | 130.31 | +1.15 | 83.9% | **FAIL** |
| Cand_B2 | 21 | 0.01 | 129.72 | +0.56 | 90.5% | **FAIL** |
| Cand_B2 | 42 | 0.01 | 129.31 | +0.15 | 96.0% | **FAIL** |
