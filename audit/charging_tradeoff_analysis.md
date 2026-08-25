# Track A: Charging Reward Analytical Screen & Tradeoff Analysis

**Objective**: Analytically verify instantaneous reward landscape across temperatures and candidate parameter sets before short diagnostic execution.

---

## 1. Candidate Parameter Definitions

| Candidate | Thermal Ref ($T_{\text{ref}}$) | Thermal Scale ($T_{\text{scale}}$) | Thermal Weight ($w_{\text{th}}$) | Time Penalty ($w_{\text{time}}$) |
|---|---|---|---|---|
| **ExpC_Baseline** | 33.0°C | 22.0°C | 3.0 | 0.05 |
| **Candidate_A1** | 35.0°C | 20.0°C | 2.5 | 0.05 |
| **Candidate_A2** | 36.0°C | 19.0°C | 3.0 | 0.05 |
| **Candidate_A3** | 35.0°C | 20.0°C | 3.0 | 0.08 |

---

## 2. Instantaneous Reward by Temperature & Current

| Candidate | Temp (°C) | R(120A) | R(140A) | R(160A) | $\Delta R(160 - 120)$ | Optimal Current |
|---|---|---|---|---|---|---|
| ExpC_Baseline | 33°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| ExpC_Baseline | 34°C | -0.05255 | -0.05347 | -0.05453 | -0.00198 | **120A** |
| ExpC_Baseline | 35°C | -0.06020 | -0.06388 | -0.06813 | -0.00793 | **120A** |
| ExpC_Baseline | 36°C | -0.07294 | -0.08123 | -0.09079 | -0.01785 | **120A** |
| ExpC_Baseline | 38°C | -0.11373 | -0.13675 | -0.16330 | -0.04957 | **120A** |
| ExpC_Baseline | 40°C | -0.17492 | -0.22003 | -0.27208 | -0.09716 | **120A** |
| ExpC_Baseline | 42°C | -0.35650 | -0.43107 | -0.51711 | -0.16061 | **120A** |
| ExpC_Baseline | 45°C | -0.66711 | -0.79967 | -0.95263 | -0.28552 | **120A** |
| Candidate_A1 | 33°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| Candidate_A1 | 34°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| Candidate_A1 | 35°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| Candidate_A1 | 36°C | -0.05257 | -0.05350 | -0.05457 | -0.00200 | **120A** |
| Candidate_A1 | 38°C | -0.07314 | -0.08149 | -0.09113 | -0.01799 | **120A** |
| Candidate_A1 | 40°C | -0.11427 | -0.13747 | -0.16425 | -0.04998 | **120A** |
| Candidate_A1 | 42°C | -0.27596 | -0.32144 | -0.37393 | -0.09797 | **120A** |
| Candidate_A1 | 45°C | -0.55706 | -0.64989 | -0.75699 | -0.19993 | **120A** |
| Candidate_A2 | 33°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| Candidate_A2 | 34°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| Candidate_A2 | 35°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| Candidate_A2 | 36°C | -0.05000 | -0.05000 | -0.05000 | +0.00000 | **160A** |
| Candidate_A2 | 38°C | -0.06367 | -0.06861 | -0.07431 | -0.01063 | **120A** |
| Candidate_A2 | 40°C | -0.10469 | -0.12444 | -0.14722 | -0.04253 | **120A** |
| Candidate_A2 | 42°C | -0.27305 | -0.31748 | -0.36875 | -0.09570 | **120A** |
| Candidate_A2 | 45°C | -0.57686 | -0.67683 | -0.79219 | -0.21533 | **120A** |
| Candidate_A3 | 33°C | -0.08000 | -0.08000 | -0.08000 | +0.00000 | **160A** |
| Candidate_A3 | 34°C | -0.08000 | -0.08000 | -0.08000 | +0.00000 | **160A** |
| Candidate_A3 | 35°C | -0.08000 | -0.08000 | -0.08000 | +0.00000 | **160A** |
| Candidate_A3 | 36°C | -0.08308 | -0.08420 | -0.08548 | -0.00240 | **120A** |
| Candidate_A3 | 38°C | -0.10776 | -0.11779 | -0.12936 | -0.02159 | **120A** |
| Candidate_A3 | 40°C | -0.15712 | -0.18497 | -0.21710 | -0.05998 | **120A** |
| Candidate_A3 | 42°C | -0.33115 | -0.38573 | -0.44871 | -0.11756 | **120A** |
| Candidate_A3 | 45°C | -0.63847 | -0.74986 | -0.87839 | -0.23992 | **120A** |

---

## 3. Analytical Screening Evaluation

### ExpC_Baseline
- **Normal ($T \le 35^\circ\text{C}$)**: Optimal at 33°C = 160A, at 35°C = 120A $\rightarrow$ FAIL
- **Moderate ($T = 38\text{--}40^\circ\text{C}$)**: Optimal at 38°C = 120A, at 40°C = 120A $\rightarrow$ PASS (Derating Begins)
- **High Stress ($T \ge 42^\circ\text{C}$)**: Optimal at 42°C = 120A, at 45°C = 120A $\rightarrow$ PASS (Strong Derating)
- **Analytical Screen Status**: **REJECTED**

### Candidate_A1
- **Normal ($T \le 35^\circ\text{C}$)**: Optimal at 33°C = 160A, at 35°C = 160A $\rightarrow$ PASS (160A)
- **Moderate ($T = 38\text{--}40^\circ\text{C}$)**: Optimal at 38°C = 120A, at 40°C = 120A $\rightarrow$ PASS (Derating Begins)
- **High Stress ($T \ge 42^\circ\text{C}$)**: Optimal at 42°C = 120A, at 45°C = 120A $\rightarrow$ PASS (Strong Derating)
- **Analytical Screen Status**: **PASSED FOR DIAGNOSTIC TESTING**

### Candidate_A2
- **Normal ($T \le 35^\circ\text{C}$)**: Optimal at 33°C = 160A, at 35°C = 160A $\rightarrow$ PASS (160A)
- **Moderate ($T = 38\text{--}40^\circ\text{C}$)**: Optimal at 38°C = 120A, at 40°C = 120A $\rightarrow$ PASS (Derating Begins)
- **High Stress ($T \ge 42^\circ\text{C}$)**: Optimal at 42°C = 120A, at 45°C = 120A $\rightarrow$ PASS (Strong Derating)
- **Analytical Screen Status**: **PASSED FOR DIAGNOSTIC TESTING**

### Candidate_A3
- **Normal ($T \le 35^\circ\text{C}$)**: Optimal at 33°C = 160A, at 35°C = 160A $\rightarrow$ PASS (160A)
- **Moderate ($T = 38\text{--}40^\circ\text{C}$)**: Optimal at 38°C = 120A, at 40°C = 120A $\rightarrow$ PASS (Derating Begins)
- **High Stress ($T \ge 42^\circ\text{C}$)**: Optimal at 42°C = 120A, at 45°C = 120A $\rightarrow$ PASS (Strong Derating)
- **Analytical Screen Status**: **PASSED FOR DIAGNOSTIC TESTING**

