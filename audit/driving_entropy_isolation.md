# Track B: Entropy Isolation & Action Distribution Analysis

**Objective**: Isolate whether non-zero policy entropy (`ent_coef = 0.005`) alone resolves the regen discovery problem without modifying reward weights.

---

## 1. Action Distribution at Braking Opportunity States

| Experiment | Seed | ent_coef | Policy Std | $P(a > 0 \mid \text{braking})$ | $P(a \ge 0.5 \mid \text{braking})$ | Mean Action (Braking) | Mean Action (Propulsion) |
|---|---|---|---|---|---|---|---|
| StageQ_Baseline_ent0 | 7 | 0.0 | 0.7960 | **14.1%** | 6.2% | -0.864 | -1.244 |
| StageQ_Baseline_ent0 | 21 | 0.0 | 0.8116 | **24.4%** | 10.5% | -0.631 | -1.280 |
| StageQ_Baseline_ent0 | 42 | 0.0 | 0.8621 | **48.3%** | 31.3% | -0.023 | -1.211 |
| Entropy_Isolated_ent0005 | 7 | 0.005 | 0.8237 | **27.3%** | 10.3% | -0.609 | -1.082 |
| Entropy_Isolated_ent0005 | 21 | 0.005 | 0.8576 | **23.7%** | 8.6% | -0.601 | -1.349 |
| Entropy_Isolated_ent0005 | 42 | 0.005 | 0.8423 | **48.1%** | 30.1% | -0.029 | -1.385 |

---

## 2. Multi-Cycle Benchmark with Entropy Isolation (`ent_coef = 0.005`)

| Seed | UDDS (Wh/km) | HWFET (Wh/km) | US06 (Wh/km) | WLTP 3b (Wh/km) | Mean Wh/km | Regen Recovery (%) |
|---|---|---|---|---|---|---|
| Seed 7 | 102.54 | 132.36 | 186.60 | 125.58 | **136.77** | **10.2%** |
| Seed 21 | 102.20 | 132.43 | 185.17 | 125.63 | **136.36** | **13.8%** |
| Seed 42 | 90.91 | 131.09 | 180.31 | 121.86 | **131.04** | **77.3%** |

**Rule-Based EMS Cross-Cycle Baseline**: **129.16 Wh/km** (100% regen)
