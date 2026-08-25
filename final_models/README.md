# Final Validated Models

These are the **only models that have passed full validation** and are suitable for
demonstration, evaluation, or citation. They were selected based on objective
performance criteria — not recency of training.

> [!IMPORTANT]
> The 1M-step charging model (`runs/charging_final_1m_v2/`) is **not included here**
> because it fails the primary target-reach gate. The 1M-step driving runs are not
> included because they were not formally gated against validation criteria.
> See `docs/research_findings.md` for the full analysis.

---

## Validated Model Inventory

### Track A — BMS Charging Controller (PPO)

| Directory | Seed | Training Steps | Gate Metric | Result |
| :--- | :---: | :---: | :--- | :---: |
| `charging_A1_50k_seed7/` | 7 | ~50 k | Charging time ≤ 2199.3 s | ✅ **2114.5 s** |
| `charging_A1_50k_seed21/` | 21 | ~50 k | Charging time ≤ 2199.3 s | ✅ **2095.0 s** |
| `charging_A1_50k_seed42/` | 42 | ~50 k | Charging time ≤ 2199.3 s | ✅ **2095.1 s** |

All three seeds achieved **15/15 standard scenarios reached** (target SoC hit in every scenario).

**Source run:** `runs/charging_A1_regate/seed_{7,21,42}/trained_model.zip`  
**Audit data:** `audit/charging_A1_regate_standard_combined.csv`

---

### Track B — Driving Energy Management System (PPO EMS)

Cross-cycle mean Wh/km (distance-weighted average across UDDS + HWFET + US06 + WLTP):

| Directory | Seed | Steps | Cross-cycle mean Wh/km | WLTP Wh/km | Regen Recovery | Safety Interventions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `driving_B3_100k_seed7/` | 7 | 100 k | **128.53** | 117.75 | 99.88% | 0 |
| `driving_B3_100k_seed21/` | 21 | 100 k | **129.34** | 119.53 | 97.36% | 0 |
| `driving_B3_100k_seed42/` | 42 | 100 k | **128.92** | 119.01 | 97.70% | 0 |

> [!NOTE]
> **Cross-cycle mean** = arithmetic mean of per-cycle Wh/km across the four cycles.
> **WLTP Wh/km** is the single-cycle value for WLTP Class 3b only — the lower number
> because WLTP has a longer distance denominator. Use cross-cycle mean when comparing
> across seeds; use per-cycle values when comparing specific cycles.

Validated across all four standard drive cycles: **EPA UDDS, EPA HWFET, EPA US06, WLTP Class 3b**.
Zero safety interventions on every cycle for every seed.

**Source run:** `runs/driving_final_1m_v2_seed{7,21,42}/seed_{7,21,42}/checkpoints/ppo_driving_100000_steps.zip`  
**Audit data:** `audit/driving_Cand_B3_100k_benchmark.csv`

---

## How to Load These Models

### Charging model

```python
from stable_baselines3 import PPO

model = PPO.load("final_models/charging_A1_50k_seed7/trained_model")
```

### Driving EMS model

```python
from stable_baselines3 import PPO

model = PPO.load("final_models/driving_B3_100k_seed7/ppo_driving_100000_steps")
```

---

## What Is NOT in This Directory

| Item | Location | Status |
| :--- | :--- | :--- |
| 1M-step charging model | `runs/charging_final_1m_v2/` | ❌ Fails target-reach gate |
| 1M-step driving model | `runs/driving_final_1m_v2_seed*/` | ⚠️ Not fully evaluated against gate |
| Historical candidate runs | `runs/charging_A1_regate/`, `runs/driving_final_*/` | ✅ Preserved, not promoted |

Do **not** delete the historical runs. The 1M charging training failure is a documented
research finding (see `docs/research_findings.md`).
