# Pre-Redesign Checkpoint (Frozen Baseline)

**Timestamp**: 2026-08-16T08:20:00Z  
**Purpose**: Immutable snapshot of the validated baseline before any objective/reward redesign changes are introduced.  
**Overall Classification at Freeze**: `NOT_READY`  

---

## 1. Test Suite State

| Suite | Result |
|-------|--------|
| `python -m compileall .` | ✅ 0 syntax errors |
| `python -m pytest tests/ -v` | ✅ **197 passed / 0 failed / 0 skipped** |

---

## 2. Track Classifications at Freeze

| Track | Classification |
|-------|---------------|
| Track A — Charging BMS | `NEEDS_CHARGING_OBJECTIVE_REVISION` |
| Track B — Driving EMS | `NEEDS_DRIVING_REWARD_REVISION` |
| Overall | `NOT_READY` |

---

## 3. Package Sizes at Freeze

| Archive | Size |
|---------|------|
| `rl-bms-Driving-source-clean.zip` | 0.26 MB |
| `rl-bms-Driving-results.zip` | 32.65 MB |

Source ZIP contains 139 files with 0 dirty/cache files verified programmatically.

---

## 4. Config Fingerprints (MD5)

| Config File | MD5 |
|-------------|-----|
| `configs/reward.yaml` | `b9622b215174b9e0bac35a879988cee0` |
| `configs/energy_management.yaml` | `8a3d9cc29e629edf64ca6d008581e1ec` |
| `configs/ppo.yaml` | `a7f6097f8ed8cdd8dd3e6b3359b94f66` |
| `configs/ppo_drive_ems.yaml` | `34e59febc5cd6f9c382199b61312cc95` |
| `configs/simulation.yaml` | `5d11084f03ac0a5ea3ab720025e7df4b` |

---

## 5. Key Benchmark Values at Freeze

### Track A — Charging BMS

| Controller | Reached (15/15) | Mean Time (s) | Peak Temp (°C) | Mean Req I (A) |
|------------|----------------|--------------|----------------|----------------|
| Max Current Baseline | 100% | 2094.6 | 43.07 | 160.00 |
| run_001 (1M PPO) | 100% | 2094.6 | 43.07 | 160.00 |
| PPO Exp A (50k, 3 seeds) | 100% | 2094.9 ± 0.1 | 43.07 | 159.43 ± 0.60 |
| PPO Exp B (Stage F, 50k) | 100% | 2094.9 ± 0.1 | 43.07 | 159.39 ± 0.39 |

**Root cause**: `thermal_reward` and `temp_penalty` identically zero across all 24,576 logged steps in run_001. Training ambient [15–35°C] never drives T above 40°C during bulk charging.

### Track B — Driving EMS

| Controller | Cross-Cycle Wh/km | Regen Recovery | Power Deficit (Wh) |
|-----------|------------------|----------------|-------------------|
| Rule-Based EMS | 129.16 ± 38.5 | 100.0% | 0.00 |
| PPO (seed 42, 50k steps) | 131.04 ± 37.4 | 75.2% | 0.23 ± 0.27 |

**Root cause**: `energy_cost` (82.5%) + `regen_recovery` (17.5%) dominate. Thermal/safety/tracking terms stayed zero under nominal cycles. PPO 1.46% worse than Rule-Based due to suboptimal regen capture.

---

## 6. Historical Baseline Preservation

- `runs/run_001/` — **CHARGING_PPO_BASELINE_1M** — intact and preserved, never to be modified.
- `runs/driving_ppo_baseline/` — Previous driving diagnostic checkpoints (seeds 7/21/42, pre-stageQ).
- `audit/driving_ppo_training_curves.csv` — Previous driving training curves (immutable).
- `audit/driving_multicycle_benchmark.csv` — Previous driving benchmark (immutable).

---

## 7. Workspace State

- Git: Unversioned local workspace (no `.git` repository).
- Python: venv at `venv/` (not included in source ZIP).
- Platform: Windows, PowerShell.
