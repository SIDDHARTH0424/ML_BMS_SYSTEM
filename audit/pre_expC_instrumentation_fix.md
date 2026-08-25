# Pre-Exp C Instrumentation Fix Checkpoint

**Timestamp**: 2026-08-16T12:34:00Z  
**Snapshot Type**: PRE-FIX SNAPSHOT  
**Purpose**: Immutable baseline record prior to fixing ambient temperature step-level instrumentation and re-running Experiment C with the instrumented mixed sampler.

---

## 1. Track Status at Snapshot

| Track | Classification |
|---|---|
| Track A (Charging BMS) | `NEEDS_CHARGING_OBJECTIVE_REVISION` / `READY_FOR_STRESS_TRAINING` |
| Track B (Driving EMS) | `NEEDS_DRIVING_REWARD_REVISION` |
| Overall Project Status | `NOT_READY` |

---

## 2. Package Names at Snapshot

- `rl-bms-Driving-source-clean.zip`
- `rl-bms-Driving-source-clean-stageQ.zip`
- `rl-bms-Driving-results.zip`

---

## 3. Preserved Historical Run Directories & Artifacts

- `runs/run_001/` (CHARGING_PPO_BASELINE_1M, preserved immutable)
- `runs/driving_ppo_baseline_refresh/` (Preserved immutable)
- `runs/driving_ppo_stageQ/` (Preserved immutable)
- `audit/charging_reward_analysis.md` & `audit/charging_reward_analysis.csv`
- `audit/diagnostic_ab_seed_metrics.csv` & `audit/diagnostic_ab_training_curves.csv`
- `audit/driving_reward_distribution.csv` & `audit/driving_action_authority.md`
- `audit/driving_multicycle_benchmark_*.csv` & `audit/driving_ppo_training_curves_*.csv`
- `audit/pre_redesign_checkpoint.md`

---

## 4. Configuration MD5 Fingerprints

| Configuration File | MD5 Hash |
|---|---|
| `configs/reward.yaml` | `b9622b215174b9e0bac35a879988cee0` |
| `configs/simulation.yaml` | `0350e2c3318a3ba232ac9b7a7789eb3a` |
| `configs/ppo.yaml` | `a7f6097f8ed8cdd8dd3e6b3359b94f66` |
| `configs/ppo_drive_ems.yaml` | `34e59febc5cd6f9c382199b61312cc95` |
| `configs/energy_management.yaml` | `a5cf6e5f3f70b458fe2495fb11a10520` |

---

## 5. Motivation for Fix

The training environment `BatteryChargingEnv` did not include `"ambient_temp_c"` in the step-level `info` dictionary returned to `Monitor`/callbacks during training. As a result, the callback accumulator could not directly extract the exact ambient temperature of every completed episode to verify the empirical sampling distribution ($p_{\text{normal}} = 0.75, p_{\text{stress}} = 0.25$).

This fix adds `info["ambient_temp_c"] = self._ambient_temp_c` directly to `step()`, adds unit regression tests, validates the sampler offline with $N \ge 1,000$ episodes, and reruns Experiment C with full per-episode logging.
