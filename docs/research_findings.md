# Research Findings — Long-Horizon PPO Stability

## Summary

This document records the honest outcome of the long-horizon (1M-step) training
experiments for both the charging controller (Track A) and the driving EMS
(Track B). These findings are **kept in the research record intentionally** —
they represent real empirical observations, not failures to hide.

---

## Track A — Charging Controller Long-Training Failure

### What was attempted

After validating the A1 50k candidate, a 1M-step long-training run was launched
from the same A1 configuration to see whether additional training would further
improve charging time.

**Run:** `runs/charging_final_1m_v2/`  
**Checkpoints evaluated:** 39 × 25k-step intervals from 25k to 1M steps

### What happened

| Checkpoint range | Target-reach behaviour |
| :--- | :--- |
| ~50k–250k steps | Target SoC reached (consistent with A1 50k result) |
| ~250k–1M steps | Progressive collapse — target-reach rate degrades |
| Final 1M endpoint | **Fails primary charging objective** |

The 1M-step model (`trained_model.zip`) cannot reliably complete the 20→80% SoC
charge cycle. It is **not suitable** as a final deliverable.

### Why this matters

This result demonstrates a well-known phenomenon in deep RL:

> **Short-horizon validation does not guarantee long-horizon PPO stability.**

A PPO agent can begin to overfit its policy to a narrow region of the value
landscape under prolonged training, causing it to lose behaviours that were
previously well-established. This is especially likely without:

- Learning-rate annealing tuned to prevent late-stage drift
- Checkpoint selection based on validation performance (not final step)
- KL-divergence or entropy regularisation appropriate for long runs

### Research conclusion (Track A)

The **validated deliverable is the A1 50k candidate** (`final_models/charging_A1_50k_seed*/`).

The 1M-step result is a documented failure case that:
- Confirms the need for policy stability monitoring in long RL runs
- Provides empirical data on the collapse trajectory (see `audit/charging_stage4_collapse_investigation.md`)
- Motivates future work on curriculum scheduling and checkpoint selection

---

## Track B — Driving EMS Long-Training Status

### What was attempted

Three-seed 1M-step training runs were launched for the driving PPO EMS
(seeds 7, 21, 42), producing checkpoints every 50k steps.

**Runs:** `runs/driving_final_1m_v2_seed{7,21,42}/`

### What is established

The **B3 100k checkpoint** (50k-step interval checkpoint at step 100k) was
fully validated across all four standard drive cycles:

| Metric | Seed 7 | Seed 21 | Seed 42 |
| :--- | :---: | :---: | :---: |
| Mean Wh/km (WLTP) | 117.75 | 119.53 | 119.01 |
| Regen recovery | 99.88% | 97.36% | 97.70% |
| Safety interventions | 0 | 0 | 0 |
| Power deficit (Wh) | 38.15 | 0.0 | 11.63 |

### What is NOT established

The **final 1M-step trained model** for driving was evaluated (step 6 of the
final evaluation script), and raw results were written to `audit/driving_ppo_benchmark.csv`.
However, **no formal comparison against the B3 100k validation gate** was
performed for the 1M endpoint. The 1M driving result is therefore:

> ⚠️ **Unverified** — present in audit files, not formally gated.

The driving 1M model performance appears plausible from the raw numbers, but
it has not been subject to the same structured validation protocol as B3 100k.

### Research conclusion (Track B)

The **validated deliverable is the B3 100k candidate** (`final_models/driving_B3_100k_seed*/`).

The 1M driving checkpoints remain available for future evaluation. Completing
the driving gate evaluation is the most direct path to claiming a final 1M
driving result.

---

## Overall Project Status

| Claim | Supported? |
| :--- | :---: |
| Core RL-BMS implementation is complete | ✅ Yes |
| 214/214 unit tests pass | ✅ Yes |
| Charging PPO learns meaningful control | ✅ Yes (A1 50k) |
| Driving PPO achieves competitive efficiency | ✅ Yes (B3 100k) |
| Long-horizon charging training is stable | ❌ No — 1M collapses |
| Long-horizon driving training is validated | ⚠️ Partially — raw results exist, gate not applied |

---

## Recommended Citation Language

When describing results in a report or presentation, use:

> *"The charging PPO controller was validated at the A1 50k checkpoint,
> achieving a mean charge time of 2101 s across seeds 7, 21, and 42
> (gate: ≤ 2199 s), with 15/15 standard scenarios completed.
> Extended training to 1M steps did not preserve this performance,
> indicating long-horizon instability that remains an open research question."*

> *"The driving EMS PPO controller was validated at the B3 100k checkpoint,
> achieving a mean of 128.8 Wh/km on WLTP Class 3b with zero safety
> interventions across three random seeds."*
