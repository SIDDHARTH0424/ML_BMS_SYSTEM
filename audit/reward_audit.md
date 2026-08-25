# Reward Function Audit

Source: `environment/battery_env.py::_compute_reward` (per-step) and
`::step` (terminal bonuses/penalties), weights from `configs/reward.yaml`.
All formulas and weights below are transcribed directly from those two
files — no values were invented. Per-step magnitude ranges are computed
from the weight × the variable's own physical range (also taken from
config, e.g. `i_max_a`, `temperature_penalty_start_c`), not measured from
an actual training run (none exists in this project — see ISSUE-005).

## Per-step components

| Component | Formula | Weight | Source variable | Sign | Active when | Approx. per-step magnitude |
|---|---|---|---|---|---|---|
| `charging_progress` | `w * ΔSoC` | 1000.0 | `new_state.soc - prev_state.soc` | + | every step | ~0.37 at full current (160 A, 121 Ah pack, dt=1s → ΔSoC=3.67e-4/step; documented derivation in `configs/reward.yaml`) |
| `temperature_penalty` | `w * max(0, T - 40°C)` | 0.05 | `new_state.temperature_c` | − | T > 40°C | 0 below 40°C; ~0.5 at 50°C excess (10°C over threshold) |
| `safety_penalty` | `w * safety_info.magnitude` | 5.0 | fractional clamp magnitude, `[0,1]` | − | safety layer clamps the request at all | 0 to 5.0 |
| `overrequest_penalty` | `w * (requested−applied)/i_max` | 2.0 | absolute wasted current, normalized | − | request exceeds the safe ceiling | 0 to 2.0 |
| `smoothness_penalty` | `w * |I_t − I_(t-1)|/i_max` | 0.5 | step-to-step current change | − | every step after the first | 0 to 0.5 |
| `time_penalty` | constant `w` | 0.05 | none (unconditional) | − | every step | 0.05 |

## Terminal (one-time) components

| Component | Formula | Weight | Active when |
|---|---|---|---|
| `target_reached_bonus` | constant | 50.0 | episode ends via `target_soc_reached` |
| `terminal_shortfall_penalty` | `w * max(0, target_soc − final_soc)` | 1000.0 | episode truncates without reaching target (e.g. 0.025 shortfall → ~25) |
| `overvoltage_penalty` | constant | 20.0 | episode ends via `overvoltage` |
| `overtemperature_penalty` | constant | 20.0 | episode ends via `overtemperature` |

## Dominance analysis

Per-step, the largest possible single-step penalty is `safety_penalty`
(max 5.0) if the request is fully blocked, but `overrequest_penalty` (max
2.0) and `smoothness_penalty` (max 0.5) can co-occur with it. Against
that, `charging_progress` contributes at most ~0.37/step at full current —
meaningfully **smaller** than a fully-triggered `safety_penalty` alone.
This is intentional per the `configs/reward.yaml` comments: the weights
were explicitly re-derived (documented as a "v2"/"v3" correction) after an
earlier version let the agent's easiest optimum be "minimize penalties"
rather than "charge" (progress weight raised from 10 → 1000 specifically
to fix this — see the yaml file's own derivation comment).

No single per-step component structurally dominates all others across the
full range: `charging_progress` dominates when the episode is running
cleanly (no safety intervention, smooth current, low temperature), while
`safety_penalty`/`overrequest_penalty` dominate specifically when the
agent over-requests into the safety ceiling. This asymmetry (penalties can
outweigh progress only when the agent is actively misbehaving) is
consistent with the design intent stated in the reward comments, not a
sign of an unexamined imbalance.

**Terminal components dwarf all per-step components** (50–1000 vs. ≤5 per
step), by design — they are one-time episode-outcome signals, not
per-step shaping. This is expected for a sparse task-completion bonus and
is not a bug.

## What this audit does NOT claim

This document reports the *configured* weights and their *theoretical*
ranges. It does not report *empirically observed* reward component
distributions from an actual training run, because no run artifacts
(`reward_components.csv` from a real Stage 2 run) exist in this project
to analyze — consistent with ISSUE-005. `results_and_discussion.md`
Section 2 describes empirical findings; those are treated as unverified
against this codebase per ISSUE-005, not incorporated here as if
confirmed.

**No reward weights were changed in this pass.**
