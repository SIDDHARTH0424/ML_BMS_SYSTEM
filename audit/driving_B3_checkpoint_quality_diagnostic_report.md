# Track B — Candidate B3 Checkpoint-Quality Diagnostic (closes B8 gap)

**Purpose**: `audit/long_training_freeze_v2.md` marked gate condition
B8 (training stability) "PASS (qualified)" because the original
`Cand_B3_100k` run only captured end-of-training evaluation, not
per-chunk training curves. This is a diagnostic-only re-run of the
**identical, unchanged** frozen B3 config
(`w_regen_recovery=1.4, w_energy_cost=0.2, w_tracking_error=1.5,
ent_coef=0.01`, 100,000 steps/seed) with stable-baselines3's CSV
logger attached — no reward search, no hyperparameter change.
`experiments/driving_B3_checkpoint_quality_diagnostic.py` imports
`CAND_WEIGHTS`/`ENT_COEF`/`TIMESTEPS` directly from the original
`driving_candidate_B3_100k.py` so there is no possibility of drift
between what was gated and what was curve-checked.

**Reproduction sanity check** (same seeds, same config, run to
completion a second time): all 3 seeds reproduced the originally
gated evaluation numbers exactly —

| Seed | Original Gate Wh/km | This Run's Wh/km | Original Regen | This Run's Regen |
|---|---|---|---|---|
| 7 | 128.53 | 128.53 | 99.96% | 100.0% |
| 21 | 129.34 | 129.34 | 96.67% | 96.7% |
| 42 | 128.92 | 128.92 | 98.87% | 98.9% |

This confirms the curve-logged run is genuinely the same training
process the gate evaluated, not a different one.

**Data**: 49 chunks/seed (147 rows total, `n_steps=2048`), saved to
`audit/driving_Cand_B3_100k_training_curves_seed{7,21,42}.csv` and
combined in `audit/driving_Cand_B3_100k_training_curves_combined.csv`.

## Findings

### No NaN/Inf
Zero NaN or Inf values in any of `train/approx_kl`,
`train/explained_variance`, `train/value_loss`, `train/entropy_loss`,
`train/std`, `rollout/ep_rew_mean`, `rollout/ep_len_mean` across all
147 chunks (after the expected first-chunk NaN at step 2048, before
any policy update has occurred — standard SB3 behavior, not an
anomaly).

### approx_kl — one isolated exceedance, otherwise well-controlled
| Seed | Max approx_kl | Mean approx_kl |
|---|---|---|
| 7 | 0.0102 | 0.0046 |
| 21 | 0.0143 | 0.0048 |
| 42 | 0.0219 | 0.0056 |

`target_kl=0.01` triggers early-stopping within a rollout's epoch
loop, so the *logged* approx_kl (computed post-hoc over completed
epochs) can occasionally land above target_kl at an epoch boundary.
**One single chunk out of 147** (seed 42, step 40,960) reached 0.0219,
marginally above the project's stated "generally ≤0.02" tolerance.
This is disclosed rather than rounded away: it is a single transient
event with no accompanying divergence in `value_loss`, `entropy_loss`,
or `ep_rew_mean` in that chunk or the ones around it (seed 42's
`ep_rew_mean` continued improving smoothly through and past step
40,960 — see combined CSV), so it reads as ordinary PPO noise, not
instability. All other 146/147 chunks across all 3 seeds are at or
below 0.02, and the great majority are below 0.01.

### explained_variance — positive well before the final chunk
| Seed | First chunk (step 4096) | Min after warmup | Last chunk (step 100352) |
|---|---|---|---|
| 7 | -0.007 | -0.007 | 0.904 |
| 21 | -0.096 | -0.096 | 0.869 |
| 42 | 0.099 | 0.099 | 0.863 |

Near-zero/negative at the very first chunk (expected — the value
function hasn't learned anything yet) then climbs and stabilizes in
the 0.85–0.94 range for the bulk of training, ending at 0.86–0.90 for
all 3 seeds. Condition A7/B8's "explained_variance positive by final
chunk" is satisfied with a wide margin, not just barely.

### Learning progress (rollout/ep_rew_mean)
| Seed | First chunk | Last chunk | Improvement |
|---|---|---|---|
| 7 | -71.5 | -23.1 | +48.4 |
| 21 | -70.4 | -17.3 | +53.0 |
| 42 | -74.0 | -14.6 | +59.4 |

Monotonic-ish improvement of 48–59 reward points across training for
all 3 seeds — consistent with genuine learning, not a policy that
plateaued early or degraded.

`train/std` (policy standard deviation) decreases smoothly from
~1.0–1.04 to ~0.65–0.78 across training for all seeds — consistent
with the policy sharpening around a solution rather than collapsing
abruptly or oscillating.

## Updated Gate Condition B8

**PASS — no longer qualified.** Full per-chunk curves now exist for
all 3 seeds: zero NaN/Inf, approx_kl controlled (one isolated,
disclosed exceedance of 0.0219 vs the ~0.02 tolerance, out of 147
chunks), explained_variance strongly positive by the final chunk, and
monotonic learning progress. This supersedes the "qualified" language
in `audit/long_training_freeze_v2.md` §1 and `audit/driving_power_deficit_and_full_gate.md` §3.

This was a targeted instrumentation pass, not a new reward search —
consistent with the request to close the weakest evidence gap without
reopening decisions the short gate already settled.
