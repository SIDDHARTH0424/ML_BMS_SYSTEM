# Track A — Investigation: Reported Long-Training (Stage 4) Collapse

**Trigger**: an external analysis reported that the actual staged
long-training pipeline (`training.train --stages 1 2 3 4`, using
`configs/final_charging`) produces a Stage 4 (1,000,000-step) model
that fails catastrophically — 7200s charging time (= episode timeout),
0% target reached, vs Stage 1-3 all reaching target but progressively
slower than the gated Candidate A1 diagnostic. This contradicts
`audit/charging_final_candidate_evaluation.md`'s 7/7-PASS conclusion,
which was based entirely on `experiments/run_readiness_diagnostics.py`
— a *different* training script from the actual production entry
point (`agents/train_ppo.py::run_stage()` / `training/train.py`).

This document reports an independent investigation into whether the
short-gate evidence actually transfers to the real long-training
pipeline. **Bottom line: it does not, fully — this is a genuine,
previously-undisclosed validation gap, not just a documentation
issue.**

## 1. Confirmed: Stage 1-3's slower-than-gated results are real and reproducible

`agents/train_ppo.py::build_agent()` (the real pipeline) constructs
PPO with:
```python
learning_rate=linear_schedule(ppo_cfg["learning_rate"]),
```
— a schedule that linearly decays the learning rate from the config
value toward ~0 over the course of `total_timesteps`. The gated
Candidate A1 diagnostic (`experiments/run_readiness_diagnostics.py`)
instead passes `learning_rate=ppo_cfg["learning_rate"]` as a **plain
constant float**. This is a genuine code-path difference between what
was gated and what the frozen config actually runs through — it was
not visible in my earlier "does `--config-dir` load correctly"
verification, because that only checked the raw config *values*
(`effective_ppo_stage1.yaml`), not how `build_agent()` *uses* the
`learning_rate` field at runtime.

**Reproduced directly** (`experiments/charging_lr_schedule_repro.py`):
training via the real `build_agent()` pipeline, same seed (7), same
50,000-step budget as Stage 3 and as the gated Candidate A1 run,
using `configs/final_charging` throughout:

| | Mean charging time | Reached |
|---|---|---|
| Gated Candidate A1 diagnostic (flat LR) | 2114.53s | 15/15 |
| Real pipeline (`linear_schedule` LR) | **2577.60s** | 15/15 |

**2577.60s matches the externally-reported Stage 3 result to the
decimal.** This confirms the LR-schedule difference is real,
reproducible, and fully explains the Stage 1-3 slowdown — at this
budget the policy still reaches target every time, just more slowly
(average LR over the run is lower than the diagnostic's constant
value). This is a genuine short-gate/production mismatch: **the thing
that was gated (flat-LR PPO) is not exactly the thing the frozen
config actually trains with (decaying-LR PPO).**

## 2. An earlier reproduction of "Stage 4 collapse" in this
   investigation was itself a false positive — corrected here

A first attempt (`experiments/charging_stage4_collapse_repro.py`)
chunked training via `PPO.save()`/`PPO.load()` across separate process
invocations to reach checkpoint targets without exceeding sandbox time
limits. It appeared to reproduce collapse almost exactly: 100k steps →
2094.8s/100% reached; 200k steps → **7200.0s/0% reached, current
dropped to 44.6A**. This looked like a clean confirmation.

**It was not.** The script's seeding call
(`set_global_seed(SEED)`) only runs in the fresh-model branch — **not
when resuming from a saved checkpoint** (confirmed by inspection:
`grep -n set_global_seed` shows exactly one call site, inside the
`else` branch). The 100k→200k chunk therefore ran with an
uncontrolled RNG state, not a reproducible continuation of the same
trajectory a real uninterrupted `model.learn(total_timesteps=1_000_000)`
call would follow. **This result is retracted as unreliable evidence**
and should not be cited as a reproduction of the reported collapse.
Kept in the repo (`experiments/charging_stage4_collapse_repro.py`,
`audit/charging_stage4_repro_checkpoints.csv`) for transparency, with
this correction attached, rather than deleted.

## 3. Properly-controlled investigation: genuine drift precursors found, full collapse not yet confirmed

A second, single continuous process (`experiments/charging_stage4_collapse_diagnostic.py`)
trains via the real `build_agent()` pipeline from step 0, seeded once,
never interrupted/resumed — eliminating the reseeding bug above. SB3's
CSV logger captured full per-chunk curves. Reached **450,560 steps**
before hitting the sandbox's compute budget (full 1,000,000 was not
reached — this remains a real limitation of this investigation, not a
"no collapse" conclusion).

**0 → ~200k: healthy.** `approx_kl` stayed low (0.002-0.008, well
under the 0.02 alarm and the 0.01 target), `explained_variance` climbed
smoothly to 0.87-0.98, `ep_rew_mean` improved substantially, and
`ep_len_mean` **decreased** steadily from 3711 to a minimum of
**2771.2 at step 163,840** — shorter episodes in this task mean faster
charging completion, i.e. the policy was getting *better*, not worse,
through this window. This directly contradicts the retracted §2 result
at the same step range.

**~165k onward: a real, quantified reversal.** `ep_len_mean` turns
around after its 163,840-step minimum and climbs steadily back up
through the rest of the observed range, reaching 3925.3 by step
450,560 — longer than where it started at step 8,192 (3711). Episodes
are taking *longer* again, consistent with the policy drifting toward
slower/more conservative charging.

**~400k onward: value-function instability emerges.** `value_loss`
volatility (std) jumps from 15.5 (steps <400k) to 85.0 (steps ≥400k)
— a 5.5x increase — with individual spikes to 129.6 followed by a
near-immediate drop to 0.7, then 20.2, 76.1, 1.4: erratic, not a
smooth trend. `explained_variance` volatility roughly doubles over the
same boundary (std 0.158 → 0.329) and swings as low as 0.56 having
been consistently >0.95 earlier. This is a plausible **early-warning
signature** of the value function starting to lose track of a moving
policy target — a known PPO failure precursor — though it had not yet
produced a reward collapse by step 450,560 (`ep_rew_mean` was still
in the same -350 to -450 range it had occupied since ~250k, not
crashing).

**What this does and doesn't show**: the reported catastrophic Stage 4
outcome (7200s / 0% reached at 1,000,000 steps) was **not
independently reproduced** in this investigation — compute budget ran
out at 450,560 steps, well short of 1,000,000, with reward not yet
collapsed. But the trajectory through that point is genuinely
consistent with continued drift toward exactly the reported failure
mode: episode length reversing upward (slower charging), and value-
function volatility increasing sharply in the back half of the
observed range. This is meaningfully different from either "config
mismatch, otherwise fine" or "definitely genuinely collapses" —
it's real evidence of drift, without full confirmation of the
endpoint.

## 4. Current assessment

- **§1 (LR-schedule mismatch) is confirmed, reproduced, and
  sufficient on its own to invalidate treating the flat-LR Candidate
  A1 short-gate result as validating the real long-training pipeline's
  short-budget behavior.** This alone means Track A's "READY" claim
  needs a caveat: it was validated for a training procedure that
  differs from the one the frozen config actually runs.
- **§3 (drift precursors)** independently corroborates the external
  report's core concern — genuine training-dynamics drift over a long
  run, not merely a config-file mismatch — via a properly-controlled
  reproduction reaching 45% of the full budget, without fully
  confirming the endpoint.
- **§2 is a retraction**: an earlier finding in this same
  investigation looked like a clean confirmation but was traced to a
  bug in the investigation's own script, not genuine evidence.

## 5. Recommendation (updates Track A status)

**Track A reverts to `NOT_READY` for the 1,000,000-step long-training
claim**, pending either:
(a) fixing the LR-schedule mismatch — either remove `linear_schedule`
    from `build_agent()` (reverting to what was actually gated) or
    re-run the short gate through `build_agent()` itself so the gated
    procedure and the production procedure are the same code path, and
(b) checkpoint-based validation during any long run, exactly as
    externally recommended: evaluate the standard grid periodically
    (this investigation suggests every 50k-100k steps given where
    drift began) and select/restore the best validated checkpoint
    rather than trusting the final one. The `value_loss`/
    `explained_variance` volatility onset around step 400k is a
    reasonable place to prioritize close monitoring if a full run is
    attempted.

This does not affect Track B, which used a from-scratch, flat-LR,
100,000-step short-gate procedure (`experiments/driving_candidate_B3_100k.py`)
that has no long-training-pipeline equivalent yet to diverge from —
Track B's status is unaffected by this investigation.

## 6. Follow-up: option (a) executed — result

Recommendation (a)'s "re-run the short gate through `build_agent()`
itself" was carried out exactly as specified: same Candidate A1
config, same 50k budget, same 3 seeds, evaluated with the same logic.
**Result: FAIL on all 3 seeds** (charging time +13% to +28% over the
gate threshold; normal-temperature current also drops from ~155-158A
to ~122-138A, so this is not merely slower convergence to the same
policy). Full detail, including an important discovered nuance about
Stage 3 and Stage 4 having independently-anchored LR schedules (this
result specifically indicts Stage 3, not necessarily Stage 4):
`audit/charging_A1_regate_via_build_agent_report.md`.

Track A status is unchanged by this (`NOT_READY` was already the
conclusion) — this adds a second, independently confirmed failure mode
on top of the Stage 4 drift evidence in §3 above.

## 7. Fix applied and re-verified: LR-schedule mismatch closed

`agents/train_ppo.py::build_agent()` was subsequently fixed to pass
`learning_rate=ppo_cfg["learning_rate"]` directly (a constant),
removing `linear_schedule()` entirely. Re-running the exact same §6
test against this fix: **all 3 seeds now PASS the gate** (2115.53s /
2095.00s / 2095.27s — within ~1s of the original flat-LR diagnostic's
numbers for every seed). Full detail:
`audit/charging_A1_regate_flat_lr_fixed_report.md`.

**This closes the confirmed Stage 3 failure mode from §6.** It does
**not** by itself confirm Stage 4's full 1,000,000-step run — §3's
drift precursors were observed under the *old* decaying-LR
`build_agent()`, so that specific mechanism no longer applies as
described, but no full long run has yet been executed against the
*fixed* flat-LR pipeline. Track A remains `NOT_READY` pending a
checkpointed 1,000,000-step run with the fix in place.

## Files produced by this investigation

- `experiments/charging_lr_schedule_repro.py`, output above (§1, confirmed)
- `experiments/charging_stage4_collapse_repro.py`, `audit/charging_stage4_repro_checkpoints.csv` (§2, retracted -- reseeding bug)
- `experiments/charging_stage4_collapse_diagnostic.py`, `audit/charging_stage4_collapse_diagnostic_curves_0to200k.csv`, `audit/charging_stage4_collapse_diagnostic_curves_0to450k_partial.csv` (§3, the trustworthy evidence)
- `experiments/charging_A1_regate_via_build_agent.py`, `audit/charging_A1_regate_via_build_agent_report.md` (§6, full 3-seed gate re-run, confirmed FAIL for Stage 3's own schedule)
