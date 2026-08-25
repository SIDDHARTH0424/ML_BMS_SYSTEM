# Track A Gate Re-Validation Through the Real `build_agent()` Pipeline

**Purpose**: `audit/charging_stage4_collapse_investigation.md` recommended
two paths forward; this executes option 2 ("keep the production LR
schedule and re-run the short gate through the actual
`agents/train_ppo.py::build_agent()` path") exactly as specified: same
Candidate A1 config, same 50,000-step budget, same 3 seeds (7, 21, 42),
same evaluation logic (copied verbatim from
`experiments/run_readiness_diagnostics.py`), only the model-construction
call changed from a raw `PPO(...)` to `build_agent()`.

## Result: the gate fails on all 3 seeds

| Seed | Charging Time (s) | Δ vs 2094.6s baseline | vs 2199.3s gate | Reached | Mean Applied Current (A) |
|---|---|---|---|---|---|
| 7 | 2577.60 | +23.06% | **FAIL** | 15/15 | 127.1 |
| 21 | 2671.00 | +27.52% | **FAIL** | 15/15 | 122.3 |
| 42 | 2368.93 | +13.10% | **FAIL** | 15/15 | 137.9 |

**Gate conditions**: A1 PASS (all reach target), **A2 FAIL** (all 3
seeds exceed the ±5% threshold, by 13-28%), A3 PASS (derating still
present: 129A normal → 102A at 45°C → 58A at 50°C), A4 PASS (peak
temp beats Max Current's 50.75°C on all 3 seeds at 45°C stress),
**A5 FAIL** (mean normal-temp current 129.1A, well below the ~155-158A
the flat-LR diagnostic achieved — the policy is measurably less
aggressive, not just slower to converge). A6 PASS (no catastrophic
failure — always reaches target, just slowly). A7 not evaluated by
this script's summary stats.

**This confirms, with the full 3-seed gate (not just seed 7's time
figure from the earlier investigation), that Candidate A1 does not
pass Track A's gate when trained through the real production pipeline
at this budget (with the pre-fix decaying-LR `build_agent()`).** Full
data: `audit/charging_A1_regate_prefix_seed{7,21,42}_standard.csv`,
`_stress.csv`, combined in
`audit/charging_A1_regate_prefix_standard_combined.csv` /
`_stress_combined.csv`. (Filenames carry a `_prefix_` tag added when
these files were regenerated after the LR-schedule fix landed — the
original run overwrote these same files under their un-prefixed names
before the fix's PASS re-run could be safely captured under distinct
names; see `audit/charging_A1_regate_flat_lr_fixed_report.md` for that
correction and the resulting `_postfix_` files, and
`experiments/charging_A1_regate_prefix_evidence_regen.py` for the
one-off script used to regenerate this exact evidence, confirmed
byte-for-byte identical to the original numbers reported above.)

## Important nuance discovered while running this: Stage 3 and Stage 4 have independent, differently-anchored LR schedules

This result should not be over-generalized to "Track A's real pipeline
always fails, at every budget." Inspecting `agents/train_ppo.py::run_stage()`
directly:

```python
timesteps_by_stage = {
    1: ppo_cfg["stage1_sanity_timesteps"],       # 5,000
    2: ppo_cfg["stage2_reward_verification_timesteps"],  # 20,000
    3: ppo_cfg["stage3_hpo_timesteps"],           # 50,000
    4: ppo_cfg["stage4_full_training_timesteps"], # 1,000,000
}
total_timesteps = timesteps_by_stage[stage]
...
model = build_agent(env, ppo_cfg, ...)  # fresh model, every stage
model.learn(total_timesteps=total_timesteps, ...)
```

Each stage builds a **fresh** model and calls `learn(total_timesteps=<that stage's own budget>)`.
`linear_schedule`'s decay is computed against whatever `total_timesteps`
is passed to that specific `learn()` call — so **each stage's LR decay
is anchored to its own budget, independently**:

- **Stage 3 in isolation** decays LR from ~3e-4 to ~0 across its own
  50,000 steps — by the end of Stage 3, LR has fully decayed
  (confirmed: `learning_rate: 5.09e-06` in this run's final training
  log for all 3 seeds). **This experiment is an exact reproduction of
  what production Stage 3 does, and it genuinely fails the gate.**
- **Stage 4 in isolation** decays LR from ~3e-4 to ~0 across its own
  1,000,000 steps. At Stage 4's *own* 100,000-step mark (10% of its
  schedule), LR has barely moved (~2.7e-4, confirmed in
  `audit/charging_stage4_repro_checkpoints.csv`'s properly-seeded first
  chunk) — and the resulting policy at that point looks healthy:
  **2094.8s, 100% reached, 155.9A mean current — matching the original
  flat-LR diagnostic almost exactly.**

These are genuinely different training runs, not the same schedule
sampled at two points. **This document's 50k result indicts Stage 3
specifically** (relevant if Stage 3's checkpoint is ever used as a
standalone deliverable, and relevant as further confirmation that the
staged pipeline's intermediate checkpoints are not simply "a slower
version of the same policy" — Stage 3's own complete training run
converges to a measurably worse solution than either the flat-LR
diagnostic or Stage 4's early trajectory). **It does not, by itself,
resolve whether Stage 4 — the actual long-training stage, with its own
independently-anchored 1,000,000-step schedule — will end well.** That
remains governed by `audit/charging_stage4_collapse_investigation.md`'s
findings: healthy through ~165k, episode length reversing from there,
value-function volatility rising sharply after ~400k, endpoint not
independently confirmed (investigation stopped at 450,560/1,000,000
steps on compute budget).

## Updated recommendation

Track A remains `NOT_READY`. This result adds a second, independent,
confirmed failure specific to Stage 3, on top of the Stage-4 drift
evidence already documented. Two separate problems now have direct
evidence:

1. **Stage 3's own 50k-step run is not a usable short-training
   checkpoint** — if the project ever wants a fast/short-budget
   charging model, it should come from the properly-gated Candidate A1
   diagnostic (flat LR) or from Stage 4's *own* schedule sampled early
   (e.g. 100k steps into Stage 4's 1M-step run, which looked healthy),
   not from Stage 3 as currently configured.
2. **Stage 4's full 1,000,000-step run remains unvalidated at its
   endpoint** — the drift precursors found through 450k are real
   grounds for caution, but neither confirm nor rule out the originally
   reported 7200s/0% collapse at 1M.

Neither problem is fixed by this document; both require either (a)
removing/adjusting the LR schedule so a 50k-budget stage behaves
consistently with what was gated, or (b) accepting the schedule as
designed and re-gating specifically against Stage 4's own budget with
proper checkpoint-based validation during the run, as originally
recommended.
