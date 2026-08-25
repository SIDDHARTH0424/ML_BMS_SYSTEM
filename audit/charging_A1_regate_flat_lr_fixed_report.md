# Track A Gate Re-Validation: Production Pipeline, Post-Fix (Flat LR)

**Trigger**: `agents/train_ppo.py::build_agent()` was fixed to pass
`learning_rate=ppo_cfg["learning_rate"]` directly (a constant), removing
the `linear_schedule()` wrapper that
`audit/charging_A1_regate_via_build_agent_report.md` found caused
Candidate A1 to fail the gate when trained through the real pipeline
(all 3 seeds, +13% to +28% over the time threshold). This is the
decisive re-test: same script
(`experiments/charging_A1_regate_via_build_agent.py`), same config,
same seeds, same evaluation logic — only the underlying `build_agent()`
implementation changed (by the fix, not by this test).

## Result: PASSES, and closely matches the original short-gate diagnostic

| Seed | Time (this run) | Time (original flat-LR diagnostic) | Δ | Reached | Applied Current |
|---|---|---|---|---|---|
| 7 | 2115.53s | 2114.53s | +1.00s | 15/15 | 154.4A |
| 21 | 2095.00s | 2095.00s | +0.00s | 15/15 | 155.9A |
| 42 | 2095.27s | 2095.13s | +0.14s | 15/15 | 155.9A |

All three seeds now reproduce the original diagnostic's numbers to
within ~1 second — consistent with `build_agent()`'s flat LR now being
functionally equivalent to the diagnostic's own PPO construction
(both pass a constant `ppo_cfg["learning_rate"]`, both use
`ent_coef=0.01`, `target_kl=0.01`, `use_sde=False`). The residual
sub-second differences are consistent with the two code paths still
being separate implementations (not the exact same function call) —
e.g. `run_readiness_diagnostics.py` passes `use_sde=False` explicitly
as a keyword while `build_agent()` relies on it being in `ppo_cfg` (it
is, in `configs/final_charging/ppo.yaml`) — not a remaining
discrepancy of concern.

## Gate Evaluation

| # | Condition | Result |
|---|---|---|
| A1 | 3/3 seeds reach 95% SoC, standard grid | **PASS** (45/45) |
| A2 | Every seed ≤ 2199.3s | **PASS** (max 2115.53s, seed 7 — 84s of margin) |
| A3 | Measurable stress-dependent derating | **PASS** (155A → 108A → 57A) |
| A4 | ≥2/3 seeds lower peak temp than Max Current (50.75°C) at 45°C stress | **PASS** (3/3: 49.50/49.81/49.81°C) |
| A5 | Normal-temp behavior stays aggressive | **PASS** (155.4A mean vs 160A ceiling) |
| A6 | No catastrophic standard-grid failure | **PASS** |
| A7 | Training numerically stable | Not evaluated by this script's summary stats (no NaN/Inf/crash observed across all 3 seeds' training logs) |

**6/6 automatically-checked conditions PASS**, closely reproducing the
original short-gate result. Full data:
`audit/charging_A1_regate_postfix_seed{7,21,42}_standard.csv`,
`_stress.csv`, combined in
`audit/charging_A1_regate_postfix_standard_combined.csv` /
`_stress_combined.csv`. **Correction**: these files were initially
saved under the same un-prefixed names as the pre-fix FAIL results in
`audit/charging_A1_regate_via_build_agent_report.md`, overwriting that
evidence. This was caught and fixed: the PASS results here were
renamed to `_postfix_`, and the original FAIL evidence was
byte-for-byte regenerated (build_agent() temporarily reverted to the
old decaying-LR version, exact same 3 seeds re-run, confirmed to
reproduce the original numbers exactly) and saved under `_prefix_`
names — see `audit/charging_A1_regate_via_build_agent_report.md`'s
updated file references.

## What this does and doesn't establish

**Establishes**: with the LR-schedule fix applied, Stage 1-3 (which
each run their own short, from-scratch training using `build_agent()`)
should now behave consistently with what was originally gated —
directly addressing the confirmed Stage 3 failure mode from
`audit/charging_A1_regate_via_build_agent_report.md`.

**Does not establish**: this is still a 50,000-step test. It says
nothing new about Stage 4's full 1,000,000-step run specifically,
beyond removing one confirmed source of divergence between it and the
short-gate evidence. `audit/charging_stage4_collapse_investigation.md`
§3's drift precursors (episode length reversing after ~165k steps,
value-function volatility rising after ~400k) were observed using the
*old* decaying-LR `build_agent()` — with the schedule now removed,
that investigation's specific mechanism no longer applies as
described, but a full 1,000,000-step run with the *fixed* flat-LR
`build_agent()` has not yet been run or evaluated at checkpoints.
**A full long-training run with periodic checkpoint evaluation (as
previously recommended) is still required before Track A can be
called `READY_FOR_LONG_TRAINING`** — this document closes the
short-gate/production mismatch, not the long-horizon question.

## Updated Track A status

**`NOT_READY` remains the correct status pending a checkpointed long
run**, but the specific, confirmed short-budget failure mode (Stage 3
producing a measurably slower, less aggressive policy than gated) is
now resolved. This is a genuine, verified improvement, not yet a full
clearance.
