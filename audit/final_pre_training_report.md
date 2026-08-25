# Final Pre-Training Report

This report covers the final pre-training fix pass. It builds on three
prior audit passes (see `audit/ISSUES.md` ISSUE-001 through ISSUE-010,
`audit/STAGE_PIPELINE.md`, `audit/reward_audit.md`, `audit/before_after.md`)
and addresses the remaining items required before a full Stage-4 training
run: ISSUE-011 through ISSUE-017 below.

## 1. Files changed

Functional changes:
- `agents/train_ppo.py` — `run_stage()` now accepts an optional `run_dir`
  parameter; per-stage artifacts (`effective_ppo.yaml`, `command.txt`,
  `trained_model.zip`) namespaced per stage.
- `training/train.py` — creates the run directory exactly once, reuses it
  across all stages.

Documentation-only changes:
- `README.md` — added an explicit action-space statement (Section was
  previously silent on this, not contradictory).
- `all_rl_bms_code.py` — regenerated via `python combine_all_code.py`
  (mechanical, not hand-edited).
- `audit/ISSUES.md` — ISSUE-011 through ISSUE-017 added.

New test files (no existing test was modified to make it pass):
- `tests/test_multistage_run_dir.py`
- `tests/test_action_mapping.py`
- `tests/test_environment_invariants.py`
- `tests/test_checkpoint_saving.py`

No changes to: `environment/battery_env.py`, `environment/ecm_model.py`,
`safety/safety_layer.py`, any `configs/*.yaml`, `utils/metrics.py`,
`utils/logger.py`, `training/select_best_checkpoint.py`, or
`training/evaluate.py`.

## 2. Exact changes made

**ISSUE-011 (Critical) — multi-stage run-directory crash.**
`training/train.py::main()` previously called `run_stage(stage,
run_name=args.run_name)` for every stage, and `run_stage` unconditionally
called `create_run_dir(run_name=...)` every time. Since Stage 1 writes
files into the run directory, Stage 2's `create_run_dir` call hit the
(correct, unchanged) non-empty-directory guard and raised
`FileExistsError` — **confirmed by direct reproduction before fixing**
(see `audit/ISSUES.md` ISSUE-011 for the reproduction). Fixed by creating
the run directory exactly once in `training/train.py`, before the stage
loop, and passing that same `run_dir` into every `run_stage()` call;
`run_stage` now only calls `create_run_dir` when no `run_dir` is supplied
(preserving the single-stage CLI path and the overwrite guard exactly as
before). Also fixed the artifact-loss this exposed: each stage's model is
now saved to `trained_model_stage{N}.zip` (preserved across stages)
alongside a `trained_model.zip` copy pointing at the most-recently-
completed stage (for backward compatibility with
`training/select_best_checkpoint.py`/`training/evaluate.py`); per-stage
config/command snapshots are similarly namespaced.

**ISSUE-012 — README action-space documentation.** Added an explicit
`Box(low=-1, high=1, shape=(1,))` statement to `README.md` (previously
absent, not contradictory — `docs/model_specification.md` and
`docs/model_contract.md` already stated this correctly).

**ISSUE-013 through ISSUE-017 — verification only, no code changed.**
Action clipping, safety-layer ordering, reward finiteness, observation
finiteness, and checkpoint save/load were all traced through the actual
code and confirmed already correct. New regression tests were added for
each (none previously existed) rather than changing any source.

## 3. Tests executed

```
python -m compileall .
python -m pytest -q
```

in the project's declared environment (Python 3.12, `requirements.txt`
installed exactly as specified: gymnasium 1.3.0, stable-baselines3 2.9.0,
torch 2.13.0, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, PyYAML 6.0.3,
pytest 9.1.1).

## 4. Exact pytest result

```
109 passed in 9.58s
```

(0 failed, 0 warnings — full output saved to
`audit/final_pretraining_tests.txt`.) This is 29 more than the prior
pass's 80 (17 new tests across the four new files, plus... — exact
breakdown: `test_multistage_run_dir.py` +3, `test_action_mapping.py` +11,
`test_environment_invariants.py` +14, `test_checkpoint_saving.py` +1 = 29
new; 80 + 29 = 109).

## 5. Compile result

```
python -m compileall .
```
→ clean, no output, exit code 0.

## 6. Smoke-test result

Ran the actual CLI entry point end-to-end (not a synthetic test):

```
python -m training.train --run-name run_smoketest_final --stages 1 2
```

Result: **Stage 1 completed, Stage 2 completed, reusing the same
`runs/run_smoketest_final/` directory — no `FileExistsError`, no crash.**
This is the exact failure this report's Critical fix (ISSUE-011)
addresses; prior to the fix, Stage 2 would have crashed immediately.

Verified directly afterward:
- Both stages' TensorBoard logs present in separate `PPO_1`/`PPO_2`
  subfolders (SB3 auto-increments within the shared `tensorboard/` dir).
- Both stages' config/command snapshots present, distinctly named
  (`effective_ppo_stage1.yaml`, `effective_ppo_stage2.yaml`,
  `command_stage1.txt`, `command_stage2.txt`).
- Both stages' models preserved distinctly (`trained_model_stage1.zip`
  and `trained_model_stage2.zip` have different MD5 hashes), with
  `trained_model.zip` correctly matching Stage 2 (the latest).
- `reward_components.csv` (Stage-2-only artifact) present and populated
  (24,577 rows).
- `created_at.txt` contains a proper timezone-aware ISO 8601 timestamp
  (`...+00:00`), confirming the `datetime.utcnow()` fix from ISSUE-009
  remains in effect.
- No warnings printed during the run.

Smoke-test artifacts were deleted after verification (`runs/` is
gitignored and was never a delivered artifact — this was a smoke test
only, not a claim of full training, per the task's explicit instruction).

## 7. Action-space verification

- `environment/battery_env.py`: `action_space = Box(low=-1.0, high=1.0,
  shape=(1,), dtype=np.float32)` — confirmed unchanged.
- Mapping: `action_val = (raw_action + 1.0) / 2.0; requested_current =
  action_val * i_max` — confirmed unchanged.
- Clipping: `raw_action = float(np.clip(np.asarray(action).flatten()[0],
  -1.0, 1.0))` — confirmed already present, verified with tests exercising
  actions of 5.0, -5.0, and 1.2 (all clip correctly, continuous at the
  boundary).
- `tests/test_action_mapping.py`: 11/11 passed, including
  action=-1→0A, action=0→0.5×i_max, action=+1→i_max, and intermediate
  values (-0.5→0.25×i_max, +0.5→0.75×i_max), all derived from the
  environment's actual configured `i_max` (160.0 A currently, read from
  `configs/battery.yaml` at test time, not hard-coded).

## 8. Run-directory verification

- `tests/test_multistage_run_dir.py`: 3/3 passed — fresh single-stage
  creation, fresh-create overwrite guard still enforced, and a real
  4-stage run reusing one directory with every stage's artifacts
  surviving every subsequent stage.
- Live CLI smoke test (Section 6): confirmed on the actual
  `python -m training.train` entry point, not just the test harness.
- Overwrite protection (`FileExistsError` on a fresh `create_run_dir`
  call against a non-empty directory) explicitly re-verified as still
  intact via `test_case2_fresh_create_still_blocked_on_nonempty_dir`.

## 9. Safety verification

- `tests/test_environment_invariants.py`:
  `test_applied_current_never_exceeds_i_max_across_episode`,
  `test_applied_current_never_exceeds_out_of_range_actions`,
  `test_safety_intervention_flagged_when_requesting_max_in_derated_state`
  — all passed. Confirmed `0 <= applied_current_a <= i_max` at every step
  across multiple full episodes (including out-of-range raw actions), and
  confirmed the safety layer actually intervenes (doesn't just exist) when
  a full-current request is made in a high-temperature state.
- Pre-existing `tests/test_safety.py` (34 tests, unchanged) continues to
  pass, covering unit-level current-limiting, temperature derating,
  voltage tapering, SoC tapering, and v2 monotonicity.

## 10. Checkpoint verification

- `tests/test_checkpoint_saving.py`: 1/1 passed. Ran a real (short) Stage
  4 training with `CheckpointCallback` active, confirmed the final model
  and intermediate checkpoint(s) exist, are non-trivially sized, and —
  critically — actually **load** via `PPO.load()` and produce a finite,
  correctly-shaped action via `.predict()`. Not just file-existence
  checking.
- Checkpoint *selection* (as opposed to saving) was already fixed in the
  prior pass — see ISSUE-008 in `audit/ISSUES.md` — and is unchanged here.

## 11. Any remaining warnings

**None.** `pytest -q` output shows `109 passed in 9.58s` with no warnings
line at all (previously 14 `DeprecationWarning`s from `datetime.utcnow()`,
fixed in ISSUE-009 during the prior pass and reconfirmed unaffected here).

## 12. Whether full training is now safe to start

**Yes, for the specific things this report can verify — with one
standing caveat carried over from ISSUE-005, which this task explicitly
did not ask to resolve.**

What's verified: compileall clean, full test suite passing (109/109, 0
warnings), the multi-stage run-directory bug that would have crashed a
real `--stages 1 2 3 4` invocation is fixed and verified live, action
mapping/clipping verified against the actual configured `i_max`, safety
layer ordering verified end-to-end, reward and observation finiteness
verified under both normal and deliberately stressed conditions, and
checkpoint saving/loading verified with a real (short) training run.

**Standing caveat (unchanged from ISSUE-005):** `results_and_discussion.md`
still carries the "⚠️ UNVERIFIED RESULTS" banner added in the prior pass,
because no `runs/run_010/` artifacts exist in this project to confirm its
reported numbers. This does not block starting a *new* Stage-4 training
run — it means any *previous* claimed results should not be relied upon
until that gap is resolved.

### Explicit change summary

```
PPO architecture changed?        NO
Action space changed?            NO
Observation space changed?       NO
Reward weights changed?          NO
Battery equations changed?       NO
Battery parameters changed?      NO
Safety thresholds changed?       NO
```

Only `agents/train_ppo.py` and `training/train.py` received functional
changes, both scoped to run-directory orchestration (not training logic,
not the environment, not the model).
