# Existing Project Audit — Issue Register

Scope: static code review + test-suite execution of `rl-bms-full.zip` as
uploaded. No training was run (no `runs/` directory or checkpoints exist in
the uploaded project — see ISSUE-005). All 63 pre-existing tests passed at
baseline (`audit/baseline_tests.txt`); the codebase was already in
substantially good shape, with several prior bug fixes documented inline
(comments reference "v2"/"v3" fixes to the safety layer, energy metric,
reward, and config-snapshot ordering). This register covers what was still
outstanding.

---

## ISSUE-001

**File:** `utils/logger.py` (`create_run_dir`)

**Category:** Bug

**Severity:** Medium

**Current behavior (before fix):** `next_idx = len(existing) + 1`, where
`existing` is the list of `run_*` directories. This uses the *count* of run
directories, not the *highest number present*.

**Expected behavior:** Next run number should be `max(existing numeric
suffixes) + 1`.

**Evidence:** Manually traced: with `runs/` containing `run_001, run_002,
run_005`, `len(existing)=3` → `next_idx=4` → `"run_004"`, which is wrong on
two counts — it isn't the next free number (`run_004` may already be
reserved/planned) and it doesn't reflect `run_005` being the highest run so
far. Confirmed no test existed for this function prior to the fix
(`grep -rn create_run_dir tests/` returned nothing).

**Proposed fix:** Parse numeric suffixes with a regex (`^run_(\d+)$`),
ignore non-matching directory names instead of counting them, take
`max(...)+1`. Also refuse to create a run directory that already exists and
is non-empty (previously any collision would silently reuse/merge into an
existing directory).

**Does this invalidate existing checkpoints?** No — pure directory-naming
logic, does not touch physics, reward, action space, or model weights.

**Requires retraining?** No.

**Verification test:** `tests/test_utils_fixes.py::test_gap_in_run_numbers_uses_max_plus_one`,
`::test_scattered_run_numbers_uses_max_plus_one`,
`::test_non_run_directories_are_ignored`,
`::test_malformed_run_names_do_not_crash_or_count`,
`::test_existing_nonempty_run_dir_refuses_overwrite`.

**Status:** Fixed.

---

## ISSUE-002

**File:** `utils/metrics.py` (`aggregate_runs`)

**Category:** Metric

**Severity:** Medium-High

**Current behavior (before fix):** Used `np.mean()` / `np.std()` directly
on each metric key across runs. `summarize_episode` intentionally emits
`NaN` for `time_to_target_s` when `target_soc` was not reached (by design —
see its docstring). A single `NaN` in one run's value for a key propagates
`NaN` through `np.mean`/`np.std` for *every* run's aggregate of that key,
silently hiding all other runs' real results behind a single failed run.

**Expected behavior:** NaNs should be excluded from the mean/std
calculation (not treated as 0 or dropped from the whole row), and the
number of valid vs. failed (NaN) runs should be reported explicitly so a
high failure rate isn't hidden behind a clean-looking mean.

**Evidence:** No test existed for `aggregate_runs` prior to the fix
(confirmed via grep). Reproduced the contamination directly: `aggregate_runs([{"x": 1800.0}, {"x": 1900.0}, {"x": float("nan")}])`
under the old code returns `mean=nan` for `x`, discarding the two valid
1800/1900 values.

**Proposed fix:** Use `np.nanmean`/`np.nanstd`, but only when at least one
value is finite (all-NaN columns now explicitly return `nan` without
triggering a RuntimeWarning, rather than relying on nanmean's own
behavior). Added `valid_runs`/`failed_runs` counts to every aggregated
metric's result dict.

**Does this invalidate existing checkpoints?** No — evaluation
post-processing only, does not touch training.

**Requires retraining?** No.

**Verification test:** `tests/test_utils_fixes.py::test_single_nan_does_not_contaminate_other_runs`,
`::test_all_nan_reports_nan_explicitly_not_a_crash`,
`::test_aggregate_no_nans_matches_plain_mean_std`.

**Status:** Fixed.

**Note:** `aggregate_runs` has no callers anywhere in the current codebase
(only defined; not imported by `training/evaluate.py`,
`training/generate_report.py`, or elsewhere). It appears to be intended for
future multi-run comparison but is not yet wired in. The fix is still
correct and necessary once it is used, and does not change any current
runtime behavior.

---

## ISSUE-003

**File:** `training/train.py`, `agents/train_ppo.py`

**Category:** Documentation / Training design

**Severity:** Low (behavior is correct as coded; only the documentation
risk is real)

**Current behavior:** Each stage (1–4) calls `build_agent()` and
constructs a **new** `PPO` model from scratch. Stage transitions do **not**
load a previous stage's weights. This is confirmed directly in
`agents/train_ppo.py::run_stage` — `model = build_agent(env, ppo_cfg, ...)`
is called unconditionally for every stage, with no `PPO.load(...)` path.

**Investigation:** Per `agents/train_ppo.py`'s own module docstring, this
is **intentional**: Stage 1 = sanity run, Stage 2 = reward-component
logging run, Stage 3 = manual hyperparameter experiments, Stage 4 = full
training run. These are described as four **independent diagnostic
experiments**, not a sequential curriculum — consistent with
`README.md`'s phase table, which lists each as a separately-validated
step. There is no documentation anywhere claiming stages continue from one
another's weights.

**Decision:** Option A (independent experiments) — matches both the code
and existing documentation. **No code or doc change required**; see
`audit/STAGE_PIPELINE.md` for the full trace.

**Does this invalidate existing checkpoints?** No.

**Requires retraining?** No.

**Status:** Verified as intentional, no fix needed.

---

## ISSUE-004

**File:** `agents/train_ppo.py`, `README.md`

**Category:** Documentation

**Severity:** Low

**Current behavior:** Stage 3 supports CLI overrides for `--lr`,
`--batch-size`, `--ent-coef`, applied manually per invocation
(`run_stage(..., hpo_overrides=...)`). There is no automated search loop,
no Optuna/Ray Tune integration, and no sweep orchestration — the user must
invoke the script once per hyperparameter combination.

`README.md` line 107 uses the phrase `# HPO sweep example` next to a single
example invocation; `results_and_discussion.md` similarly uses "Stage-3
hyperparameter-sweep runs" (line ~182). Neither claims an *automated*
search, but "sweep" is ambiguous enough to be read that way.

**Evidence:** `grep -n "Optuna\|ray.tune\|automl"` across the codebase:
no matches. `agents/train_ppo.py::main()` only defines three scalar
`--lr/--batch-size/--ent-coef` overrides, applied to one `run_stage` call
per process invocation.

**Fix applied (by maintainer request):** Reworded "HPO sweep" → "manual
hyperparameter configuration" in both `README.md` (line 107, the
Stage-3 CLI example comment) and `results_and_discussion.md` (line ~182,
the config-snapshot-ordering bullet). This is a wording-only change — no
numbers, claims of automation, or results were altered, and neither file
now implies an automated search exists.

**Does this invalidate existing checkpoints?** No.

**Requires retraining?** No.

**Status:** Fixed.

---

## ISSUE-005 (Critical — documentation/evidence conflict)

**File:** `README.md` vs. `results_and_discussion.md`

**Category:** Documentation

**Severity:** Critical

**Finding:** `README.md` states explicitly: *"Stage 4 full PPO training
(1M timesteps) has not been run — that's a compute/time commitment for you
to kick off (see below), not something to run silently in a sandbox."*

`results_and_discussion.md` is written as a **completed final report**: it
reports a specific trained model (`run_010`, checkpoint `75000_steps`),
1,000,000-timestep training with checkpointing every 25,000 steps, a
7-issue debugging journey, and comparative results against CC/CCCV/
Adaptive/Max-Current baselines with specific numeric findings.

**Evidence:** The uploaded project contains **no `runs/` directory, no
checkpoint files, no TensorBoard logs, and no CSV result logs anywhere in
the zip** (confirmed via full directory listing — only source code,
configs, tests, and two documentation files exist). There is no artifact
in this project to verify any claim in `results_and_discussion.md` against.

**Per audit instructions, "use actual artifacts, not documentation, to
decide which statement is true"** — but no artifacts exist to check
against either statement. I cannot confirm `results_and_discussion.md`'s
reported results actually came from a real training run in *this*
codebase, nor can I confirm they didn't (e.g. they may be from a run
executed elsewhere, whose output artifacts were never included in this
zip/export).

**I am not able to resolve this from the information available and am not
fabricating a resolution.** Flagging for the maintainer:

1. If `run_010` was trained elsewhere, the `runs/run_010/` artifacts
   (checkpoints, `command.txt`, `config/effective_ppo.yaml`,
   `reward_components.csv`, TensorBoard logs) should be included in future
   exports so results are independently verifiable.
2. Until then, treat every specific number in
   `results_and_discussion.md` (Section 4 onward) as **unverified against
   this codebase** rather than as confirmed fact.
3. `README.md`'s "has not been run" line and `results_and_discussion.md`'s
   "final report" framing are directly contradictory and should not both
   ship as-is.

**Does this invalidate existing checkpoints?** N/A — no checkpoints exist
in the provided project.

**Requires retraining?** Not determinable without first locating (or
re-running to produce) the missing `runs/run_010/` artifacts.

**Status:** Flagged, unresolved — requires maintainer input, not fixed.

---

## ISSUE-006

**File:** `select_best_checkpoint.py` (repo root) vs.
`training/select_best_checkpoint.py`

**Category:** Documentation / repo hygiene

**Severity:** Low

**Current behavior:** Two files with the same name and overlapping
purpose exist. `training/select_best_checkpoint.py` is the current,
more capable version (reports `mean_reward`, `target_reached_rate`, and a
reward/final-SoC correlation check to detect reward misalignment).
`select_best_checkpoint.py` at the repo root is an older, simpler version
(final-SoC-only scoring) — a stale duplicate, most likely left over from
`create_zip.py`/`combine_all_code.py` bundling utilities that copy files
to the root for packaging.

**Evidence:** `diff training/select_best_checkpoint.py select_best_checkpoint.py`
shows the root copy is missing `mean_reward`, `episode_length`,
`target_reached_rate`, and the reward/objective-alignment correlation
check entirely present in the `training/` version.

**Fix applied (by maintainer request):** Confirmed via grep that no other
source file references the root-level path (`combine_all_code.py` already
bundles only `training/select_best_checkpoint.py`, the canonical version —
see line 41). This is a stray script, not a "result" or generated
artifact, so it is not covered by the "do not delete old results" rule.
Removed `select_best_checkpoint.py` (repo root) via `git rm`;
`training/select_best_checkpoint.py` remains as the single source of
truth. Full test suite re-run after removal: 74/74 still passing.

**Does this invalidate existing checkpoints?** No.

**Requires retraining?** No.

**Status:** Fixed (removed).

---

## ISSUE-007

**File:** Project root (`all_rl_bms_code.py`, `rl_bms_inspection.txt`,
`run010_code_inspection_2.txt`)

**Category:** Documentation / repo hygiene

**Severity:** Low

**Current behavior:** `all_rl_bms_code.py` is a generated combined dump of
the entire codebase (per `combine_all_code.py`), now stale relative to the
fixes in this pass (still contains the pre-fix `next_idx = len(existing) +
1` and the old `aggregate_runs`). `rl_bms_inspection.txt` and
`run010_code_inspection_2.txt` are large inspection dumps whose provenance
(what tool/session produced them, and for which code state) is not stated
in the files themselves.

**Proposed fix:** Left unmodified — these are generated/derived
artifacts, not source of truth, and regenerating them is a mechanical
step the maintainer can run via `python combine_all_code.py` after
merging these fixes. Not treated as an "issue" requiring a code change,
only flagged so the maintainer knows they're now stale.

**Status:** Flagged, informational only.

---

## ISSUE-006 — CORRECTION

**A correction to the record above, not a new issue.** In the original
ISSUE-006 write-up I misread my own `diff training/select_best_checkpoint.py
select_best_checkpoint.py` output and stated the root-level file was the
"older, simpler version" and `training/`'s was "more capable." **This was
backwards.** The root-level file (deleted) was actually the more capable
one — it computed `mean_reward`, `mean_episode_len`, `target_reached_rate`,
and a reward/final-SoC correlation diagnostic; `training/`'s version only
computed `mean_final_soc`. I deleted the better file and kept the weaker
one. This was an analysis error on my part, not a deliberate choice, and
it was caught by a follow-up review that checked the actual file contents
in the delivered zip rather than trusting my prior summary. See ISSUE-008
for the fix (which supersedes both original versions).

---

## ISSUE-008

**File:** `training/select_best_checkpoint.py`

**Category:** Logic / Metric

**Severity:** High

**Current behavior (before this fix):** Selected the best checkpoint by
`max(results, key=lambda r: r[1])` where the score was `mean_final_soc`
alone (confirmed by reading the file directly, not by trusting the prior
audit summary — see ISSUE-006 correction above). A checkpoint reaching
96% SoC while running hot, intervening on safety constantly, and taking
far longer than a 95%-SoC checkpoint would still win purely on the 96% >
95% comparison.

**Expected behavior:** Selection should account for the full evaluation
criteria (target reached, safety interventions, charging time, thermal
stress, energy efficiency), matching Phase 15's specified lexicographic
policy.

**Evidence:** Read `training/select_best_checkpoint.py` directly (both
the pre-fix root and `training/` copies, and the post-ISSUE-006 state).
Confirmed no test previously existed for the selection logic.

**Fix:** Rewrote `_quick_score` → `_score_checkpoint`, which now runs each
checkpoint through the evaluation scenarios using the exact same per-step
logging (`training.evaluate._run_ppo_episode`, imported and reused, not
duplicated) and `utils.metrics.summarize_episode` that
`training/evaluate.py` uses for final reported results — so checkpoint
selection and final evaluation can never define a metric differently.
Added an explicit lexicographic selector (`_select_best`) applying, in
order: `target_reached_rate` (higher wins) → `safety_interventions`
(lower wins) → `charging_time_s` (lower wins) → `peak_temperature_c`
(lower wins) → `energy_efficiency` (higher wins), each with a small
tolerance so near-identical checkpoints aren't ordered by float noise.
Also now writes a full `checkpoint_selection.csv` comparison table to the
run directory (all metrics, all checkpoints), not just a print statement.

**Does this invalidate existing checkpoints?** No — selection-time
analysis only, does not touch training or the checkpoints themselves.

**Requires retraining?** No.

**Verification test:** `tests/test_checkpoint_selection.py` — 6 tests
covering: higher final_soc losing to a checkpoint better on every other
criterion, target_reached_rate dominating, safety/time/thermal tie-breaks
in order, single-candidate and fully-tied edge cases. Also smoke-tested
end-to-end against a real trained checkpoint (`agents.train_ppo --stage
1` → `training.select_best_checkpoint`), producing a real
`checkpoint_selection.csv` with correct values (confirmed manually).

**Status:** Fixed.

---

## ISSUE-009

**File:** `utils/logger.py`

**Category:** Bug (deprecation warning)

**Severity:** Low

**Current behavior (before fix):** `datetime.utcnow()`, deprecated in
Python 3.12+, producing a `DeprecationWarning` on every run-directory
creation (visible in `audit/post_fix_tests.txt`'s warnings summary).

**Fix:** Changed to `datetime.now(timezone.utc)`, timezone-aware.
`created_at.txt` now contains a proper `+00:00`-suffixed ISO 8601
timestamp instead of a bare timestamp with a manually appended `"Z"`.

**Does this invalidate existing checkpoints?** No.

**Requires retraining?** No.

**Verification:** Full test suite (`audit/final_tests.txt`) now runs with
**zero warnings** (previously 14, all from this one deprecation).

**Status:** Fixed.

---

## ISSUE-010

**File:** `README.md`

**Category:** Documentation

**Severity:** Medium

**Correction to the record:** In the previous review pass I stated the
README thermal equation was "already correct" and needed no change. **That
was wrong.** I did grep the file at the time, but evidently checked a
stale in-memory/earlier state rather than the actual current line in the
delivered zip — line 65 of `README.md` read:

```
Q_gen = I²R0 + I·V_RC
```

which does not match the implementation
(`environment/ecm_model.py::_derivatives`, `q_gen = I²*R0 + Vrc²/R1`) or
`docs/model_specification.md` (already correct since the earlier audit
pass). This was only caught because a follow-up review re-checked the
actual delivered file instead of trusting my prior claim — the same
category of error as the ISSUE-006 correction (asserting a fact about
file contents without re-reading them at the point of the claim).

**Fix:** Changed `README.md` line 65 to
`Q_gen = I²*R0 + V_RC²/R1`, matching both the code and
`docs/model_specification.md` exactly. Verified via grep that no other
occurrence of the stale `I·V_RC`/`I*V_RC` form remains anywhere in
`README.md`, `results_and_discussion.md`, or `docs/*.md`.

**Does this invalidate existing checkpoints?** No — documentation only.

**Requires retraining?** No.

**Verification:** `grep -rn "Q_gen" README.md docs/model_specification.md`
now shows identical formulas in both files. Full test suite re-run: 80/80
passing, 0 warnings.

**Status:** Fixed.

---

## ISSUE-011

**File:** `training/train.py`, `agents/train_ppo.py`

**Category:** Bug (Critical — blocked all multi-stage training)

**Severity:** Critical

**Current behavior (before fix):** `training/train.py::main()` looped over
stages 1–4, calling `run_stage(stage, run_name=args.run_name)` every time.
`run_stage` unconditionally called `create_run_dir(..., run_name=run_name)`
on every invocation. Since Stage 1 writes files into the run directory
before Stage 2 starts, Stage 2's `create_run_dir` call hit the
(correct, must-stay) non-empty-directory guard added in ISSUE-001 and
raised `FileExistsError` — **the entire multi-stage pipeline was broken**,
including the actual `training/train.py --stages 1 2 3 4` path intended
for Stage 4's full run.

**Evidence:** Reproduced directly before fixing: called `create_run_dir`
twice with the same `run_name` after writing a file into the first
result — confirmed `FileExistsError` exactly as described.

**Fix:** Separated CREATE from REUSE. `training/train.py::main()` now
calls `create_run_dir` **exactly once**, before the stage loop, and passes
the resulting `run_dir` into every `run_stage(stage, run_dir=run_dir)`
call. `run_stage` now accepts an optional `run_dir` parameter: if given,
it skips `create_run_dir` entirely (directory and subdirs already exist);
if not given (the single-stage `python -m agents.train_ppo --stage N` CLI
path), it calls `create_run_dir` exactly as before, **preserving the
overwrite-protection test from ISSUE-001 unchanged**.

Additionally fixed a related artifact-loss issue this exposed: each stage
builds a fresh PPO model (see `audit/STAGE_PIPELINE.md`), so before this
fix, `model.save(run_dir/trained_model.zip)` on Stage 2 would silently
overwrite Stage 1's saved model. Now each stage's model is saved to
`trained_model_stage{N}.zip` (preserved), with `trained_model.zip` kept as
a copy of the most-recently-completed stage's model for backward
compatibility with `training/select_best_checkpoint.py` and
`training/evaluate.py`, which both reference that fixed path. Similarly,
`effective_ppo.yaml`/`command.txt` are now namespaced per stage
(`effective_ppo_stage{N}.yaml`, `command_stage{N}.txt`) so Stage 3's HPO
override values remain inspectable even after Stage 4 has also run.

**Does this invalidate existing checkpoints?** No — no checkpoints exist
in this project (ISSUE-005). This fix only affects directory/file naming
for *future* runs.

**Requires retraining?** No, but this fix is a **prerequisite** for any
multi-stage training run to work at all — training could not previously
proceed past Stage 1 via `training/train.py`.

**Verification test:** `tests/test_multistage_run_dir.py` — 3 tests
covering: fresh single-stage creation, the fresh-create overwrite guard
still applying at the `run_stage` level, and a real 4-stage run (with a
monkeypatched tiny PPO config for speed) verifying all four stages reuse
one directory without crashing and that every stage's artifacts survive
every subsequent stage. Also verified live via the actual CLI entry point
(`python -m training.train --run-name run_smoketest_final --stages 1 2`)
— see `audit/final_pre_training_report.md`.

**Status:** Fixed.

---

## ISSUE-012

**File:** `environment/battery_env.py`, `README.md`

**Category:** Documentation

**Severity:** Low

**Finding:** `README.md` had no explicit statement of the action space at
all (not contradictory, just absent) — `docs/model_specification.md` and
`docs/model_contract.md` already correctly documented `[-1, 1]`
(confirmed by direct grep across all docs; no stale `[0,1]` claims found
anywhere in this project).

**Fix:** Added an explicit action-space line to `README.md`'s architecture
section, matching `docs/model_specification.md` Section 6 exactly:
`Box(low=-1, high=1, shape=(1,))`, with the mapping formula and a pointer
to why the range is symmetric rather than `[0,1]`.

**Status:** Fixed (documentation completeness, not a contradiction fix).

---

## ISSUE-013

**File:** `environment/battery_env.py`

**Category:** Verification (no code change)

**Severity:** N/A

**Finding:** Action clipping already existed:
`raw_action = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))`
— an action outside the declared `[-1, 1]` Box is clamped before mapping,
not passed through raw.

**Verification test:** `tests/test_action_mapping.py` —
`test_action_above_one_is_clipped_to_i_max`,
`test_action_below_minus_one_is_clipped_to_zero`,
`test_action_clip_is_continuous_at_boundary` (actions of 5.0/-5.0/1.2 all
verified to behave identically to their clipped equivalents).

**Status:** Verified correct, no code change needed.

---

## ISSUE-014

**File:** `environment/battery_env.py`

**Category:** Verification (no code change)

**Severity:** N/A

**Finding:** Traced the action pipeline directly: `requested_current` →
`safety_layer(...)` → `clamped_current` → `applied_current` (when
`enforce_safety=True`, the default and what `env_factory.make_env` always
uses) → `self.ecm.step(self._state, applied_current, ...)`. The battery
model receives `applied_current`, never the raw request. A separate
`enforce_safety=False` ablation/monitoring path exists but is
opt-in-only, documented in the constructor, and not used by training or
evaluation.

**Verification test:** `tests/test_environment_invariants.py` — full
episodes (including out-of-range raw actions) asserting
`0 <= applied_current_a <= i_max` at every single step, plus a
forced-high-temperature scenario confirming the safety layer actually
intervenes (`applied_current_a < requested`) when requesting full current
in a derated state — proving the request genuinely passes through the
safety layer rather than bypassing it.

**Status:** Verified correct, no code change needed.

---

## ISSUE-015

**File:** `environment/battery_env.py`

**Category:** Verification (no code change)

**Severity:** N/A

**Finding:** No mechanism in the reward computation can silently produce
NaN/Inf under normal or stressed conditions — all divisions are by
config-fixed nonzero denominators (`i_max`, `R1`, etc.), and clamps are
applied before any log/sqrt-like operation (none are used).

**Verification test:** `tests/test_environment_invariants.py` —
`test_reward_finite_under_normal_conditions` (500-step random-action
episodes) and `test_reward_finite_in_stressed_states` (parametrized over
normal/high-temperature/near-target-SoC/extreme-temperature initial
conditions, including terminal steps) — all rewards and all individual
`reward_components` dict entries asserted finite.

**Status:** Verified correct, no code change needed.

---

## ISSUE-016

**File:** `environment/battery_env.py`

**Category:** Verification (no code change)

**Severity:** N/A

**Finding:** All 6 observation features are explicitly `np.clip(...,
0.0, 1.0)`'d after normalization (confirmed by reading
`_get_observation` directly), so no unbounded division-by-near-zero
propagates into the observation even at extreme states.

**Verification test:** `tests/test_environment_invariants.py` —
`test_observation_finite_under_normal_conditions` and
`test_observation_finite_at_extreme_initial_conditions` (parametrized
over near-empty SoC/cold ambient, near-full SoC/hot ambient, sub-zero
ambient, and above-hard-cutoff ambient temperature).

**Status:** Verified correct, no code change needed.

---

## ISSUE-017

**File:** `agents/train_ppo.py` (`CheckpointCallback` usage)

**Category:** Verification (no code change)

**Severity:** N/A

**Finding:** `CheckpointCallback` from Stable-Baselines3 is already wired
correctly for Stage 4 (`save_freq=ppo_cfg["checkpoint_freq"]`,
`save_path=os.path.join(run_dir, "checkpoints")`).

**Verification test:** `tests/test_checkpoint_saving.py` — runs a real
(monkeypatched-short) Stage-4 training with checkpointing active, then
asserts: the final and intermediate checkpoint files exist and are
non-trivially sized (not empty/corrupted), **and** actually loads each
one back via `PPO.load()` and runs `.predict()` on it to confirm it's a
genuinely usable policy, not just a file that happens to exist.

**Status:** Verified correct, no code change needed.

---

## Summary Table

| ID | Category | Severity | Status |
|----|----------|----------|--------|
| ISSUE-001 | Bug | Medium | Fixed |
| ISSUE-002 | Metric | Medium-High | Fixed |
| ISSUE-003 | Training design | Low | Verified intentional, no fix needed |
| ISSUE-004 | Documentation | Low | Fixed |
| ISSUE-005 | Documentation/evidence conflict | **Critical** | Flagged, unresolved |
| ISSUE-006 | Repo hygiene | Low | **Corrected** — original fix deleted the wrong (better) file; see ISSUE-006 correction and ISSUE-008 |
| ISSUE-007 | Repo hygiene | Low | Fixed (regenerated via `python combine_all_code.py`) |
| ISSUE-008 | Logic/Metric | High | Fixed — lexicographic checkpoint selection |
| ISSUE-009 | Bug (deprecation) | Low | Fixed — timezone-aware `datetime.now(timezone.utc)` |
| ISSUE-010 | Documentation | Medium | Fixed — README thermal equation corrected to match code/spec (correcting a prior false "already correct" claim) |
| ISSUE-011 | Bug | **Critical** | Fixed — multi-stage run-directory reuse (previously crashed every multi-stage run past Stage 1) |
| ISSUE-012 | Documentation | Low | Fixed — added explicit action-space statement to README |
| ISSUE-013 | Verification | N/A | Verified — action clipping already correct |
| ISSUE-014 | Verification | N/A | Verified — safety layer confirmed in the applied-current path |
| ISSUE-015 | Verification | N/A | Verified — reward finiteness under normal + stressed states |
| ISSUE-016 | Verification | N/A | Verified — observation finiteness under normal + extreme states |
| ISSUE-017 | Verification | N/A | Verified — checkpoint save/load/inference all work correctly |

No changes were made to: action space, observation space, PPO
hyperparameters, network architecture, ECM equations, reward weights, or
safety-layer thresholds. Nothing in `configs/*.yaml` was modified.

## ISSUE-018 — Stable V3: state-aware thermal reward added (config-gated, default off)

**Category:** Feature addition (reward). **Severity:** N/A (opt-in, `thermal_enabled: false` by default).

Following extensive diagnostic sweeps of a flat, always-on thermal penalty
(`-w_thermal * normalized_q_gen`, tested at w in {0.10, 0.20, 0.30, 0.35,
0.40, 0.45, 0.50} across 3 seeds each — see `audit/thermal_weight_040_045_report.md`),
the flat form was found seed-fragile with no reliable operating point:
mostly inert, occasionally a small effect in one seed, and prone to
zero-current/non-convergent collapse at higher weights.

Replaced with a state-aware form in `environment/battery_env.py
_compute_reward`, gated by new `reward.yaml` keys (`thermal_enabled`,
`thermal_weight`, `thermal_reference_temp_c`, `thermal_scale_c`,
`thermal_q_reference_w`) — see `audit/stable_v3_report.md` for the full
equation, reference-value derivation (all reused from existing
`battery.yaml`/`safety.yaml`/`reward.yaml` constants, no new physical
limits invented), test coverage, and diagnostic results.

**Finding:** with `thermal_reference_temp_c=40.0` (reusing the existing
`temperature_penalty_start_c`), the new term measured `thermal_reward=0.0`
in every training chunk across all 3 diagnostic seeds — this environment's
observed temperature never exceeds ~32.3°C even under continuous
full-current charging (see `audit/stable_v3_report.md` §14), so the term
never activates at this reference. Training is stable and reproduces the
exact baseline trajectory in all 3 seeds. **Status:** implemented, tested
(10 new tests, all passing; 119/119 total), default-off, functionally a
no-op at the current reference temperature — flagged as needing either a
lower reference or acceptance that this environment's normal operating
envelope doesn't warrant a proactive thermal term. Not yet a validated
"better" configuration; see `audit/stable_v3_report.md` §21 for
recommendation.
