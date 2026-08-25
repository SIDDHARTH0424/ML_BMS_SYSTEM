# Before / After Audit Summary

| Item | Before | After |
|---|---|---|
| Tests | 63 passed (`audit/baseline_tests.txt`) | **80 passed, 0 warnings** — 17 new tests added, 0 removed, 0 modified (`audit/final_tests.txt`; Python 3.12.3, `requirements.txt` versions exactly as declared) |
| `compileall` | Clean | Clean |
| Run numbering | `len(existing)+1` — wrong under gaps/non-numeric dirs (ISSUE-001) | `max(numeric suffixes)+1`, ignores non-`run_N` dirs, refuses to overwrite a non-empty existing run dir |
| NaN aggregation | `np.mean`/`np.std` — one NaN poisons the whole column (ISSUE-002) | `nanmean`/`nanstd` + explicit `valid_runs`/`failed_runs` counts; all-NaN columns explicitly return NaN, not a warning |
| Stage behavior | 4 independent PPO runs, no weight continuation | Unchanged — verified intentional (ISSUE-003), no code change |
| Stage-3 HPO claim | "sweep" wording is ambiguous re: automated vs. manual | Fixed (ISSUE-004) — reworded to "manual hyperparameter configuration" in `README.md` and `results_and_discussion.md` |
| Safety layer tests | Already comprehensive (34 tests in `tests/test_safety.py`, incl. v2 monotonicity) | Unchanged — no gaps found |
| ECM/physics tests | Already comprehensive (19 tests in `tests/test_ecm.py`, incl. zero-current, RC relaxation, heat generation, thermal balance, SoC bounds) | Unchanged — no gaps found |
| Energy/voltage metrics | Already correctly named and documented in-code (`energy_efficiency`, `voltage_stability` docstrings already flag the coarse-proxy caveat) | Unchanged — already accurate, no fix needed |
| Checkpoint selection | Selected by `mean_final_soc` alone (confirmed on direct file read — an earlier audit pass mischaracterized this, see ISSUE-006 correction) | **Fixed (ISSUE-008)** — full lexicographic selection: target_reached_rate → safety_interventions → charging_time_s → peak_temperature_c → energy_efficiency, using the same `summarize_episode`/`_run_ppo_episode` logic as `training/evaluate.py`. Also writes a full `checkpoint_selection.csv` comparison table |
| `datetime.utcnow()` warning | 14 `DeprecationWarning`s on every test run | **Fixed (ISSUE-009)** — `datetime.now(timezone.utc)`; 0 warnings now |
| `all_rl_bms_code.py` | Stale (still had pre-fix `logger.py`/`metrics.py` code) | Regenerated via `python combine_all_code.py`; verified current fixes present |
| SOH claims | `README.md` already correctly states "monitoring only" | Unchanged — already accurate |
| Documentation vs. artifacts | `README.md` says Stage 4 hasn't been run; `results_and_discussion.md` reports a completed run with no supporting artifacts anywhere in the project (ISSUE-005) | **Unresolved** — flagged as Critical, requires maintainer input (missing `runs/run_010/` artifacts) |
| Reward | Already well-documented with derivations (`configs/reward.yaml` comments) | Unchanged. New `audit/reward_audit.md` transcribes the full breakdown |
| PPO architecture / action space / observation space | — | **Unchanged** |

## Explicit compatibility statements

- **Does old checkpoint remain compatible?** N/A — no checkpoint files
  exist in the uploaded project (see ISSUE-005). Nothing that would
  invalidate a checkpoint (action space, observation space, ECM, reward
  weights) was touched, so *if* a checkpoint from this codebase existed
  elsewhere, it would remain compatible.
- **Does retraining need to happen?** No. Both fixes are non-invasive
  (directory naming, evaluation-time aggregation) and do not touch
  anything in the training loop, environment, or reward.
- **Were battery physics changed?** No.
- **Were reward weights changed?** No.
- **Was action space changed?** No.
- **Was observation space changed?** No.
- **Was the safety layer changed?** No.

## Smoke test (Phase 22)

Ran a genuine Stage 1 sanity training (`python -m agents.train_ppo --stage
1 --run-name run_smoketest`, 8192 timesteps — the actual configured
rollout length, since `n_steps=8192` > `stage1_sanity_timesteps=5000`):

- Training started and completed without crashing.
- No NaN/Inf detected in policy parameters (the built-in Stage-1 check).
- `runs/run_smoketest/` was created correctly: config snapshot,
  `effective_ppo.yaml`, `command.txt`, `git_commit_hash.txt`,
  `created_at.txt`, TensorBoard log directory, and `trained_model.zip` all
  present.
- Verified the run-numbering fix live: after the named `run_smoketest`
  run, an auto-numbered `create_run_dir("runs")` call correctly produced
  `run_001` (ignoring the non-numeric `run_smoketest` name) rather than
  erroring or miscounting.
- The throwaway `runs/run_smoketest/` and `runs/run_001/` directories were
  deleted after verification (`runs/` is gitignored and was never part of
  the delivered project — they were smoke-test-only, per the plan's
  "small test first" instruction, not a claim of full training).

## What was explicitly NOT done (per the constraints given)

- Action space, observation space, PPO algorithm/hyperparameters, ECM
  topology, reward weights, and safety thresholds were left untouched.
- No new battery parameters, literature values, or experimental results
  were invented.
- No files were deleted (stale duplicates were flagged, not removed).
- No full 1,000,000-timestep Stage-4 training was run — only the Stage-1
  smoke test, per the plan's explicit instruction not to run expensive
  training in this pass.
- `results_and_discussion.md`'s specific numeric claims were not rewritten
  to match a run that doesn't exist in this project, and were not deleted
  either — they're flagged as unverified (ISSUE-005) for the maintainer to
  resolve with the actual missing artifacts.

## Stable V3 — state-aware thermal reward (this pass)

- Added `BatteryECM.heat_generation_w()` (exposes existing `q_gen` formula,
  no duplication) and a config-gated `thermal_reward` term in
  `_compute_reward` (`reward.yaml: thermal_enabled`, default `false`).
- 10 new tests added (`tests/test_thermal_reward_stable_v3.py`), all
  passing; full suite 119/119 passing.
- 3-seed/50k-step diagnostic (real production env path, original stable
  PPO config, no gSDE/squash) reproduced the exact baseline trajectory in
  every seed — the term was measured `0.0` in every training chunk,
  because this environment's observed operating temperature (~25–32°C)
  never crosses the reused 40°C reference. See `audit/stable_v3_report.md`
  for the full writeup and recommendation.
- Not done in this pass: tuning `thermal_reference_temp_c` below 40°C
  (would no longer be a "reused existing constant" per the task's
  constraint), and no 1M-step training was run.
