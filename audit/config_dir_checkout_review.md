# Checkout Review: rl-bms-Driving-final-fixed.zip (config_dir plumbing)

**Timestamp**: 2026-08-16T15:45:00Z

## What this delivery added (verified, not just diffed)

`--config-dir` plumbing through `agents/train_ppo.py::run_stage`,
`training/train.py`, `training/train_drive_ems.py` (new `train_long()`),
and `training/evaluate_drive_ems.py`, plus two assembled frozen config
sets: `configs/final_charging/` (Candidate A1) and
`configs/final_driving/` (Candidate B3). This is the missing piece
needed to actually launch long training against the frozen, gate-passed
configurations rather than the mutable dev configs.

## Bugs found by running the suite + smoke-testing both entry points

1. **3 test failures** (`test_checkpoint_saving.py`,
   `test_multistage_run_dir.py`): monkeypatched `load_config` fakes had
   signature `_fake_load_config(name)`, didn't accept the new
   `config_dir` kwarg the production code now always passes. Fixed both
   fakes to accept `config_dir=None` and forward it.

2. **Reintroduced `datetime.utcnow()`** in the new
   `train_drive_ems.py::train_long()` — same deprecation the project
   already fixed once (ISSUES.md ISSUE-009). Fixed to
   `datetime.now(timezone.utc)`.

3. **`configs/ppo_final.yaml` / `configs/final_charging/ppo.yaml` were
   broken and silently wrong**: missing `seed` (crashed
   `set_global_seed(ppo_cfg["seed"])` immediately) and
   `stage*_timesteps`/`tensorboard_log`/`checkpoint_freq` (required by
   `run_stage()` for any stage). Also had `ent_coef: 0.0`, but the
   actual Candidate A1 short-gate runs
   (`experiments/run_readiness_diagnostics.py::train_and_eval_charging_candidate`)
   loaded `ent_coef` from the dev `configs/ppo.yaml`, which is `0.01`.
   A long run launched from the un-fixed frozen config would not have
   matched the configuration the short gate actually validated. Fixed:
   restored `seed: 7`, correct `ent_coef: 0.01`, and the stage/logging
   keys.

4. **`configs/simulation_final.yaml` / `configs/final_charging/simulation.yaml`
   missing `target_soc: 0.95`** — required by
   `BatteryChargingEnv.__init__`, crashed `make_env()` immediately.
   Fixed.

## Verification performed (not just read the diff)

- Full test suite: 213/213 passing after fixes (`audit/long_training_freeze_tests_v2.txt`).
- `python -m compileall . -q`: 0 errors (`audit/final_compileall_v2.txt`).
- `python -m agents.train_ppo --stage 1 --config-dir configs/final_charging
  --run-name <tmp>`: ran end-to-end, checkpoint saved, config snapshot
  written. (smoke-test run directory deleted afterward, not shipped.)
- `python -m training.train_drive_ems --train --config-dir
  configs/final_driving --drive-cycle tests/fixtures/synthetic_test_cycle.csv
  --timesteps 2048 --seed 7 --run-name <tmp>`: ran end-to-end, checkpoint
  + `experiment_config.json` written with a correct timezone-aware
  timestamp. (smoke-test run directory deleted afterward, not shipped.)

## Status

Both tracks remain `READY_FOR_LONG_TRAINING`
(see `audit/long_training_freeze_v2.md`). The frozen configs are now
verified loadable and runnable, not just present on disk.
