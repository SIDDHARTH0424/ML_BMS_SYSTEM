# Bug Found & Fixed: Test Suite Polluting Real `runs/` Directory

**Discovered**: while verifying Part 11 (config-dir plumbing) of the
"Final Improvements Before Long Training" task, prior to final
packaging.

## Root cause

`agents/train_ppo.py::run_stage()`'s single-stage code path
(`run_dir=None`, e.g. `python -m agents.train_ppo --stage N`) creates
its run directory from a path derived from the module-level
`CONFIG_DIR` constant:

```python
run_dir = create_run_dir(os.path.join(os.path.dirname(CONFIG_DIR), "runs"), run_name=run_name)
```

`tests/test_multistage_run_dir.py::test_case1_no_run_creates_run_001`
calls `run_stage(1, run_name=None, run_dir=None)` — the single-stage
path — and depended on a `runs_root` fixture that computed a
`tmp_path`-based isolation root but **never applied it**: the fixture
returned a value without ever monkeypatching anything, so every test
run created a real `run_NNN` directory inside the actual project's
`runs/`.

## Evidence

`runs/run_004`, `runs/run_005`, and `runs/run_006` all carried the
test fixture's exact signature (`stage1_sanity_timesteps: 64,
checkpoint_freq: 64` — the tiny values `_tiny_ppo_cfg()` sets to make
the test run in seconds). `run_004` predates this session, meaning
this bug had already silently polluted the shipped project at least
once before it was caught here.

## Fix

`tests/test_multistage_run_dir.py`'s `runs_root` fixture now actually
monkeypatches `agents.train_ppo.CONFIG_DIR` to a `tmp_path`-based
directory (populated with a copy of the real `configs/` YAMLs, since
`make_env()` still loads battery/safety/reward/simulation for real —
only `"ppo"` is faked). `test_case1_no_run_creates_run_001` now also
asserts the created run directory is actually under the isolated root,
so a regression would be caught immediately rather than silently
recurring.

Verified: ran the test 1x before/after — `runs/` directory listing
(md5 of `ls runs`) is now byte-identical before and after the test
suite runs; it was not before the fix.

## Cleanup performed

`runs/run_004`, `run_005`, `run_006` — all confirmed test artifacts,
not real experiments — deleted. `runs/run_001`–`run_003` (the
genuine historical baselines referenced throughout `audit/`) are
untouched.

## Regression

213/213 tests pass after the fix (unchanged count — this was a test
isolation bug, not a test-count change).
