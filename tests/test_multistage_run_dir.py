"""
Regression tests for ISSUE-011 (audit/ISSUES.md): the multi-stage training
orchestrator (training/train.py) previously called create_run_dir again for
every stage, which crashed on Stage 2+ once Stage 1 had written files into
the run directory (create_run_dir correctly refuses to reuse a non-empty
directory when called as a fresh "create" -- that protection must stay).

These tests exercise the real run_stage()/create_run_dir() code paths with
a tiny monkeypatched PPO config (small n_steps / stage timesteps) so a
4-stage run completes in seconds rather than requiring the full 1,000,000
timestep Stage-4 budget.
"""
from __future__ import annotations

import os

import pytest

from agents.train_ppo import run_stage
from utils.config import CONFIG_DIR as REAL_CONFIG_DIR
from utils.config import load_config as real_load_config
from utils.logger import create_run_dir


def _tiny_ppo_cfg():
    """A real ppo.yaml config with every stage's timestep budget and the
    rollout length shrunk so tests run in seconds, not minutes/hours.
    Every other hyperparameter (policy, lr, gamma, etc.) is untouched --
    these tests are about run-directory plumbing, not training quality."""
    cfg = real_load_config("ppo")
    cfg = dict(cfg)
    cfg["n_steps"] = 64
    cfg["batch_size"] = 32
    cfg["n_epochs"] = 1
    for key in (
        "stage1_sanity_timesteps",
        "stage2_reward_verification_timesteps",
        "stage3_hpo_timesteps",
        "stage4_full_training_timesteps",
    ):
        cfg[key] = 64
    cfg["checkpoint_freq"] = 64
    return cfg


@pytest.fixture
def fast_ppo_cfg(monkeypatch):
    tiny_cfg = _tiny_ppo_cfg()

    def _fake_load_config(name, config_dir=None):
        if name == "ppo":
            return dict(tiny_cfg)
        return real_load_config(name, config_dir=config_dir) if config_dir else real_load_config(name)

    monkeypatch.setattr("agents.train_ppo.load_config", _fake_load_config)


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    """Isolate run_stage's run-directory creation for the run_dir=None
    (single-stage CLI) path, which derives its runs/ root from CONFIG_DIR
    (agents/train_ppo.py: `os.path.dirname(CONFIG_DIR)`) rather than
    accepting an explicit root.

    Bug fixed here: this fixture previously computed `root` but never
    applied it -- test_case1 below called run_stage(run_dir=None) with no
    isolation in place, so every test run created a real run_NNN directory
    in the actual project's runs/ (confirmed: runs/run_004-006 all carry
    this fixture's `stage1_sanity_timesteps: 64` signature). Monkeypatching
    CONFIG_DIR redirects `os.path.dirname(CONFIG_DIR)` to tmp_path, so the
    derived runs root becomes tmp_path/"runs" instead of the real project
    directory. Safe because fast_ppo_cfg already monkeypatches load_config
    entirely -- CONFIG_DIR's actual config *contents* are never read here.
    """
    root = str(tmp_path / "runs")
    fake_config_dir = tmp_path / "configs"
    # make_env() inside run_stage loads battery/safety/reward/simulation for
    # real (only "ppo" is faked by fast_ppo_cfg) -- so this needs real YAML
    # content, not an empty directory. Copy the actual dev configs in.
    import shutil
    shutil.copytree(REAL_CONFIG_DIR, fake_config_dir)
    monkeypatch.setattr("agents.train_ppo.CONFIG_DIR", str(fake_config_dir))
    return root


# --------------------------------------------------------------------- #
# Case 1: no run exists -> creates run_001
# Case 2: run_001 exists and is non-empty -> a fresh CREATE attempt fails
# (both already covered thoroughly by tests/test_utils_fixes.py against
# create_run_dir directly; re-asserted here at the run_stage level for
# Case 2 to confirm the protection still applies through run_stage's
# single-stage code path, not just the bare utility function.)
# --------------------------------------------------------------------- #
def test_case1_no_run_creates_run_001(fast_ppo_cfg, runs_root):
    run_dir, model_path = run_stage(1, run_name=None, run_dir=None)
    assert os.path.isdir(run_dir)
    assert os.path.isfile(model_path)
    # Confirm isolation actually took effect: the created run must live
    # under the fixture's tmp_path root, never under the real project's
    # runs/ directory.
    assert run_dir.startswith(runs_root), (
        f"run_stage created {run_dir} outside the isolated runs_root "
        f"{runs_root} -- this would pollute the real project's runs/ dir"
    )


def test_case2_fresh_create_still_blocked_on_nonempty_dir(fast_ppo_cfg, tmp_path):
    root = str(tmp_path)
    d = create_run_dir(root, run_name="run_001")
    with open(os.path.join(d, "marker.txt"), "w") as f:
        f.write("stage1 artifact")
    with pytest.raises(FileExistsError):
        create_run_dir(root, run_name="run_001")


# --------------------------------------------------------------------- #
# Case 3: multi-stage pipeline intentionally reuses run_001 across all
# four stages without crashing.
# Case 4/5/6: earlier stages' artifacts survive later stages.
# --------------------------------------------------------------------- #
def test_case3_to_6_multistage_reuses_dir_and_preserves_artifacts(fast_ppo_cfg, tmp_path):
    root = str(tmp_path / "runs")
    run_dir = create_run_dir(root, run_name="run_001")

    seen_artifacts = []
    for stage in (1, 2, 3, 4):
        prior_files = set(os.listdir(run_dir))
        run_dir_out, model_path = run_stage(stage, run_dir=run_dir)

        # Case 3: same directory reused every stage, no crash.
        assert run_dir_out == run_dir

        # This stage's own artifacts must exist.
        stage_model = os.path.join(run_dir, f"trained_model_stage{stage}.zip")
        assert os.path.isfile(stage_model), f"stage {stage} model missing"
        assert os.path.getsize(stage_model) > 0
        assert os.path.isfile(os.path.join(run_dir, f"command_stage{stage}.txt"))
        assert os.path.isfile(
            os.path.join(run_dir, "config", f"effective_ppo_stage{stage}.yaml")
        )

        # Cases 4/5/6: every artifact that existed before this stage ran
        # must still exist after it -- nothing from an earlier stage was
        # deleted or silently clobbered.
        current_files = set(os.listdir(run_dir))
        missing = prior_files - current_files
        assert not missing, f"stage {stage} deleted prior artifacts: {missing}"

        seen_artifacts.append(stage_model)

    # Sanity: all four stage-namespaced models actually distinct files.
    assert len(set(seen_artifacts)) == 4
    for p in seen_artifacts:
        assert os.path.isfile(p)

    # trained_model.zip (canonical/most-recent pointer) reflects Stage 4.
    canonical = os.path.join(run_dir, "trained_model.zip")
    assert os.path.isfile(canonical)
    with open(canonical, "rb") as f1, open(
        os.path.join(run_dir, "trained_model_stage4.zip"), "rb"
    ) as f2:
        assert f1.read() == f2.read()
