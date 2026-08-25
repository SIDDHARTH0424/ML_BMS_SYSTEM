"""
Regression test for audit ISSUE-017 (Issue 10 in the task): verify
checkpoint saving is not just "a file appears" but produces a genuinely
loadable, non-empty, usable SB3 checkpoint. Runs a short real Stage-4-style
training with CheckpointCallback active (monkeypatched to a tiny budget so
this completes in seconds), then loads the saved checkpoint back and
confirms it can actually predict an action.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
from stable_baselines3 import PPO

from agents.train_ppo import run_stage
from utils.config import load_config as real_load_config
from utils.logger import create_run_dir


@pytest.fixture
def fast_stage4_cfg(monkeypatch):
    cfg = dict(real_load_config("ppo"))
    cfg["n_steps"] = 64
    cfg["batch_size"] = 32
    cfg["n_epochs"] = 1
    cfg["stage4_full_training_timesteps"] = 128
    cfg["checkpoint_freq"] = 64  # forces >=1 intermediate checkpoint at 128 steps

    def _fake_load_config(name, config_dir=None):
        if name == "ppo":
            return dict(cfg)
        return real_load_config(name, config_dir=config_dir) if config_dir else real_load_config(name)

    monkeypatch.setattr("agents.train_ppo.load_config", _fake_load_config)


def test_stage4_checkpoint_saved_loadable_and_usable(fast_stage4_cfg, tmp_path):
    root = str(tmp_path / "runs")
    run_dir = create_run_dir(root, run_name="run_ckpt_test")

    run_dir_out, model_path = run_stage(4, run_dir=run_dir)

    # Final model path exists and is non-trivially sized (a truly empty/
    # corrupted SB3 zip would be a few hundred bytes at most; a real MLP
    # policy checkpoint is tens of KB+).
    assert os.path.isfile(model_path)
    assert os.path.getsize(model_path) > 1000

    # CheckpointCallback's intermediate checkpoint(s) also exist.
    intermediate = glob.glob(os.path.join(run_dir, "checkpoints", "*.zip"))
    assert len(intermediate) >= 1, "CheckpointCallback produced no intermediate checkpoint files"
    for ckpt in intermediate:
        assert os.path.getsize(ckpt) > 1000

    # The checkpoint must actually be loadable (not corrupted) and usable
    # for inference -- this is the real test of "checkpoint saving works",
    # not just file existence.
    loaded = PPO.load(model_path)
    dummy_obs = np.zeros((6,), dtype=np.float32)  # matches the 6-dim observation space
    action, _ = loaded.predict(dummy_obs, deterministic=True)
    assert np.isfinite(action).all()
    assert action.shape == (1,)  # matches the 1-dim action space

    for ckpt in intermediate:
        loaded_ckpt = PPO.load(ckpt)
        action_ckpt, _ = loaded_ckpt.predict(dummy_obs, deterministic=True)
        assert np.isfinite(action_ckpt).all()
