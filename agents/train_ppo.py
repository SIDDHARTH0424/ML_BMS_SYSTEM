"""
PPO agent training, staged per the implementation plan:

  Stage 1: short sanity run       (crashes / NaNs / reward sanity)
  Stage 2: reward verification    (log every reward component)
  Stage 3: hyperparameter search  (small sweep over lr/batch/ent_coef)
  Stage 4: full training          (full timestep budget, checkpoints, TensorBoard)

Usage:
    python -m agents.train_ppo --stage 1
    python -m agents.train_ppo --stage 4 --run-name run_004
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.env_factory import make_env
from utils.config import CONFIG_DIR, load_config, snapshot_configs
from utils.logger import CSVLogger, create_run_dir
from utils.seed import set_global_seed


class RewardComponentLoggingCallback(BaseCallback):
    """Stage-2 style callback: logs each reward component to CSV every step."""

    def __init__(self, csv_logger: CSVLogger, verbose: int = 0):
        super().__init__(verbose)
        self.csv_logger = csv_logger

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            components = info.get("reward_components")
            if components:
                row = dict(components)
                row["timestep"] = self.num_timesteps
                self.csv_logger.log(row)
        return True


def build_agent(env, ppo_cfg: dict, tensorboard_dir: str) -> PPO:
    return PPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=dict(ppo_cfg["policy_kwargs"]),
        target_kl=ppo_cfg.get("target_kl"),
        tensorboard_log=tensorboard_dir,
        seed=ppo_cfg["seed"],
        verbose=1,
    )


def run_stage(
    stage: int,
    run_name: str | None = None,
    hpo_overrides: dict | None = None,
    run_dir: str | None = None,
    config_dir: str | None = None,
):
    """Run one training stage.

    v4 fix (audit ISSUE-011, see audit/ISSUES.md): previously this function
    unconditionally called create_run_dir(run_name=run_name) every time it
    was invoked. training/train.py's multi-stage orchestrator reused the
    run name it got back from Stage 1 for Stages 2-4, which meant Stage 2
    called create_run_dir("run_001") again -- and since Stage 1 had already
    written files into runs/run_001/, create_run_dir's (correct, must-stay)
    non-empty-directory guard raised FileExistsError on every multi-stage
    run past Stage 1. Confirmed by reproducing it directly before this fix.

    Fix: separate CREATE from REUSE.
      - If `run_dir` is given (multi-stage orchestrator path): use it
        directly, do NOT call create_run_dir again. The directory (and its
        config/checkpoints/tensorboard/plots subdirs) was already created
        once, by the caller, before the stage loop started.
      - If `run_dir` is None (single-stage CLI path, `python -m
        agents.train_ppo --stage N`, unchanged from before): call
        create_run_dir as before, including its overwrite protection.
    """
    active_config_dir = config_dir or CONFIG_DIR
    ppo_cfg = load_config("ppo", config_dir=active_config_dir)

    # v3 fix: apply CLI/HPO overrides BEFORE snapshotting — the previous
    # order saved the raw YAML defaults regardless of e.g. `--lr 0.0001`,
    # so a run's saved config/ directory didn't reflect what was actually
    # used to train it (a real reproducibility bug: Stage 3 HPO sweep runs
    # were undocumented).
    if hpo_overrides:
        for k, v in hpo_overrides.items():
            ppo_cfg[k] = v

    set_global_seed(ppo_cfg["seed"])

    if run_dir is None:
        run_dir = create_run_dir(os.path.join(os.path.dirname(CONFIG_DIR), "runs"), run_name=run_name)
    snapshot_configs(active_config_dir, os.path.join(run_dir, "config"))

    # Save the EFFECTIVE ppo config (post-override) separately from the raw
    # snapshot above, plus the exact invoking command, so a run directory is
    # self-describing even when CLI overrides were used. Namespaced per
    # stage (v4 fix, ISSUE-011) so a later stage's snapshot never silently
    # overwrites an earlier stage's -- e.g. Stage 3's HPO override values
    # must remain inspectable even after Stage 4 has also run in the same
    # run_dir.
    def _to_plain(obj):
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_plain(v) for v in obj]
        return obj

    with open(os.path.join(run_dir, "config", f"effective_ppo_stage{stage}.yaml"), "w") as f:
        yaml.safe_dump(_to_plain(ppo_cfg), f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(run_dir, f"command_stage{stage}.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    env = Monitor(make_env(mode="train", config_dir=active_config_dir))

    model = build_agent(env, ppo_cfg, tensorboard_dir=os.path.join(run_dir, "tensorboard"))

    timesteps_by_stage = {
        1: ppo_cfg["stage1_sanity_timesteps"],
        2: ppo_cfg["stage2_reward_verification_timesteps"],
        3: ppo_cfg["stage3_hpo_timesteps"],
        4: ppo_cfg["stage4_full_training_timesteps"],
    }
    total_timesteps = timesteps_by_stage[stage]

    callbacks = []
    if stage == 2:
        csv_logger = CSVLogger(os.path.join(run_dir, "reward_components.csv"))
        callbacks.append(RewardComponentLoggingCallback(csv_logger))
    if stage == 4:
        callbacks.append(
            CheckpointCallback(
                save_freq=ppo_cfg["checkpoint_freq"],
                save_path=os.path.join(run_dir, "checkpoints"),
                name_prefix="ppo_bms",
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks or None)

    # v4 fix (ISSUE-011): save a stage-namespaced copy FIRST so an earlier
    # stage's model is never silently lost when a later stage overwrites
    # the canonical trained_model.zip path (each stage builds a fresh PPO
    # model from scratch -- see audit/STAGE_PIPELINE.md -- so without this,
    # Stage 2 running would delete/replace Stage 1's saved model, etc.).
    # trained_model.zip itself is kept pointing at the most-recently-
    # completed stage's model, since training/select_best_checkpoint.py,
    # training/evaluate.py, and the README/docs all reference that fixed
    # path as "the" trained model for a run.
    stage_model_path = os.path.join(run_dir, f"trained_model_stage{stage}.zip")
    model.save(stage_model_path)
    model_path = os.path.join(run_dir, "trained_model.zip")
    shutil.copyfile(stage_model_path, model_path)

    # Stage 1 sanity check: verify no NaNs crept into the policy weights.
    if stage == 1:
        for param in model.policy.parameters():
            if not np.all(np.isfinite(param.detach().cpu().numpy())):
                raise RuntimeError("NaN/Inf detected in policy parameters after sanity run.")
        print(f"[Stage 1] Sanity run complete, no NaNs detected. Model saved to {model_path}")
    else:
        print(f"[Stage {stage}] Training complete. Model saved to {model_path}")

    return run_dir, model_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for RL-BMS.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (stage 3 HPO)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (stage 3 HPO)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Override entropy coef (stage 3 HPO)")
    parser.add_argument("--config-dir", type=str, default=None, help="Directory containing final/default config set")
    args = parser.parse_args()

    overrides = {}
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.ent_coef is not None:
        overrides["ent_coef"] = args.ent_coef

    run_stage(args.stage, run_name=args.run_name, hpo_overrides=overrides or None, config_dir=args.config_dir)


if __name__ == "__main__":
    main()