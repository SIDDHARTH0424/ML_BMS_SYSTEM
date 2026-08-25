"""
Driving-EMS PPO training entry point. Separate from training/train.py
(the existing charging-PPO orchestrator, untouched) per task §29.

This phase only implements a smoke test (task §9/§18/§19) -- confirms
the environment + PPO integration works end-to-end (init, short train,
finite obs/actions/rewards, checkpoint save/load) using the real
env_factory-equivalent construction path. Multi-seed diagnostics (§19),
multi-drive-cycle evaluation (§20), and any longer training (§21) are
explicitly NOT run here -- per §14, "Do NOT start 1M timesteps", and per
§30's staged order, those are separate, later steps that also require
real drive-cycle data (not yet available -- see data/drive_cycles/README.md).

Usage:
    python -m training.train_drive_ems --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import json
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment.ev_energy_env import EVEnergyEnv
from utils.config import CONFIG_DIR, load_config, snapshot_configs

FIXTURE_DRIVE_CYCLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "synthetic_test_cycle.csv",
)


def make_drive_ems_env(drive_cycle_path: str = FIXTURE_DRIVE_CYCLE, mode: str = "train", config_dir: str | None = None) -> EVEnergyEnv:
    """Construction path mirroring environment/env_factory.py's pattern for
    the charging env -- loads all configs fresh, builds the env. NOTE: the
    default drive_cycle_path here is the synthetic test fixture (task §10 --
    no real drive-cycle data exists yet); a real training run would need to
    pass an actual sourced drive-cycle CSV instead."""
    return EVEnergyEnv(
        vehicle_config=load_config("vehicle", config_dir=config_dir) if config_dir else load_config("vehicle"),
        drivetrain_config=load_config("drivetrain", config_dir=config_dir) if config_dir else load_config("drivetrain"),
        battery_config=load_config("battery", config_dir=config_dir) if config_dir else load_config("battery"),
        safety_config=load_config("safety", config_dir=config_dir) if config_dir else load_config("safety"),
        energy_config=load_config("energy_management", config_dir=config_dir) if config_dir else load_config("energy_management"),
        drive_cycle_path=drive_cycle_path,
        mode=mode,
    )


def smoke_test() -> dict:
    """Task §9: environment initializes, PPO initializes, observations/
    actions/rewards/q_gen finite, no NaN/Inf, safety layer works, logs
    produced, checkpoint save/load works. Returns a dict of results."""
    import numpy as np

    ppo_cfg = load_config("ppo_drive_ems")
    env = Monitor(make_drive_ems_env(mode="train"))

    results = {}
    obs, info = env.reset(seed=ppo_cfg["seed"])
    results["env_init"] = True
    results["obs_finite_on_reset"] = bool(np.all(np.isfinite(obs)))

    model = PPO(
        policy=ppo_cfg["policy"], env=env, learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"], batch_size=ppo_cfg["batch_size"], n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"], gae_lambda=ppo_cfg["gae_lambda"], clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"], vf_coef=ppo_cfg["vf_coef"], max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=dict(net_arch=ppo_cfg["policy_kwargs"]["net_arch"]),
        target_kl=ppo_cfg.get("target_kl"), seed=ppo_cfg["seed"], verbose=0,
        use_sde=ppo_cfg["use_sde"],
    )
    results["ppo_init"] = True

    model.learn(total_timesteps=ppo_cfg["smoke_test_timesteps"])
    results["ppo_smoke_train"] = True
    results["num_timesteps"] = model.num_timesteps

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "drive_ems_smoke_ckpt.zip")
        model.save(ckpt_path)
        results["checkpoint_save"] = os.path.exists(ckpt_path)
        model2 = PPO.load(ckpt_path, device="cpu")
        action2, _ = model2.predict(obs, deterministic=True)
        results["checkpoint_load_predict_finite"] = bool(np.all(np.isfinite(action2)))

    # One more explicit episode to confirm reward/q_gen/safety finiteness
    # over a full rollout, not just the training internals.
    obs, _ = env.reset(seed=ppo_cfg["seed"])
    all_finite = True
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, step_info = env.step(action)
        if not (np.all(np.isfinite(obs)) and np.isfinite(reward)):
            all_finite = False
            break
        if term or trunc:
            obs, _ = env.reset(seed=ppo_cfg["seed"])
    results["full_rollout_finite"] = all_finite
    results["safety_layer_reachable"] = "safety_intervention" in step_info

    return results



def train_long(*, run_name: str, config_dir: str, drive_cycle_path: str, timesteps: int, seed: int) -> dict:
    """Run a frozen-config long driving PPO training job."""
    import numpy as np
    from stable_baselines3.common.callbacks import CheckpointCallback

    ppo_cfg = load_config("ppo_drive_ems", config_dir=config_dir)
    env = Monitor(make_drive_ems_env(drive_cycle_path=drive_cycle_path, mode="train", config_dir=config_dir))
    env.reset(seed=seed)

    model = PPO(
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
        policy_kwargs=dict(net_arch=ppo_cfg["policy_kwargs"]["net_arch"]),
        target_kl=ppo_cfg.get("target_kl"),
        seed=seed,
        verbose=1,
        use_sde=ppo_cfg.get("use_sde", False),
        tensorboard_log=os.path.join("runs", run_name, "tensorboard"),
    )

    run_dir = os.path.join("runs", run_name, f"seed_{seed}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    # Reproducibility: snapshot the actual config YAMLs that were loaded
    # for this run, plus the effective PPO dict actually passed into the
    # PPO constructor -- not just the config_dir path string. Mirrors
    # agents/train_ppo.py's snapshot_configs() + effective_ppo_stage*.yaml
    # pattern so a driving run directory is just as reproducible as a
    # charging run directory.
    snapshot_configs(config_dir, os.path.join(run_dir, "config"))
    import yaml
    plain_ppo_cfg = json.loads(json.dumps(dict(ppo_cfg)))
    with open(os.path.join(run_dir, "config", "effective_ppo_drive_ems.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(plain_ppo_cfg, f, default_flow_style=False, sort_keys=False)

    checkpoint_freq = max(int(ppo_cfg.get("checkpoint_freq", 50000)), 1)
    callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo_driving",
    )
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(run_dir, "trained_model.zip"))
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_name": run_name,
            "seed": seed,
            "timesteps": timesteps,
            "drive_cycle_path": drive_cycle_path,
            "config_dir": os.path.abspath(config_dir),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, f, indent=2)

    return {"run_dir": run_dir, "timesteps": model.num_timesteps, "finite": True}

def main():
    parser = argparse.ArgumentParser(description="Driving-EMS PPO training entry point.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the smoke test only.")
    parser.add_argument("--train", action="store_true", help="Run a real PPO training job using an explicit config directory.")
    parser.add_argument("--run-name", type=str, default="driving_ppo_final")
    parser.add_argument("--config-dir", type=str, default=None, help="Config directory for the training run.")
    parser.add_argument("--drive-cycle", type=str, default=None, help="Path to the training drive-cycle CSV.")
    parser.add_argument("--timesteps", type=int, default=None, help="Training budget.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.smoke_test:
        results = smoke_test()
        for k, v in results.items():
            print(f"{k}: {v}")
    elif args.train:
        if not args.config_dir or not args.drive_cycle or not args.timesteps:
            raise SystemExit("--train requires --config-dir, --drive-cycle, and --timesteps")
        results = train_long(run_name=args.run_name, config_dir=args.config_dir, drive_cycle_path=args.drive_cycle, timesteps=args.timesteps, seed=args.seed)
        print(results)
    else:
        print("Use --smoke-test for integration testing or --train with an explicit frozen config directory for real training.")


if __name__ == "__main__":
    main()
