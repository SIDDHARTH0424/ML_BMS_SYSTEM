"""
Orchestrates the full PPO training pipeline: Stage 1 -> 2 -> 3 -> 4.

Usage:
    python -m training.train --run-name run_004
    python -m training.train --run-name run_004 --stages 1 2   # partial run

v2 fix (audit ISSUE-011, see audit/ISSUES.md and agents/train_ppo.py's
run_stage docstring): previously this called run_stage(stage,
run_name=args.run_name) for every stage, which made run_stage call
create_run_dir(run_name=...) again on Stages 2-4 -- and since Stage 1 had
already written files into that directory, create_run_dir's (intentional,
must-stay) non-empty-directory guard raised FileExistsError. Confirmed by
reproducing it directly before applying this fix.

Fix: create the run directory exactly ONCE, before the stage loop, then
pass that same run_dir into every stage. run_stage no longer calls
create_run_dir when a run_dir is supplied.
"""

from __future__ import annotations

import argparse
import os

from agents.train_ppo import run_stage
from utils.config import CONFIG_DIR
from utils.logger import create_run_dir


def main():
    parser = argparse.ArgumentParser(description="Run the full staged PPO training pipeline.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--stages", type=int, nargs="+", default=[1, 2, 3, 4],
        help="Which stages to run, in order (default: all four).",
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="Directory containing the config set to use for this run (for final frozen configs).",
    )
    args = parser.parse_args()

    # Create (or, if --run-name was given and the directory doesn't exist
    # yet, name) the run directory exactly once. This is the ONLY
    # create_run_dir call in the whole pipeline -- it keeps the
    # non-empty-directory overwrite protection fully intact for the
    # "starting a brand new experiment" case, while every stage below
    # reuses this same directory instead of trying to recreate it.
    runs_root = os.path.join(os.path.dirname(CONFIG_DIR), "runs")
    run_dir = create_run_dir(runs_root, run_name=args.run_name)
    print(f"Run directory: {run_dir}")

    model_path = None
    for stage in args.stages:
        print(f"\n=== Running Stage {stage} (run_dir={run_dir}) ===")
        run_dir, model_path = run_stage(stage, run_dir=run_dir, config_dir=args.config_dir)

    print(f"\nPipeline complete. Final model: {model_path}")


if __name__ == "__main__":
    main()
