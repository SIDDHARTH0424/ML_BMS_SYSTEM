"""
Script to aggregate all RL-BMS repository code and config files into a single labeled file `all_rl_bms_code.py`.
"""

import os

FILES_TO_COMBINE = [
    # 1. Configurations
    "configs/simulation.yaml",
    "configs/battery.yaml",
    "configs/safety.yaml",
    "configs/reward.yaml",
    "configs/ppo.yaml",
    "configs/evaluation.yaml",

    # 2. Environment & Physics Model
    "environment/__init__.py",
    "environment/ecm_model.py",
    "environment/battery_env.py",
    "environment/env_factory.py",

    # 3. Safety Layer
    "safety/__init__.py",
    "safety/safety_layer.py",

    # 4. Baseline Controllers
    "baselines/__init__.py",
    "baselines/base_controller.py",
    "baselines/cc.py",
    "baselines/cccv.py",
    "baselines/adaptive.py",

    # 5. Agents & Training
    "agents/__init__.py",
    "agents/train_ppo.py",

    # 6. Evaluation & Analysis
    "training/__init__.py",
    "training/evaluate.py",
    "training/policy_sensitivity_analysis.py",
    "training/select_best_checkpoint.py",

    # 7. Utilities
    "utils/__init__.py",
    "utils/config.py",
    "utils/logger.py",
    "utils/metrics.py",
    "utils/plotting.py",
    "utils/seed.py",

    # 8. Unit Tests
    "tests/__init__.py",
    "tests/test_ecm.py",
    "tests/test_safety.py",
    "tests/test_environment.py",
    "tests/test_baselines.py",
]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, "all_rl_bms_code.py")

    header_banner = (
        'from __future__ import annotations\n\n'
        '"""\n'
        '==============================================================================\n'
        'RL-BMS CONSOLIDATED CODEBASE\n'
        '==============================================================================\n'
        'This single file contains the complete source code and configuration files\n'
        'for the RL-BMS project. Each component is labeled with its original file path.\n'
        '==============================================================================\n'
        '"""\n\n'
    )

    combined_count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(header_banner)

        for rel_path in FILES_TO_COMBINE:
            abs_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(abs_path):
                print(f"[-] File not found, skipping: {rel_path}")
                continue

            file_header = (
                f"\n\n"
                f"# {'=' * 78}\n"
                f"# FILE: {rel_path}\n"
                f"# LOCAL PATH: file:///{abs_path.replace(os.sep, '/')}\n"
                f"# {'=' * 78}\n\n"
            )
            out_f.write(file_header)

            with open(abs_path, "r", encoding="utf-8") as in_f:
                content = in_f.read()
                if rel_path.endswith((".yaml", ".yml")):
                    content = f'"""\n{content}\n"""'
                else:
                    lines = content.splitlines(keepends=True)
                    filtered_lines = []
                    for line in lines:
                        if line.strip().startswith("from __future__ import"):
                            filtered_lines.append(f"# {line}")
                        else:
                            filtered_lines.append(line)
                    content = "".join(filtered_lines)
                out_f.write(content)

            combined_count += 1
            print(f"[+] Appended: {rel_path}")

    print(f"\nSuccessfully combined {combined_count} files into '{os.path.basename(output_file)}'.")

if __name__ == "__main__":
    main()
