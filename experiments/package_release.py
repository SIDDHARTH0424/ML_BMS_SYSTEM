"""
Release Packaging Script (Stage R)
==================================
Builds two clean ZIP archives:
1. rl-bms-Driving-source-clean.zip (Source, configs, tests, docs, audit reports, drive cycles)
   - Excludes venv/, .pytest_cache/, __pycache__/, *.pyc, and massive binary model files.
2. rl-bms-Driving-results.zip (Runs, evaluation metrics, CSVs, checkpoints, plots, audit logs)

Usage:
    python -m experiments.package_release
"""

from __future__ import annotations

import os
import zipfile


SOURCE_ZIP = "rl-bms-Driving-source-clean.zip"
RESULTS_ZIP = "rl-bms-Driving-results.zip"

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".pytest_cache", ".git", ".gemini"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd"}


def build_source_clean_zip(base_dir: str = "."):
    print(f"Building {SOURCE_ZIP}...")
    with zipfile.ZipFile(SOURCE_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            
            # Skip runs/ folder in clean source (included in results zip)
            rel_root = os.path.relpath(root, base_dir)
            if rel_root.startswith("runs") or rel_root.startswith("."):
                continue
                
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in EXCLUDE_EXTS or file.endswith(".zip"):
                    continue
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, base_dir)
                zf.write(file_path, archive_name)
                
    size_mb = os.path.getsize(SOURCE_ZIP) / (1024 * 1024)
    print(f"  -> Created {SOURCE_ZIP} ({size_mb:.2f} MB)")


def build_results_zip(base_dir: str = "."):
    print(f"Building {RESULTS_ZIP}...")
    with zipfile.ZipFile(RESULTS_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include runs/ folder and audit/ folder
        for target_dir in ["runs", "audit"]:
            if not os.path.exists(target_dir):
                continue
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in EXCLUDE_EXTS:
                        continue
                    file_path = os.path.join(root, file)
                    archive_name = os.path.relpath(file_path, base_dir)
                    zf.write(file_path, archive_name)
                    
    size_mb = os.path.getsize(RESULTS_ZIP) / (1024 * 1024)
    print(f"  -> Created {RESULTS_ZIP} ({size_mb:.2f} MB)")


SOURCE_STAGEQ_ZIP = "rl-bms-Driving-source-clean-stageQ.zip"
SOURCE_FINAL_ZIP = "rl-bms-Driving-source-clean-final.zip"
RESULTS_FINAL_ZIP = "rl-bms-Driving-results-final.zip"


def verify_clean_zip(zip_path: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        dirty = [
            n for n in names
            if "__pycache__" in n
            or ".pytest_cache" in n
            or n.endswith((".pyc", ".pyo", ".pyd"))
            or n.lower().startswith(("venv/", ".venv/"))
            or "runs/" in n
            or n.endswith(".zip")
        ]
        if dirty:
            raise RuntimeError(f"Clean ZIP {zip_path} contained {len(dirty)} dirty files: {dirty[:5]}")
        print(f"  -> Verified {zip_path}: {len(names)} files, 0 dirty/cache files (BAD FILES: 0)")


def build_results_final_zip(base_dir: str = "."):
    print(f"Building {RESULTS_FINAL_ZIP}...")
    with zipfile.ZipFile(RESULTS_FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for target_dir in ["runs", "audit"]:
            if not os.path.exists(target_dir):
                continue
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in EXCLUDE_EXTS:
                        continue
                    file_path = os.path.join(root, file)
                    archive_name = os.path.relpath(file_path, base_dir)
                    zf.write(file_path, archive_name)

    size_mb = os.path.getsize(RESULTS_FINAL_ZIP) / (1024 * 1024)
    print(f"  -> Created {RESULTS_FINAL_ZIP} ({size_mb:.2f} MB)")


def main():
    import shutil
    build_source_clean_zip()
    shutil.copyfile(SOURCE_ZIP, SOURCE_STAGEQ_ZIP)
    print(f"  -> Created {SOURCE_STAGEQ_ZIP} (copy of clean source)")
    shutil.copyfile(SOURCE_ZIP, SOURCE_FINAL_ZIP)
    print(f"  -> Created {SOURCE_FINAL_ZIP} (copy of clean source)")
    build_results_zip()
    build_results_final_zip()
    verify_clean_zip(SOURCE_ZIP)
    verify_clean_zip(SOURCE_STAGEQ_ZIP)
    verify_clean_zip(SOURCE_FINAL_ZIP)
    print("Packaging complete.")


if __name__ == "__main__":
    main()
