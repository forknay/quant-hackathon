"""Orchestrator to run profiling then cleaning for ret_sample.csv.
Usage (PowerShell):
  python -m cleaning.run_cleaning_pipeline
Or
  python cleaning\run_cleaning_pipeline.py
"""
from __future__ import annotations
import subprocess, sys, shutil

STEPS = [
    ("Profiling pass", [sys.executable, "cleaning/profile_pass.py"]),
    ("Cleaning pass", [sys.executable, "cleaning/clean_all.py"]),
]

def main():
    # Optional: check for pyarrow / pandas presence
    missing = []
    for pkg in ("pandas", "pyarrow"):
        if shutil.which(sys.executable):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install them before running: pip install " + " ".join(missing))
        return
    for label, cmd in STEPS:
        print(f"[orchestrator] START {label}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[orchestrator] ABORT: step '{label}' exited {rc}")
            break
        print(f"[orchestrator] DONE {label}")

if __name__ == "__main__":
    main()
