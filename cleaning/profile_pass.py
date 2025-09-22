"""First pass: profile numeric features of ret_sample.csv via sampling.
Outputs quantile & median stats used for winsorization & imputation.
"""
from __future__ import annotations
import os, json, numpy as np, pandas as pd, sys, pathlib
from typing import Dict, List

# Ensure project root on path when running as script
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleaning.config import (
    RAW_FILE, PROFILE_JSON, ID_COLS, TARGET_COL,
    CHUNKSIZE, SAMPLE_PER_COL_PER_CHUNK, RANDOM_SEED
)


def gather_samples() -> Dict[str, List[np.ndarray]]:
    samples: Dict[str, List[np.ndarray]] = {}
    for i, chunk in enumerate(pd.read_csv(RAW_FILE, chunksize=CHUNKSIZE)):
        num_cols = [c for c in chunk.select_dtypes(include=['number']).columns
                    if c not in ID_COLS and c != TARGET_COL]
        for c in num_cols:
            col = chunk[c].dropna()
            if col.empty:
                continue
            take = min(SAMPLE_PER_COL_PER_CHUNK, len(col))
            arr = col.sample(take, random_state=RANDOM_SEED).to_numpy()
            samples.setdefault(c, []).append(arr)
        if (i + 1) % 10 == 0:
            print(f"[profile] scanned ~{(i+1)*CHUNKSIZE:,} rows")
    return samples


def build_profile(samples: Dict[str, List[np.ndarray]]):
    prof = {}
    for col, parts in samples.items():
        vals = np.concatenate(parts)
        prof[col] = {
            "median": float(np.median(vals)),
            "q0005": float(np.quantile(vals, 0.005)),
            "q01": float(np.quantile(vals, 0.01)),
            "q99": float(np.quantile(vals, 0.99)),
            "q995": float(np.quantile(vals, 0.995)),
        }
    return prof


def main():
    os.makedirs("cleaning", exist_ok=True)
    print("[profile] starting sampling pass")
    samples = gather_samples()
    print(f"[profile] building profile for {len(samples)} features")
    prof = build_profile(samples)
    with open(PROFILE_JSON, "w") as f:
        json.dump({"profile": prof}, f)
    print("[profile] saved profile to", PROFILE_JSON)


if __name__ == "__main__":
    main()
