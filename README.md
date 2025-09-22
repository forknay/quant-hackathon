# Quant Hackathon Data Cleaning & Feature Prep

This repository contains a reproducible pipeline to transform a large raw equity cross‑section (multi‑factor panel) into a **model‑ready, outlier‑controlled, robust‑scaled feature matrix** suitable for downstream ML / alpha modeling.

## Why this exists
Raw panel data (fundamentals + market microstructure + returns) is noisy: extreme outliers, heterogeneous scales (billions vs 1e-6), skewed positive variables, sparse missing values, mixed types, and slow reload times from huge CSVs. The pipeline standardizes everything without dropping signals:

- Streaming (chunked) reading avoids RAM blow‑ups.
- Winsorization trims only the most extreme tails (quantile clamp) preserving rank ordering of the bulk.
- Skew reduction via `log1p` or signed log on heavy‑tailed features.
- Median imputation for remaining missing values.
- Robust per-period scaling (median/MAD) → comparable dispersion; resilient to leftover outliers.
- Float32 downcast + Parquet write → smaller (≈3.5GB vs ≈9.5GB raw) and faster load.

All original factor columns retained (153 features in current run). No rows were deleted; data is only transformed/stabilized.

## Key Files
| Path | Purpose |
|------|---------|
| `cleaning/config.py` | Central configuration (paths, chunk size, thresholds, category heuristics). |
| `cleaning/profile_pass.py` | Profiles quantiles + medians saved to `cleaning/profile_stats.json`. |
| `cleaning/clean_all.py` | Streaming cleaner applying winsor → transform → impute → robust scale → write Parquet. |
| `cleaning/profile_stats.json` | Persistent quantile & median stats (rebuild if raw data changes). |
| `cleaning/qa_summary.json` | QA metrics for last cleaning run (clipped counts, missing counts, elapsed time). |
| `.gitignore` | Excludes raw & derived large datasets from version control. |

Large raw data (e.g., `ret_sample.csv`, `Data/` directory) and produced Parquet (`cleaned_all.parquet`) are intentionally **not** committed.

## Installation
Create a Python 3.11+ environment (PowerShell example):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Minimal rebuild (PowerShell):
```powershell
python -m cleaning.profile_pass   # refresh stats if raw changed
$env:MAX_CHUNKS="1"; python -m cleaning.clean_all  # optional smoke
Remove-Item Env:MAX_CHUNKS -ErrorAction SilentlyContinue
python -m cleaning.clean_all      # full run
```

## Regenerate the cleaned dataset
1. Place/update raw files (e.g., `ret_sample.csv` and supporting `Data/` CSVs) in the repo root / `Data/`.
2. Run profiling (once per raw snapshot):
```powershell
python -m cleaning.profile_pass
```
3. Run streaming cleaner (env var `MAX_CHUNKS` can limit for a smoke test):
```powershell
$env:MAX_CHUNKS="1"   # (optional) quick test
python -m cleaning.clean_all
```
4. Output: `cleaned_all.parquet` and updated `cleaning/qa_summary.json`.

If `cleaned_all.parquet` already exists and you trust it, you can skip regeneration.

## Statistical Filtering Explained (Short)
| Stage | Action | Why |
|-------|--------|-----|
| Winsorize | Clamp outside pre-profiled quantiles | Neutralize extreme tails without dropping rows |
| Transform | `log1p` / signed log on skewed features | Reduce skew, improve numeric stability |
| Impute | Fill NaNs with medians | Deterministic, robust vs outliers |
| Robust Scale | (x - median) / (MAD + eps) per period | Scale comparably; resilient to remaining anomalies |
| Final Clip | Wide absolute cap | Guardrail against pathological values |
| Downcast | Convert to float32 | Memory & I/O efficiency |

## Reproducibility Contract
Given the same raw CSV(s) and unchanged `config.py` + `profile_stats.json`, the pipeline is deterministic. If raw data changes materially, **re-run** `profile_pass.py` to refresh quantiles before `clean_all.py`.

## Typical Modeling Flow (Downstream)
1. Load Parquet selectively: choose subset of factors.
2. Construct target (e.g., forward return) in a separate step.
3. Perform walk‑forward or rolling cross‑sectional training.
4. Persist trained models / predictions (never commit large artifacts unless using LFS/DVC).

Downstream modeling: create forward target, perform walk-forward; see internal notes or ask team (training example removed for brevity).

---

## Not Committed (by design)
- Raw dumps (`ret_sample.csv`, `Data/` contents).
- Cleaned Parquet (`cleaned_all.parquet`).
- Large intermediate analysis outputs.

Use object storage (S3 / GCS / Azure / internal share) or regenerate locally.

## Quick Integrity Checks
```powershell
python -m cleaning.profile_pass -h 2>$null | Out-Null  # module import sanity
python - <<'PY'
import pyarrow.parquet as pq
print(pq.read_table('cleaned_all.parquet', columns=['stock_ret']).to_pandas().stock_ret.describe())
PY
```

## Updating the Pipeline
1. Modify logic in `clean_all.py` (add feature engineering, adjust thresholds).
2. Re-run a limited chunk smoke: `MAX_CHUNKS=1`.
3. If stable, run full pass.
4. Commit only code + small JSON configs (optional; can ignore QA JSONs).

## Contributing
PRs should:
1. Avoid committing large binaries.
2. Include a brief note in README or a changelog section for new transforms.
3. Provide a smoke test (set `MAX_CHUNKS=1`).

## License
Internal / Hackathon use only (add license text if needed).

---
Prepared: (pending commit) – verify content then commit when ready.
