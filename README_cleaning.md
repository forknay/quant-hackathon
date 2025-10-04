# Data Cleaning Pipeline (Statistical Filtering)

Single-output cleaning of `ret_sample.csv` for hackathon modeling.

## Goals
- Preserve cross-sectional rank information
- Remove destructive extremes
- Normalize heterogeneous scales
- Avoid look-ahead leakage (only same-month statistics for scaling)
- Fast & reproducible (two passes; single output file)
## Reproduction Steps (Concise)
1. Place raw files: ensure `ret_sample.csv` (and any supporting CSVs under `Data/`) are in repo root.
2. Create / activate env (first time only):
	```powershell
	python -m venv .venv
	.\\.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	```
3. Profile (only rerun if raw data changed materially):
	```powershell
	python -m cleaning.profile_pass
	```
4. Smoke test (single chunk):
	```powershell
	$env:MAX_CHUNKS="1"; python -m cleaning.clean_all; Remove-Item Env:MAX_CHUNKS
	```
5. Full build:
	```powershell
	python -m cleaning.clean_all
	```
6. Inspect QA (sanity):
	```powershell
	Get-Content cleaning/qa_summary.json -First 15
	```
7. Load for modeling (example snippet):
	```powershell
	python - <<'PY'
import pandas as pd
df = pd.read_parquet('cleaned_all.parquet')
print(df.shape, 'cols:', len(df.columns))
PY
	```

Minimal command block (PowerShell cheat sheet):
```powershell
python -m cleaning.profile_pass          # profile stats (only if raw changed)
$env:MAX_CHUNKS="1"; python -m cleaning.clean_all   # smoke test
Remove-Item Env:MAX_CHUNKS -ErrorAction SilentlyContinue
python -m cleaning.clean_all             # full build -> cleaned_all.parquet
Get-Content cleaning/qa_summary.json -First 15
```

## Statistical Filtering Summary
| Stage | Operation | Purpose |
|-------|-----------|---------|
| Winsorize | Clamp tails at profiled quantiles | Remove destructive extremes |
| Transform | log1p / signed_log | Reduce skew, improve stability |
| Impute | Fill NaNs with median (+flag) | Preserve rows & signal |
| Robust Scale | (x - monthly median)/MAD | Unitless, outlier-resistant |
| Final Clip | Wide | Guardrail |
| Downcast | float32 | Memory & I/O efficiency |

## Usage Notes
- Commands above suffice; training examples intentionally omitted (create forward return separately).
- This document: conceptual intent, transformations, extension ideas.

## Future Enhancements (Conceptual)
- Sector-neutral residualization
- Feature interaction generation before scaling
- Automated drift detection for re-profiling

## Modifications / Extensions
- Replace robust z with per-month rank -> normal quantile transform
- Sector residualization: feature - sector_median (same month)
- Drift monitor: recompute profile if |median_new - median_old| > 0.5 * old_MAD

## Notes
- Large file: adjust `CHUNKSIZE` in `cleaning/config.py` based on RAM.
- Keep `PROFILE_JSON` versioned for reproducibility.

## License
Internal hackathon use.
