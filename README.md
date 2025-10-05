# Quant Hackathon: Data Cleaning & FinBERT Sentiment Analysis

This repository contains two main components:

1. **Data Cleaning Pipeline**: Transform raw equity cross‑section data into model‑ready, outlier‑controlled, robust‑scaled feature matrix
2. **FinBERT Sentiment Analysis**: Extract sentiment scores from SEC filing texts using FinBERT with enhanced normalization methods

## FinBERT Sentiment Processing Workflow

### Quick Start for FinBERT Analysis

The repository includes a complete FinBERT sentiment analysis system that:
- Extracts real SEC filing text data from the TextData database (2023-2025)
- Processes text through FinBERT for sentiment analysis
- Applies three normalization methods (Min-Max, Softmax, Linear) that all sum to 1.0
- Generates Lightning.ai-ready CSV files for cloud processing

#### Key Files for FinBERT Processing
| File | Purpose |
|------|---------|
| `sentiment_analysis/lightning_ai/process_sentiment_rankings.py` | Main FinBERT processing script with enhanced normalization |
| `sentiment_analysis/data_preparation/prepare_data.py` | Data extraction from TextData parquet files |
| `TextData/` | SEC filing text database (2023-2025) |

#### Usage
```python
# Import the main processing function
from sentiment_analysis.lightning_ai.process_sentiment_rankings import process_sentiment_rankings

# Process any stocks for FinBERT analysis
stocks_to_analyze = ['001004:01', '001013:01', '001019:01']  # GVKEY:IID format
result = process_sentiment_rankings(stocks_to_analyze)

# Output files generated:
# - finbert_input_[timestamp].csv  (ready for Lightning.ai FinBERT processing)
# - metadata_[timestamp].json     (processing metadata)
```

#### Enhanced Normalization Methods
All three normalization methods ensure scores sum exactly to 1.0 across all stocks:

1. **Min-Max Normalization**: Scales to [0,1] range, then normalizes to sum=1
2. **Softmax Normalization**: Probability distribution (natural sum=1)  
3. **Linear Normalization**: Shifts negative values, scales proportionally

#### Real TextData Integration
- Automatically handles 2023-2025 TextData with different formats
- 2023/2025: RF and MGMT columns
- 2024: Single text column  
- Flexible GVKEY handling (numeric and string formats)
- Extracts actual SEC filing content for sentiment analysis

## Data Cleaning Pipeline

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

### Data Cleaning Pipeline
| Path | Purpose |
|------|---------|
| `cleaning/config.py` | Central configuration (paths, chunk size, thresholds, category heuristics). |
| `cleaning/profile_pass.py` | Profiles quantiles + medians saved to `cleaning/profile_stats.json`. |
| `cleaning/clean_all.py` | Streaming cleaner applying winsor → transform → impute → robust scale → write Parquet. |
| `cleaning/profile_stats.json` | Persistent quantile & median stats (rebuild if raw data changes). |
| `cleaning/qa_summary.json` | QA metrics for last cleaning run (clipped counts, missing counts, elapsed time). |

### FinBERT Sentiment Analysis
| Path | Purpose |
|------|---------|
| `sentiment_analysis/lightning_ai/process_sentiment_rankings.py` | Complete FinBERT processing with normalization (sum=1.0) |
| `sentiment_analysis/data_preparation/prepare_data.py` | TextData extraction and formatting |
| `TextData/2023/`, `TextData/2024/`, `TextData/2025/` | SEC filing text database |
| `Data/` | Stock metadata and linking tables |

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

## FinBERT Processing on Lightning.ai

### Step-by-Step Lightning.ai Workflow

1. **Generate FinBERT Input**:
   ```python
   from sentiment_analysis.lightning_ai.process_sentiment_rankings import process_sentiment_rankings
   
   # Process any stocks (GVKEY:IID format)
   stocks = ['001004:01', '001013:01', '001019:01'] 
   process_sentiment_rankings(stocks)
   ```

2. **Upload to Lightning.ai**:
   - Use generated `finbert_input_[timestamp].csv`
   - Upload to Lightning.ai Studio
   - Run FinBERT processing

3. **Download Results & Apply Normalization**:
   ```python
   # After Lightning.ai processing, apply enhanced normalization
   import pandas as pd
   
   # Load FinBERT results
   finbert_results = pd.read_csv('lightning_finbert_results.csv')
   
   # Apply normalization (method='minmax', 'softmax', or 'linear')
   from sentiment_analysis.lightning_ai.process_sentiment_rankings import normalize_sentiment_scores
   normalized = normalize_sentiment_scores(finbert_results, method='minmax')
   
   # Verify normalization: should sum to 1.0
   print(f"Normalized scores sum: {normalized['normalized_score'].sum():.6f}")
   ```

### Normalization Validation
All normalization methods are mathematically validated to sum exactly to 1.000000:
- **Min-Max**: Scales to [0,1], then proportionally adjusts to sum=1
- **Softmax**: Natural probability distribution (e^x / Σe^x)
- **Linear**: Shifts negative values, scales proportionally

### TextData Coverage
- **2023**: 137,402 records with RF/MGMT columns
- **2024**: 86,966 records with single text column  
- **2025**: 2,583 records with RF/MGMT columns
- **Total**: 226,951 SEC filings available for sentiment analysis

## Contributing
PRs should:
1. Avoid committing large binaries.
2. Include a brief note in README or a changelog section for new transforms.
3. For FinBERT changes: test normalization sums to 1.0
4. Provide a smoke test (set `MAX_CHUNKS=1` for data cleaning).

## License
Internal / Hackathon use only (add license text if needed).

---
**Production Ready**: Core FinBERT processing system with enhanced normalization complete and tested.
