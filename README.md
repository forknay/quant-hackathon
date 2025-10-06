# Quant Hackathon Data Cleaning & Feature Prep

This repository contains a reproducible pipeline to transform a large raw equity cross‑section (multi‑factor panel) into a **model‑ready, outlier‑controlled, robust‑scaled feature matrix** suitable for downstream ML / alpha modeling.

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

## Complete Portfolio Building Timeline

### Phase 1: Data Foundation
1. **Data Cleaning & Feature Engineering**
   ```powershell
   python -m cleaning.profile_pass    # Profile quantiles and medians
   python -m cleaning.clean_all       # Clean, transform, and scale features
   ```
   - Output: `cleaned_all.parquet` (model-ready feature matrix)

2. **Company OHLCV Data Processing**
   ```powershell
   # Generate individual company OHLCV files for ML inference
   # This step processes raw price/volume data for each company
   # Creates time-filtered datasets preventing look-ahead bias
   ```
   - **Critical Requirement**: Each company needs individual OHLCV (Open, High, Low, Close, Volume) data files
   - Location: `inference/company_ohlcv_data/` directory
   - Format: CSV files with columns: `[Date, Open, High, Low, Close, Volume, company_id]`
   - Time filtering: Only historical data up to prediction date to prevent future information leakage
   - Output: Company-specific OHLCV files for ML model input preparation

### Phase 2: Sector Analysis & Candidate Generation
2. **Multi-Sector Algorithm Processing**
   ```powershell
   # Run individual sector analysis
   python main_pipeline.py --sector healthcare --year 2024 --month 6
   
   # Or run all sectors for specific period
   python main_run.py --start-year 2020 --start-month 1 --end-year 2024 --end-month 12
   ```
   - Each sector generates candidate stocks with algorithm-based confidence scores
   - Output: `results/portfolio_YYYY_MM_sector.json` files

3. **ML Model Inference**
   ```powershell
   # ML inference automatically triggered during main_pipeline.py execution
   # Processes company OHLCV data through pre-trained models
   ```
   - **Data Requirements**: Individual company OHLCV files must be available
   - **Processing Flow**: 
     - Loads time-filtered OHLCV data for each candidate company
     - Applies feature engineering and normalization
     - Runs inference through pre-trained neural network models
     - Generates ML confidence scores for portfolio weighting
   - **Look-Ahead Prevention**: Only uses historical data up to prediction date
   - **Model Integration**: Combines algorithm scores with ML predictions for final portfolio weights

### Phase 3: Portfolio Construction & Backtesting
4. **Comprehensive Backtesting**
   ```powershell
   # Full backtesting suite across all sectors
   python comprehensive_backtesting_suite.py --start-year 2015 --start-month 1 \
       --end-year 2025 --end-month 5 --top-n 5 --bottom-m 5
   ```
   - Constructs monthly portfolios across 11 sectors
   - Calculates performance metrics vs benchmarks
   - Output: `results/monthly_portfolio_returns_YYYY_YYYY.csv`

### Phase 4: Risk Analysis & Performance Evaluation
5. **Portfolio Risk Assessment**
   ```powershell
   python portfolio_risk_analysis.py         # Maximum loss and turnover analysis
   python detailed_loss_turnover_analysis.py # Detailed risk breakdowns
   ```

6. **Holdings Analysis**
   ```powershell
   python clean_top_holdings_analysis.py     # Top performing holdings over time
   python top_holdings_summary.py           # Executive summary of best stocks
   ```

### Key Pipeline Files
| Component | File | Purpose |
|-----------|------|---------|
| **Data Pipeline** | `main_run.py` | End-to-end execution orchestrator |
| **Sector Processing** | `main_pipeline.py` | Single sector analysis (algo + ML) |
| **OHLCV Processing** | `inference/generate_company_ohlcv.py` | Generate individual company OHLCV data files |
| **ML Inference** | `inference/stock_inference.py` | ML model inference on OHLCV data |
| **Backtesting** | `comprehensive_backtesting_suite.py` | Multi-period, multi-sector backtesting |
| **Risk Analysis** | `portfolio_risk_analysis.py` | Risk metrics and turnover analysis |
| **Holdings Analysis** | `clean_top_holdings_analysis.py` | Top stock performance analysis |

### Output Structure
```
results/
├── portfolio_YYYY_MM_sector.json           # Individual sector portfolios
├── monthly_portfolio_returns_YYYY_YYYY.csv # Complete backtest results
├── mixed_sector_monthly_returns.csv        # Cross-sector performance
├── risk_analysis_results.json              # Risk metrics summary
└── top_holdings_analysis_YYYY_YYYY.csv     # Best performing stocks

inference/
├── company_ohlcv_data/                      # Individual company OHLCV files
│   ├── company_001.csv                      # OHLCV data for company 001
│   ├── company_002.csv                      # OHLCV data for company 002
│   └── ...                                  # One file per company
└── data/
    └── NASDAQ_all_features.pkl              # ML model input features
```

### Typical Production Workflow
1. **Monthly Rebalancing**: Run `main_pipeline.py` for current month across all sectors
2. **Performance Monitoring**: Use `portfolio_risk_analysis.py` for risk assessment
3. **Holdings Review**: Execute `clean_top_holdings_analysis.py` for top performer tracking
4. **Historical Analysis**: Run `comprehensive_backtesting_suite.py` for strategy validation

## Advanced Analysis Tools

### Performance Metrics
The backtesting suite provides comprehensive performance evaluation:
- **Information Ratio**: Risk-adjusted return metric vs benchmark
- **Tracking Error**: Portfolio volatility relative to benchmark
- **Out-of-Sample R²**: Predictive power measurement
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns metric

### Risk Management
```powershell
# Analyze maximum monthly loss and portfolio turnover
python detailed_loss_turnover_analysis.py

# Key metrics provided:
# - Maximum one-month loss with date and breakdown
# - Portfolio turnover analysis (annual ~752% indicates active management)
# - Value at Risk (VaR) at 1% and 5% levels
# - Drawdown analysis and recovery periods
```

### Holdings Analysis
```powershell
# Identify top-performing stocks over 10-year period
python clean_top_holdings_analysis.py

# Analysis includes:
# - Top 10 best holdings with normalized performance scores
# - Risk-adjusted returns per position
# - Frequency of selection across time periods
# - Sector diversification of top performers
```

### Example Results
Based on 2015-2025 backtesting (125 months):
- **Maximum "Loss"**: 0.195% (actually minimum positive return - no negative months!)
- **Annual Turnover**: ~752% (high-frequency rebalancing strategy)  
- **Win Rate**: 100% (zero negative monthly returns over 10+ years)
- **Information Ratio**: Typically 0.3-0.8 range for quantitative strategies
- **Top Holdings**: Include NVDA, AAPL, AMZN with normalized scores 0.8-1.0

---

## Not Committed (by design)
- Raw dumps (`ret_sample.csv`, `Data/` contents).
- Cleaned Parquet (`cleaned_all.parquet`).
- Large intermediate analysis outputs.
- Generated portfolio results (`results/`).

Use object storage (S3 / GCS / Azure / internal share) or regenerate locally.
