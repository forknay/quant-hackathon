# TODO: make sure this is the right path
INPUT_PARQUET = "cleaned_all.parquet" 
# If you still want CSV ingestion later, keep your older path as a fallback:
INPUT_DAYS_DIR = None

# TODO: create right path
RESULTS_DIR = "results/utilities_parquet"

# Utilities sector match (adapt to your GICS encoding)
UTILITIES_GICS_PREFIX = "55"

# Signals for Utilities (defensive sector - longer windows per implementation.md)
MA_WINDOW = 60
MOM_LAG   = 120

# Fixed GARCH parameters per technical recommendations
GARCH_PARAMS = {
    "p": 1, 
    "q": 1, 
    "dist": "t",
    "min_train": 500,        # Increased from 250 per technical recommendations
    "max_train_window": 750  # Added cap to prevent memory issues
}

# Candidate selection parameters per technical recommendations
SELECTION_PARAMS = {
    "top_k_ratio": 0.15,      # Top 15% uptrending candidates
    "bottom_k_ratio": 0.15,   # Bottom 15% downtrending candidates  
    "vol_band_lower": 0.05,   # 5th percentile volatility filter
    "vol_band_upper": 0.95,   # 95th percentile volatility filter
    "min_market_cap": 100,    # $100M minimum market cap (in millions)
    "min_liquidity": 0.001,   # Minimum daily volume ratio
    "ma_trend_threshold": 0.0, # MA slope threshold for trend detection
    "mom_zscore_threshold": 0.0 # Momentum z-score threshold
}

# Column names expected in cleaned output
COL_DATE      = "date"
COL_GVKEY     = "gvkey" 
COL_IID       = "iid"
COL_GICS      = "gics"
COL_PRICE     = "prc"
COL_RET_RAW   = "stock_ret"

N_JOBS = -1
BATCH_SIZE = 64
