# TODO: make sure this is the right path
INPUT_PARQUET = "cleaned_all.parquet" 
# If you still want CSV ingestion later, keep your older path as a fallback:
INPUT_DAYS_DIR = None

# TODO: create right path
RESULTS_DIR = "results/utilities_parquet"

# Utilities sector match (adapt to your GICS encoding)
UTILITIES_GICS_PREFIX = "55"

# Signals for Utilities
MA_WINDOW = 60
MOM_LAG   = 120
GARCH_PARAMS = {"p": 1, "q": 1, "dist": "t"}
MIN_TRAIN = 250

# Column names expected in cleaned output
COL_DATE      = "date"
COL_GVKEY     = "gvkey"
COL_IID       = "iid"
COL_GICS      = "gics"
COL_PRICE     = "prc"
COL_RET_RAW   = "stock_ret"

N_JOBS = -1
BATCH_SIZE = 64
