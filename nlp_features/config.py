from pathlib import Path

# Paths
TEXTDATA_DIR = Path("TextData")
OUTPUT_DIR = Path("text_features_out")

# Columns
COL_DATE = "date"
COL_GVKEY = "gvkey"
COL_TEXT = "mgmt"

# Model/tokenization
MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_TOKENS = 512
DOC_STRIDE = 32

# Batching
BATCH_SIZE = 16
NUM_WORKERS = 0

# Embedding and PCA
EMBED_DIM = 768
PCA_OUT_DIM = 32
PCA_MODEL_NAME = "pca_finbert.joblib"

# Monthly alignment
MONTH_FREQ = "M"

# Output files
FILING_FEATURES_NAME = "filing_features.parquet"
MONTHLY_FEATURES_NAME = "text_features_monthly.parquet"

