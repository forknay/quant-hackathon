# Generalized Sector Configuration
# This config can be used for any sector by setting the appropriate parameters

import os

# ============================================================================
# SECTOR CONFIGURATION
# ============================================================================

# Default sector settings (can be overridden)
DEFAULT_SECTOR_NAME = "utilities"
DEFAULT_GICS_PREFIX = "55"

# GICS Sector Mapping (for reference and validation)
SECTOR_GICS_MAPPING = {
    'energy': '10',
    'materials': '15', 
    'industrials': '20',
    'cons_discretionary': '25',
    'cons_staples': '30',
    'healthcare': '35',
    'financials': '40',
    'it': '45',
    'telecoms': '50',
    'utilities': '55',
    're': '60'  # Real Estate
}

# Get sector configuration from environment or use defaults
SECTOR_NAME = os.getenv('SECTOR_NAME', DEFAULT_SECTOR_NAME)
GICS_PREFIX = os.getenv('GICS_PREFIX', SECTOR_GICS_MAPPING.get(SECTOR_NAME, DEFAULT_GICS_PREFIX))

# ============================================================================
# DATA PATHS
# ============================================================================

# TODO: make sure this is the right path
INPUT_PARQUET = "cleaned_all.parquet" 
# If you still want CSV ingestion later, keep your older path as a fallback:
INPUT_DAYS_DIR = None

# TODO: create right path - now dynamically based on sector
RESULTS_DIR = f"results/{SECTOR_NAME}_parquet"

# ============================================================================
# SECTOR-SPECIFIC SIGNAL PARAMETERS
# ============================================================================

# Signal parameters can be customized per sector
# Defensive sectors (utilities, cons_staples) typically use longer windows
# Growth sectors (it, healthcare) may use shorter windows

SECTOR_SIGNAL_PARAMS = {
    'utilities': {
        'MA_WINDOW': 60,
        'MOM_LAG': 120,
    },
    'cons_staples': {
        'MA_WINDOW': 60,
        'MOM_LAG': 120,
    },
    'energy': {
        'MA_WINDOW': 45,
        'MOM_LAG': 90,
    },
    'materials': {
        'MA_WINDOW': 45,
        'MOM_LAG': 90,
    },
    'industrials': {
        'MA_WINDOW': 50,
        'MOM_LAG': 100,
    },
    'cons_discretionary': {
        'MA_WINDOW': 40,
        'MOM_LAG': 80,
    },
    'healthcare': {
        'MA_WINDOW': 35,
        'MOM_LAG': 70,
    },
    'financials': {
        'MA_WINDOW': 30,
        'MOM_LAG': 60,
    },
    'it': {
        'MA_WINDOW': 25,
        'MOM_LAG': 50,
    },
    'telecoms': {
        'MA_WINDOW': 55,
        'MOM_LAG': 110,
    },
    're': {
        'MA_WINDOW': 50,
        'MOM_LAG': 100,
    }
}

# Get sector-specific parameters or use defaults
current_params = SECTOR_SIGNAL_PARAMS.get(SECTOR_NAME, SECTOR_SIGNAL_PARAMS['utilities'])
MA_WINDOW = current_params['MA_WINDOW']
MOM_LAG = current_params['MOM_LAG']

# ============================================================================
# GARCH PARAMETERS
# ============================================================================

# Fixed GARCH parameters per technical recommendations
GARCH_PARAMS = {
    "p": 1, 
    "q": 1, 
    "dist": "t",
    "min_train": 500,        # Increased from 250 per technical recommendations
    "max_train_window": 750  # Added cap to prevent memory issues
}

# ============================================================================
# CANDIDATE SELECTION PARAMETERS
# ============================================================================

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

# ============================================================================
# COLUMN NAMES
# ============================================================================

# Column names expected in cleaned output
COL_DATE      = "date"
COL_GVKEY     = "gvkey" 
COL_IID       = "iid"
COL_GICS      = "gics"
COL_PRICE     = "prc"
COL_RET_RAW   = "stock_ret"

# ============================================================================
# PERFORMANCE PARAMETERS
# ============================================================================

N_JOBS = -1
BATCH_SIZE = 64

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_sector_config(sector_name: str = None, gics_prefix: str = None):
    """
    Get configuration for a specific sector.
    
    Args:
        sector_name: Name of the sector
        gics_prefix: GICS prefix for the sector
        
    Returns:
        Dictionary with sector configuration
    """
    if sector_name is None:
        sector_name = SECTOR_NAME
    
    if gics_prefix is None:
        gics_prefix = SECTOR_GICS_MAPPING.get(sector_name, DEFAULT_GICS_PREFIX)
    
    sector_params = SECTOR_SIGNAL_PARAMS.get(sector_name, SECTOR_SIGNAL_PARAMS['utilities'])
    
    return {
        'sector_name': sector_name,
        'gics_prefix': gics_prefix,
        'results_dir': f"results/{sector_name}_parquet",
        'ma_window': sector_params['MA_WINDOW'],
        'mom_lag': sector_params['MOM_LAG'],
        'garch_params': GARCH_PARAMS.copy(),
        'selection_params': SELECTION_PARAMS.copy(),
        'input_parquet': INPUT_PARQUET,
        'input_days_dir': INPUT_DAYS_DIR,
        'n_jobs': N_JOBS,
        'batch_size': BATCH_SIZE
    }

def set_sector_config(sector_name: str):
    """
    Set global configuration for a specific sector.
    
    Args:
        sector_name: Name of the sector to configure
    """
    global SECTOR_NAME, GICS_PREFIX, RESULTS_DIR, MA_WINDOW, MOM_LAG
    
    if sector_name not in SECTOR_GICS_MAPPING:
        raise ValueError(f"Unknown sector: {sector_name}. Available sectors: {list(SECTOR_GICS_MAPPING.keys())}")
    
    SECTOR_NAME = sector_name
    GICS_PREFIX = SECTOR_GICS_MAPPING[sector_name]
    RESULTS_DIR = f"results/{sector_name}_parquet"
    
    sector_params = SECTOR_SIGNAL_PARAMS[sector_name]
    MA_WINDOW = sector_params['MA_WINDOW']
    MOM_LAG = sector_params['MOM_LAG']
    
    print(f"✓ Configuration set for sector: {sector_name}")
    print(f"  - GICS Prefix: {GICS_PREFIX}")
    print(f"  - MA Window: {MA_WINDOW}")
    print(f"  - Momentum Lag: {MOM_LAG}")
    print(f"  - Results Directory: {RESULTS_DIR}")

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Legacy constants for backward compatibility
UTILITIES_GICS_PREFIX = "55"  # Deprecated: Use GICS_PREFIX instead
