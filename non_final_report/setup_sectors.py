#!/usr/bin/env python3
"""
Setup script to copy generalized algorithm files to all sector directories
and create sector-specific configurations.
"""

import os
import shutil
from pathlib import Path

# Sector configuration
SECTORS = {
    'energy': '10',
    'materials': '15',
    'industrials': '20',
    'cons_discretionary': '25',
    'cons_staples': '30',
    'healthcare': '35',
    'financials': '40',
    'it': '45',
    'telecoms': '50',
    'utilities': '55',  # Already exists
    're': '60'
}

# Files to copy from utilities directory
SOURCE_FILES = [
    'sector_mapper.py',
    'indicators.py',
    'candidate_selection.py',
    'pipeline.py'
]

def create_sector_config(sector_name: str, gics_prefix: str) -> str:
    """Create a sector-specific config file."""
    
    # Sector-specific signal parameters
    sector_signal_params = {
        'utilities': {'MA_WINDOW': 60, 'MOM_LAG': 120},
        'cons_staples': {'MA_WINDOW': 60, 'MOM_LAG': 120},
        'energy': {'MA_WINDOW': 45, 'MOM_LAG': 90},
        'materials': {'MA_WINDOW': 45, 'MOM_LAG': 90},
        'industrials': {'MA_WINDOW': 50, 'MOM_LAG': 100},
        'cons_discretionary': {'MA_WINDOW': 40, 'MOM_LAG': 80},
        'healthcare': {'MA_WINDOW': 35, 'MOM_LAG': 70},
        'financials': {'MA_WINDOW': 30, 'MOM_LAG': 60},
        'it': {'MA_WINDOW': 25, 'MOM_LAG': 50},
        'telecoms': {'MA_WINDOW': 55, 'MOM_LAG': 110},
        're': {'MA_WINDOW': 50, 'MOM_LAG': 100}
    }
    
    params = sector_signal_params.get(sector_name, sector_signal_params['utilities'])
    
    config_content = f'''# {sector_name.upper()} Sector Configuration
# Auto-generated configuration for {sector_name} sector

import os

# ============================================================================
# SECTOR CONFIGURATION
# ============================================================================

# Sector-specific settings
SECTOR_NAME = "{sector_name}"
GICS_PREFIX = "{gics_prefix}"

# ============================================================================
# DATA PATHS
# ============================================================================

# Data paths
INPUT_PARQUET = "cleaned_all.parquet" 
INPUT_DAYS_DIR = None
RESULTS_DIR = f"results/{{{sector_name}}}_parquet"

# ============================================================================
# SECTOR-SPECIFIC SIGNAL PARAMETERS
# ============================================================================

# Signal parameters optimized for {sector_name} sector
MA_WINDOW = {params['MA_WINDOW']}
MOM_LAG = {params['MOM_LAG']}

# ============================================================================
# GARCH PARAMETERS
# ============================================================================

# Fixed GARCH parameters per technical recommendations
GARCH_PARAMS = {{
    "p": 1, 
    "q": 1, 
    "dist": "t",
    "min_train": 500,        # Increased from 250 per technical recommendations
    "max_train_window": 750  # Added cap to prevent memory issues
}}

# ============================================================================
# CANDIDATE SELECTION PARAMETERS
# ============================================================================

# Candidate selection parameters per technical recommendations
SELECTION_PARAMS = {{
    "top_k_ratio": 0.15,      # Top 15% uptrending candidates
    "bottom_k_ratio": 0.15,   # Bottom 15% downtrending candidates  
    "vol_band_lower": 0.05,   # 5th percentile volatility filter
    "vol_band_upper": 0.95,   # 95th percentile volatility filter
    "min_market_cap": 100,    # $100M minimum market cap (in millions)
    "min_liquidity": 0.001,   # Minimum daily volume ratio
    "ma_trend_threshold": 0.0, # MA slope threshold for trend detection
    "mom_zscore_threshold": 0.0 # Momentum z-score threshold
}}

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
'''
    return config_content

def setup_sector_directory(sector_name: str, gics_prefix: str):
    """Set up a sector directory with all necessary files."""
    
    print(f"Setting up {sector_name} sector directory...")
    
    # Create sector directory path
    sector_dir = Path(f"algorithm/{sector_name}")
    utilities_dir = Path("algorithm/utilities")
    
    # Create directory if it doesn't exist
    sector_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip utilities directory (already configured)
    if sector_name == 'utilities':
        print(f"  ✓ Skipping utilities (already configured)")
        return
    
    # Copy source files
    for file_name in SOURCE_FILES:
        source_file = utilities_dir / file_name
        target_file = sector_dir / file_name
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"  ✓ Copied {file_name}")
        else:
            print(f"  ❌ Source file not found: {source_file}")
    
    # Create sector-specific config
    config_content = create_sector_config(sector_name, gics_prefix)
    config_file = sector_dir / "config.py"
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"  ✓ Created config.py")
    
    # Create cache directory
    cache_dir = sector_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    print(f"  ✓ Created cache directory")
    
    # Create results directory
    results_dir = sector_dir / f"results/{sector_name}_parquet"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created results directory")

def main():
    """Main setup function."""
    print("="*60)
    print("SETTING UP GENERALIZED SECTOR ALGORITHM")
    print("="*60)
    
    # Change to the repository root
    repo_root = Path(__file__).parent
    os.chdir(repo_root)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Setting up {len(SECTORS)} sectors...")
    
    # Set up each sector directory
    for sector_name, gics_prefix in SECTORS.items():
        try:
            setup_sector_directory(sector_name, gics_prefix)
            print(f"✅ {sector_name} setup complete")
        except Exception as e:
            print(f"❌ Failed to setup {sector_name}: {e}")
        print()
    
    print("="*60)
    print("SECTOR SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Navigate to any sector directory (e.g., cd algorithm/healthcare)")
    print("2. Run the pipeline: python pipeline.py")
    print("3. Or test sector mapping: python sector_mapper.py")
    print()
    print("Available sectors:")
    for sector_name, gics_prefix in SECTORS.items():
        print(f"  - {sector_name} (GICS: {gics_prefix})")

if __name__ == "__main__":
    main() 