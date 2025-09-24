import os, glob
import pandas as pd
from joblib import Parallel, delayed

from config import (
    INPUT_PARQUET, INPUT_DAYS_DIR, RESULTS_DIR,
    UTILITIES_GICS_PREFIX,
    MA_WINDOW, MOM_LAG, GARCH_PARAMS, SELECTION_PARAMS,
    COL_DATE, COL_GVKEY, COL_IID, COL_GICS, COL_PRICE, COL_RET_RAW,
    N_JOBS, BATCH_SIZE
)
from indicators import compute_indicators
from candidate_selection import aggregate_monthly_candidates, validate_candidate_selection
from sector_mapper import SectorMapper

# Required columns for indicators computation
USE_COLS = [COL_DATE, COL_GVKEY, COL_IID, COL_PRICE, COL_RET_RAW, 'id']  # Added 'id' for sector mapping

def load_utilities_dataframe() -> pd.DataFrame:
    """
    Load utilities data using the efficient SectorMapper strategy.
    
    This replaces the old approach with:
    1. Efficient utilities identification from sectorinfo
    2. Direct filtering of main parquet data  
    3. Proper GICS sector information
    
    Returns:
        DataFrame with utilities companies data, properly filtered and cleaned
    """
    print("=== Loading Utilities Data with SectorMapper ===")
    
    # Initialize sector mapper
    mapper = SectorMapper(
        sector_csv_path="../../sectorinfo.csv",
        main_parquet_path=INPUT_PARQUET if INPUT_PARQUET.startswith('/') else f"../../{INPUT_PARQUET}",
        cache_dir="cache"
    )
    
    # Load utilities data efficiently
    utilities_data = mapper.load_utilities_data(required_columns=USE_COLS)  # Remove + ['gics'] since it's not in main parquet
    
    print(f"✓ Loaded {len(utilities_data):,} utilities observations")
    print(f"✓ Covering {utilities_data['id'].nunique():,} unique utilities companies")
    
    # Rename columns to match expected format (avoid gics duplication)
    utilities_data = utilities_data.rename(columns={
        COL_DATE: COL_DATE,  # Keep as-is
        'prc': COL_PRICE,
        'stock_ret': COL_RET_RAW,
        'gvkey': COL_GVKEY,
        'iid': COL_IID
    })
    
    # Use gics_str as our GICS column (it's the string version we want)
    if 'gics_str' in utilities_data.columns:
        utilities_data[COL_GICS] = utilities_data['gics_str']
        utilities_data = utilities_data.drop(['gics', 'gics_str'], axis=1, errors='ignore')  # Clean up duplicates
    
    # Data type optimization and cleaning
    print("  - Optimizing data types...")
    utilities_data[COL_GVKEY] = utilities_data[COL_GVKEY].astype("category")
    utilities_data[COL_IID] = utilities_data[COL_IID].astype("category")
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(utilities_data[COL_DATE]):
        utilities_data[COL_DATE] = pd.to_datetime(utilities_data[COL_DATE], errors="coerce")

    # Price and return data cleaning
    utilities_data[COL_PRICE] = pd.to_numeric(utilities_data[COL_PRICE], errors="coerce").abs()
    utilities_data[COL_RET_RAW] = pd.to_numeric(utilities_data[COL_RET_RAW], errors="coerce")

    # Handle potential return scaling issues (same logic as before)
    if utilities_data[COL_RET_RAW].abs().median() > 0.5 or utilities_data[COL_RET_RAW].abs().quantile(0.99) > 5:
        print("  - Rebuilding returns from prices (detected scaling issues)...")
        utilities_data[COL_RET_RAW] = utilities_data.sort_values([COL_GVKEY, COL_IID, COL_DATE]) \
                            .groupby([COL_GVKEY, COL_IID], observed=True)[COL_PRICE] \
                            .pct_change()
    
    # Clean data: remove invalid rows
    initial_count = len(utilities_data)
    utilities_data = utilities_data.dropna(subset=[COL_DATE, COL_GVKEY, COL_IID, COL_PRICE])
    utilities_data = utilities_data.drop_duplicates(subset=[COL_DATE, COL_GVKEY, COL_IID]) \
           .sort_values([COL_GVKEY, COL_IID, COL_DATE])
    
    print(f"  - Data cleaning: {initial_count:,} → {len(utilities_data):,} rows")
    
    # Add convenience partitions
    utilities_data["year"] = utilities_data[COL_DATE].dt.year.astype("int16")
    utilities_data["month"] = utilities_data[COL_DATE].dt.month.astype("int8")
    
    # Validation summary
    print("=== Utilities Data Summary ===")
    print(f"  - Total observations: {len(utilities_data):,}")
    print(f"  - Unique companies: {utilities_data[COL_GVKEY].nunique():,}")
    print(f"  - Date range: {utilities_data[COL_DATE].min()} to {utilities_data[COL_DATE].max()}")
    
    # Debug: print actual columns to see what's available
    print(f"  - Available columns: {list(utilities_data.columns)}")
    
    if COL_GICS in utilities_data.columns:
        gics_values = utilities_data[COL_GICS].dropna().unique()
        print(f"  - GICS codes found: {sorted(gics_values.tolist())}")
    else:
        print(f"  - GICS column '{COL_GICS}' not found after mapping")
    
    return utilities_data

def _process_one(group_key_df):
    """Process indicators for one (gvkey, iid) group."""
    (_, _), df_sec = group_key_df
    
    # Only use columns needed for indicators computation (not GICS)
    cols = [COL_DATE, COL_GVKEY, COL_IID, COL_PRICE, COL_RET_RAW]
    
    # Check that all required columns exist
    missing_cols = [col for col in cols if col not in df_sec.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols} in group {group_key_df[0]}")
        return pd.DataFrame()  # Return empty DataFrame for this group
    
    # Compute indicators with renamed columns
    out = compute_indicators(
        df_sec[cols].sort_values(COL_DATE) \
                    .rename(columns={COL_DATE:"date", COL_GVKEY:"gvkey", COL_IID:"iid",
                                     COL_PRICE:"prc", COL_RET_RAW:"stock_ret"}),
        ma_window=MA_WINDOW,
        mom_lag=MOM_LAG,
        garch_params=GARCH_PARAMS,
        min_train=GARCH_PARAMS.get("min_train", 500)
    )
    
    # Add year/month for partitioning
    out["year"]  = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    
    return out

def run():
    """
    Enhanced utilities pipeline with sector mapping and candidate selection.
    
    Steps:
    1. Load utilities data using SectorMapper (efficient sector filtering)
    2. Compute indicators for all utilities stocks (MA, MOM, GARCH)  
    3. Apply candidate selection with composite scoring
    4. Save both indicators and selected candidates with validation
    """
    print("="*80)
    print("ENHANCED UTILITIES PIPELINE WITH SECTOR MAPPING")
    print("="*80)
    
    try:
        # Step 1: Load utilities data using SectorMapper
        print("\n[STEP 1] Loading utilities data with sector mapping...")
        df = load_utilities_dataframe()
        
        if len(df) == 0:
            print("❌ No utilities data found! Check sector mapping.")
            return
        
        print(f"✓ Loaded {len(df):,} rows for {df.groupby([COL_GVKEY, COL_IID]).ngroups:,} unique securities")
        
        # Step 2: Compute indicators for all securities
        print(f"\n[STEP 2] Computing indicators (MA, MOM, GARCH) for all utilities...")
        groups = list(df.groupby([COL_GVKEY, COL_IID], sort=False))
        print(f"  - Processing {len(groups):,} unique utilities companies...")
        
        # Process in parallel
        indicator_results = Parallel(n_jobs=N_JOBS, prefer="processes", batch_size=BATCH_SIZE)(
            delayed(_process_one)(g) for g in groups
        )
        
        # Combine all results
        all_indicators = pd.concat([result for result in indicator_results if not result.empty], 
                                  ignore_index=True)
        
        print(f"✓ Computed indicators for {len(all_indicators):,} stock-date observations")
        
        # Step 3: Save raw indicators
        print(f"\n[STEP 3] Saving indicators...")
        indicators_dir = f"{RESULTS_DIR}/indicators"
        os.makedirs(indicators_dir, exist_ok=True)
        all_indicators.to_parquet(
            indicators_dir, 
            partition_cols=["year", "month"], 
            engine="pyarrow", 
            compression="zstd"
        )
        print(f"✓ Saved indicators to: {indicators_dir}")
        
        # Step 4: Apply candidate selection
        print(f"\n[STEP 4] Applying candidate selection...")
        
        # Create sector config dict for candidate selection
        sector_config = {
            'ma_window': MA_WINDOW,
            'mom_lag': MOM_LAG,
            'gics_prefix': UTILITIES_GICS_PREFIX
        }
        
        # Apply monthly candidate selection
        selected_candidates = aggregate_monthly_candidates(
            all_indicators, 
            sector_config, 
            SELECTION_PARAMS
        )
        
        if not selected_candidates.empty:
            # Step 5: Validate and save candidates
            print(f"\n[STEP 5] Validating and saving candidates...")
            validation_results = validate_candidate_selection(selected_candidates)
            
            print("✓ Candidate Selection Results:")
            print(f"  - Status: {validation_results['status']}")
            print(f"  - Total candidates: {validation_results['total_candidates']:,}")
            print(f"  - Long candidates: {validation_results['long_candidates']:,}")
            print(f"  - Short candidates: {validation_results['short_candidates']:,}")
            print(f"  - Date range: {validation_results['date_range']['start']} to {validation_results['date_range']['end']}")
            
            if validation_results.get('warnings'):
                print(f"  ⚠ Warnings: {validation_results['warnings']}")
            
            # Save selected candidates
            candidates_dir = f"{RESULTS_DIR}/candidates"
            os.makedirs(candidates_dir, exist_ok=True)
            
            # Add year/month columns for partitioning
            selected_candidates['year'] = selected_candidates['date'].dt.year.astype('int16')
            selected_candidates['month'] = selected_candidates['date'].dt.month.astype('int8')
            
            selected_candidates.to_parquet(
                candidates_dir,
                partition_cols=["year", "month"],
                engine="pyarrow",
                compression="zstd"
            )
            print(f"✓ Saved selected candidates to: {candidates_dir}")
            
            # Save validation summary
            import json
            with open(f"{RESULTS_DIR}/validation_summary.json", 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_results = validation_results.copy()
                if 'date_range' in serializable_results:
                    serializable_results['date_range'] = {
                        'start': str(serializable_results['date_range']['start']),
                        'end': str(serializable_results['date_range']['end'])
                    }
                json.dump(serializable_results, f, indent=2)
            print(f"✓ Saved validation summary to: {RESULTS_DIR}/validation_summary.json")
                
        else:
            print("⚠ Warning: No candidates were selected!")
        
        # Final summary
        print("\n" + "="*80)
        print("✅ ENHANCED UTILITIES PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Results Summary:")
        print(f"  - Utilities companies processed: {df[COL_GVKEY].nunique():,}")
        print(f"  - Total indicators computed: {len(all_indicators):,}")
        if not selected_candidates.empty:
            print(f"  - Candidates selected: {len(selected_candidates):,}")
            print(f"  - Long/Short ratio: {validation_results['long_candidates']}/{validation_results['short_candidates']}")
        print(f"  - Results directory: {RESULTS_DIR}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_sector_mapping_first():
    """Test sector mapping before running full pipeline."""
    print("Testing sector mapping functionality first...")
    
    from sector_mapper import test_sector_mapping
    success = test_sector_mapping()
    
    if success:
        print("\n✅ Sector mapping test passed! Ready to run full pipeline.")
        return True
    else:
        print("\n❌ Sector mapping test failed! Please fix issues before running pipeline.")
        return False

if __name__ == "__main__":
    # Test sector mapping first, then run full pipeline
    if test_sector_mapping_first():
        print("\nStarting full utilities pipeline...")
        run()
    else:
        print("Skipping pipeline due to sector mapping issues.")
