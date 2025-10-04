"""
END-TO-END PIPELINE TEST
========================

Tests the complete flow:
1. ALGO → Select candidate stocks from healthcare sector  
2. DATA → Process ONLY those stocks' OHLCV data into small .pkl
3. ML/INFERENCE → Rank stocks and pick top N / bottom M
4. OUTPUT → Final stock recommendations

This avoids memory issues by working with a small subset (e.g., 50 companies).

Usage:
    cd test
    python end_to_end_pipeline_test.py
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ml-model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

print("=" * 80)
print("END-TO-END PIPELINE TEST")
print("Testing: ALGO → DATA → ML → OUTPUT")
print("=" * 80)
print()

# =============================================================================
# STEP 1: Extract Candidate Companies from Algo Output
# =============================================================================

def step1_extract_algo_candidates(sector='healthcare', year=2024, month=6, max_candidates=50):
    """
    Extract candidate company IDs from algo results.
    
    Args:
        sector: Sector name (default: healthcare)
        year: Year to extract from
        month: Month to extract from
        max_candidates: Maximum number of candidates to process (for memory management)
    
    Returns:
        list: Company IDs (e.g., ['comp_001004_01', ...])
    """
    print("[STEP 1] Extracting Algo Candidates")
    print("-" * 80)
    
    # Find candidate files
    base_dir = Path(__file__).parent.parent
    candidates_dir = base_dir / f"algo/results/{sector}_parquet/candidates/year={year}/month={month}"
    
    if not candidates_dir.exists():
        print(f"❌ Candidates directory not found: {candidates_dir}")
        print(f"   Please run the algo first:")
        print(f"   cd algo && python run_sector.py")
        return []
    
    # Read all candidate files for this month
    parquet_files = list(candidates_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No candidate files found in {candidates_dir}")
        return []
    
    print(f"✓ Found {len(parquet_files)} candidate file(s)")
    
    # Load candidates
    all_candidates = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        all_candidates.append(df)
    
    df_candidates = pd.concat(all_candidates, ignore_index=True)
    
    # Construct company IDs
    df_candidates['id'] = ('comp_' + 
                           df_candidates['gvkey'].astype(str).str.zfill(6) + 
                           '_' + 
                           df_candidates['iid'].astype(str))
    
    # Get unique company IDs
    candidate_ids = df_candidates['id'].unique().tolist()
    
    print(f"✓ Total unique candidates: {len(candidate_ids)}")
    print(f"  Candidate types: {df_candidates['candidate_type'].value_counts().to_dict()}")
    
    # Limit to max_candidates for memory management
    if len(candidate_ids) > max_candidates:
        print(f"  Limiting to first {max_candidates} candidates (memory management)")
        candidate_ids = candidate_ids[:max_candidates]
    
    print(f"  Selected {len(candidate_ids)} companies for processing")
    print(f"  Sample IDs: {candidate_ids[:5]}")
    print()
    
    return candidate_ids, df_candidates


# =============================================================================
# STEP 2: Process OHLCV Data for Selected Companies Only
# =============================================================================

def step2_create_small_pkl(candidate_ids, output_name='TEST_SMALL'):
    """
    Create a small .pkl file with ONLY the candidate companies.
    
    Args:
        candidate_ids: List of company IDs to process
        output_name: Name for output files (default: TEST_SMALL)
    
    Returns:
        str: Path to created pkl file, or None if failed
    """
    print("[STEP 2] Creating Small .pkl File for Selected Companies")
    print("-" * 80)
    
    if not candidate_ids:
        print("❌ No candidate IDs provided")
        return None
    
    # Import the filtered processing function
    sys.path.insert(0, str(Path(__file__).parent.parent / "ml-model"))
    from data_filtered import process_specific_companies
    
    print(f"Processing {len(candidate_ids)} companies...")
    print("This may take 2-5 minutes depending on data size...")
    print()
    
    try:
        # Call the filtered processing function
        result = process_specific_companies(
            company_ids=candidate_ids,
            market_name=output_name,
            ohlcv_dir=str(Path(__file__).parent.parent / "inference/company_ohlcv_data"),
            begin_date_str='2012-11-19 00:00:00',
            min_data_points=30
        )
        
        pkl_path = Path(__file__).parent.parent / f"ml-model/data/{output_name}_all_features.pkl"
        
        if pkl_path.exists():
            file_size = pkl_path.stat().st_size / (1024**2)  # MB
            print(f"✓ Created {pkl_path}")
            print(f"  File size: {file_size:.1f} MB")
            print(f"  Companies in file: {len(result['all_features'])}")
            print()
            return str(pkl_path)
        else:
            print(f"❌ Failed to create {pkl_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# STEP 3: Run ML Inference on Small Dataset
# =============================================================================

def step3_run_inference(pkl_path, model_path=None, top_n=10, bottom_m=10):
    """
    Run ML inference on the small dataset.
    
    For testing purposes, this generates mock predictions.
    In production, this would call stock_inference.py with a real model.
    
    Args:
        pkl_path: Path to the small pkl file
        model_path: Path to trained model (optional for test)
        top_n: Number of top stocks to return
        bottom_m: Number of bottom stocks to return
    
    Returns:
        dict: Inference results with top/bottom stocks
    """
    print("[STEP 3] Running ML Inference")
    print("-" * 80)
    
    if not os.path.exists(pkl_path):
        print(f"❌ PKL file not found: {pkl_path}")
        return None
    
    # Load data
    print(f"  Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
    except EOFError:
        print("❌ PKL file is incomplete (EOFError)")
        return None
    except Exception as e:
        print(f"❌ Error loading PKL: {e}")
        return None
    
    all_features = data.get('all_features', {})
    num_stocks = len(all_features)
    
    print(f"  Loaded {num_stocks} stocks")
    
    if num_stocks == 0:
        print("❌ No stocks in pkl file!")
        return None
    
    # Check data quality
    valid_stocks = []
    for stock_id, features in all_features.items():
        # Check if stock has enough valid data (at least 32 days for model)
        recent_data = features[-32:, 1:]  # Last 32 days, exclude date column
        if len(recent_data) >= 32 and not np.all(recent_data == -1234):
            valid_stocks.append(stock_id)
    
    print(f"  Stocks with sufficient data for inference: {len(valid_stocks)}")
    
    if len(valid_stocks) == 0:
        print("❌ No stocks have enough valid data for inference!")
        return None
    
    # For test purposes, generate mock predictions
    # In real scenario, would call model.predict()
    print(f"  Generating predictions...")
    print(f"  (Using mock predictions for test - replace with real model)")
    
    # Mock predictions: random scores with some structure
    np.random.seed(42)  # For reproducibility
    predictions = {}
    
    for stock_id in valid_stocks:
        # Generate a score based on some fake logic
        # In reality, this would be model output
        features = all_features[stock_id]
        recent_avg = np.mean(features[features != -1234][-50:])  # Simple feature
        noise = np.random.randn() * 0.1
        score = recent_avg + noise
        predictions[stock_id] = score
    
    # Sort by prediction score
    sorted_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    top_n = min(top_n, len(sorted_stocks))
    bottom_m = min(bottom_m, len(sorted_stocks))
    
    top_stocks = [stock for stock, _ in sorted_stocks[:top_n]]
    top_scores = [score for _, score in sorted_stocks[:top_n]]
    
    bottom_stocks = [stock for stock, _ in sorted_stocks[-bottom_m:]]
    bottom_scores = [score for _, score in sorted_stocks[-bottom_m:]]
    
    results = {
        'total_stocks': num_stocks,
        'valid_stocks': len(valid_stocks),
        'top_n': {
            'stocks': top_stocks,
            'scores': top_scores
        },
        'bottom_m': {
            'stocks': bottom_stocks,
            'scores': bottom_scores
        }
    }
    
    print(f"✓ Generated predictions for {len(valid_stocks)} stocks")
    print(f"  Top {top_n} stocks identified")
    print(f"  Bottom {bottom_m} stocks identified")
    print()
    
    return results


# =============================================================================
# STEP 4: Display Results
# =============================================================================

def step4_display_results(results, algo_candidates_df=None):
    """Display final ranked stocks."""
    print("[STEP 4] Final Results")
    print("-" * 80)
    
    if not results:
        print("❌ No results to display")
        return
    
    print(f"Total stocks analyzed: {results['total_stocks']}")
    print(f"Stocks valid for inference: {results['valid_stocks']}")
    print()
    
    print("TOP STOCKS (Long Candidates):")
    print("-" * 40)
    for i, (stock, score) in enumerate(zip(results['top_n']['stocks'], 
                                           results['top_n']['scores']), 1):
        # Try to get gvkey/iid if we have algo data
        gvkey_iid = ""
        if algo_candidates_df is not None:
            match = algo_candidates_df[algo_candidates_df['id'] == stock]
            if not match.empty:
                gvkey = match.iloc[0]['gvkey']
                iid = match.iloc[0]['iid']
                gvkey_iid = f" (gvkey={gvkey}, iid={iid})"
        
        print(f"  {i:2d}. {stock:<20} Score: {score:7.4f}{gvkey_iid}")
    
    print()
    print("BOTTOM STOCKS (Short Candidates):")
    print("-" * 40)
    for i, (stock, score) in enumerate(zip(results['bottom_m']['stocks'], 
                                           results['bottom_m']['scores']), 1):
        # Try to get gvkey/iid if we have algo data
        gvkey_iid = ""
        if algo_candidates_df is not None:
            match = algo_candidates_df[algo_candidates_df['id'] == stock]
            if not match.empty:
                gvkey = match.iloc[0]['gvkey']
                iid = match.iloc[0]['iid']
                gvkey_iid = f" (gvkey={gvkey}, iid={iid})"
        
        print(f"  {i:2d}. {stock:<20} Score: {score:7.4f}{gvkey_iid}")
    
    print()
    print("=" * 80)
    print("✅ END-TO-END PIPELINE TEST COMPLETE!")
    print("=" * 80)
    print()
    print("Pipeline Flow Verified:")
    print("  1. ✅ ALGO layer selected candidates from healthcare sector")
    print("  2. ✅ DATA layer processed those companies' OHLCV into .pkl")
    print("  3. ✅ ML layer ranked the companies (mock predictions)")
    print("  4. ✅ OUTPUT layer returned top/bottom stocks")
    print()
    print("Next Steps:")
    print("  - Replace mock predictions with real ML model")
    print("  - Integrate with actual inference pipeline")
    print("  - Scale to full dataset once memory optimizations in place")
    print()


# =============================================================================
# Main Test Flow
# =============================================================================

def run_end_to_end_test(sector='healthcare', year=2024, month=6, max_candidates=50):
    """Run the complete end-to-end test."""
    
    print("Configuration:")
    print(f"  Sector: {sector}")
    print(f"  Year: {year}")
    print(f"  Month: {month}")
    print(f"  Max candidates: {max_candidates} (for memory management)")
    print()
    
    # Step 1: Get algo candidates
    result = step1_extract_algo_candidates(
        sector=sector,
        year=year,
        month=month,
        max_candidates=max_candidates
    )
    
    if not result:
        print("❌ Test failed: No candidates from algo")
        return False
    
    candidate_ids, algo_df = result
    
    if not candidate_ids:
        print("❌ Test failed: No candidates from algo")
        return False
    
    # Step 2: Create small pkl
    pkl_path = step2_create_small_pkl(candidate_ids, output_name='TEST_SMALL')
    
    if not pkl_path:
        print("❌ Test failed: Could not create pkl file")
        return False
    
    # Step 3: Run inference
    results = step3_run_inference(pkl_path, top_n=10, bottom_m=10)
    
    if not results:
        print("❌ Test failed: Inference failed")
        return False
    
    # Step 4: Display results
    step4_display_results(results, algo_df)
    
    return True


if __name__ == "__main__":
    # Allow command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='End-to-end pipeline test')
    parser.add_argument('--sector', default='healthcare', help='Sector to test (default: healthcare)')
    parser.add_argument('--year', type=int, default=2024, help='Year (default: 2024)')
    parser.add_argument('--month', type=int, default=6, help='Month (default: 6)')
    parser.add_argument('--max-companies', type=int, default=50, help='Max companies to process (default: 50)')
    
    args = parser.parse_args()
    
    success = run_end_to_end_test(
        sector=args.sector,
        year=args.year,
        month=args.month,
        max_candidates=args.max_companies
    )
    
    sys.exit(0 if success else 1)
