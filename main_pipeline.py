#!/usr/bin/env python3
"""
MAIN QUANTITATIVE TRADING PIPELINE
==================================

Complete workflow for quantitative stock selection and portfolio construction:

1. SECTOR SELECTION → Run algorithm to identify candidate stocks
2. CANDIDATE FILTERING → Extract top N and bottom M candidates with confidence scores
3. DATA PREPARATION → Generate filtered OHLCV data for each batch (preventing look-ahead bias)
4. ML INFERENCE → Run pre-trained model on each batch to get ML confidence scores
5. PORTFOLIO CONSTRUCTION → Combine algo + ML scores for final portfolio weights

Usage:
    # Basic usage with healthcare sector
    python main_pipeline.py --sector healthcare --year 2024 --month 6

    # Advanced usage with custom parameters
    python main_pipeline.py --sector healthcare --year 2024 --month 6 \
        --top-n 20 --bottom-m 15 \
        --model ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tv_100.ckpt \
        --output results/portfolio_2024_06_healthcare.json

    # List available sectors
    python main_pipeline.py --list-sectors
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import shutil
from typing import Dict, List, Tuple, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent / "algo" / "core"))
sys.path.insert(0, str(Path(__file__).parent / "ml-model"))
sys.path.insert(0, str(Path(__file__).parent / "inference"))

print("=" * 100)
print("QUANTITATIVE TRADING PIPELINE")
print("ALGO → DATA → ML → PORTFOLIO")
print("=" * 100)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

SECTOR_GICS_MAPPING = {
    'energy': '10', 'materials': '15', 'industrials': '20',
    'cons_discretionary': '25', 'cons_staples': '30', 'healthcare': '35',
    'financials': '40', 'it': '45', 'telecoms': '50',
    'utilities': '55', 're': '60'
}

DEFAULT_MODEL_PATH = "ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tv_100.ckpt"

# =============================================================================
# STEP 1: RUN ALGORITHM FOR SECTOR
# =============================================================================

def step1_run_algo(sector: str) -> bool:
    """
    Run the algorithm for the specified sector.
    
    Args:
        sector: Sector name (e.g., 'healthcare')
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"[STEP 1] Running Algorithm for {sector.upper()} Sector")
    print("-" * 80)
    
    if sector not in SECTOR_GICS_MAPPING:
        print(f"❌ Unknown sector: {sector}")
        print(f"   Available sectors: {list(SECTOR_GICS_MAPPING.keys())}")
        return False
    
    # Set environment variables for the sector
    os.environ['SECTOR_NAME'] = sector
    os.environ['GICS_PREFIX'] = SECTOR_GICS_MAPPING[sector]
    
    print(f"🎯 Running {sector} analysis (GICS: {SECTOR_GICS_MAPPING[sector]})...")
    
    try:
        # Import and run the algorithm
        from pipeline import run
        run()
        print(f"✅ {sector} algorithm completed successfully!")
        print()
        return True
    except Exception as e:
        print(f"❌ Error running {sector} algorithm: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STEP 2: EXTRACT CANDIDATES WITH CONFIDENCE SCORES
# =============================================================================

def step2_extract_candidates(sector: str, year: int, month: int, top_n: int, bottom_m: int) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Extract top N and bottom M candidates from algorithm results with look-ahead bias protection.
    
    Args:
        sector: Sector name
        year: Target year
        month: Target month
        top_n: Number of top candidates (long positions)
        bottom_m: Number of bottom candidates (short positions)
    
    Returns:
        Tuple of (top_candidates_dict, bottom_candidates_dict) or (None, None) if failed
        Each dict contains: {'company_ids': [...], 'algo_scores': [...], 'df': DataFrame}
    """
    print(f"[STEP 2] Extracting Top {top_n} and Bottom {bottom_m} Candidates")
    print("-" * 80)
    
    # Calculate cutoff date for look-ahead bias protection
    prediction_date = datetime(year, month, 1)
    cutoff_date = prediction_date - timedelta(days=1)
    
    print(f"🔒 LOOK-AHEAD BIAS PROTECTION:")
    print(f"   Prediction date: {prediction_date.strftime('%Y-%m-%d')}")
    print(f"   Data cutoff: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"   ⚠️  Only companies with sufficient data before {cutoff_date.strftime('%Y-%m-%d')} will be considered")
    print()
    
    # Find candidate files
    candidates_dir = Path(f"algo/results/{sector}_parquet/candidates/year={year}/month={month}")
    
    if not candidates_dir.exists():
        print(f"❌ Candidates directory not found: {candidates_dir}")
        print(f"   Please ensure the algorithm has been run for {sector} {year}/{month}")
        return None, None
    
    # Read all candidate files for this month
    parquet_files = list(candidates_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No candidate files found in {candidates_dir}")
        return None, None
    
    print(f"✓ Found {len(parquet_files)} candidate file(s)")
    
    # Load and combine all candidates
    all_candidates = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        all_candidates.append(df)
    
    df_candidates = pd.concat(all_candidates, ignore_index=True)
    
    # Construct company IDs (format to match OHLCV filenames)
    df_candidates['company_id'] = ('comp_' + 
                                   df_candidates['gvkey'].astype(float).astype(int).astype(str).str.zfill(6) + 
                                   '_' + 
                                   df_candidates['iid'].astype(str))
    
    print(f"✓ Total candidates found: {len(df_candidates)}")
    print(f"  Candidate types: {df_candidates['candidate_type'].value_counts().to_dict()}")
    
    # Filter candidates based on available OHLCV data (consistent with Step 3 processing)
    print(f"📊 Filtering candidates with sufficient historical data...")
    
    # Calculate begin_date (same logic as Step 3)
    begin_date = prediction_date.replace(year=prediction_date.year - 3)
    print(f"   📅 Using begin_date: {begin_date.strftime('%Y-%m-%d')} (3 years before prediction)")
    print(f"   📅 Data must span: {begin_date.strftime('%Y-%m-%d')} to {cutoff_date.strftime('%Y-%m-%d')}")
    
    original_ohlcv_dir = Path("inference/company_ohlcv_data")
    valid_candidates = []
    
    for _, row in df_candidates.iterrows():
        company_id = row['company_id']
        ohlcv_file = original_ohlcv_dir / f"{company_id}_ohlcv.csv"
        
        if ohlcv_file.exists():
            try:
                # Check if company has sufficient data in the required window
                df_ohlcv = pd.read_csv(ohlcv_file)
                df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'])
                
                # Apply both begin_date and cutoff_date filters (consistent with Step 3)
                df_filtered = df_ohlcv[
                    (df_ohlcv['Date'] >= begin_date) & 
                    (df_ohlcv['Date'] <= cutoff_date)
                ]
                
                # Require at least 30 data points in the filtered window 
                # (accounting for quarterly data frequency and Step 3 processing requirements)
                if len(df_filtered) >= 30:
                    valid_candidates.append(row)
                    
            except Exception as e:
                # Skip companies with data loading errors
                continue
    
    if not valid_candidates:
        print(f"❌ No candidates with sufficient historical data found!")
        print(f"   Try a later prediction date or check OHLCV data availability")
        return None, None
    
    df_valid = pd.DataFrame(valid_candidates)
    
    print(f"✓ Valid candidates after filtering: {len(df_valid)} (was {len(df_candidates)})")
    print(f"  Removed: {len(df_candidates) - len(df_valid)} candidates with insufficient data")
    
    # Sort by composite score to get top and bottom candidates
    df_sorted = df_valid.sort_values('composite_score', ascending=False)
    
    # Check if we have enough candidates
    if len(df_sorted) < top_n:
        print(f"⚠️  Warning: Only {len(df_sorted)} candidates available, but {top_n} requested for LONG positions")
        top_n = len(df_sorted)
    
    if len(df_sorted) < bottom_m:
        print(f"⚠️  Warning: Only {len(df_sorted)} candidates available, but {bottom_m} requested for SHORT positions")
        bottom_m = len(df_sorted)
    
    if len(df_sorted) < (top_n + bottom_m):
        print(f"⚠️  Warning: Not enough candidates for both LONG and SHORT positions")
        # Prioritize LONG positions, then SHORT
        available_for_short = max(0, len(df_sorted) - top_n)
        bottom_m = min(bottom_m, available_for_short)
    
    # Extract top N (highest scores - long positions)
    top_candidates = df_sorted.head(top_n).copy()
    top_company_ids = top_candidates['company_id'].tolist()
    top_algo_scores = top_candidates['composite_score'].tolist()
    
    # Extract bottom M (lowest scores - short positions)
    bottom_candidates = df_sorted.tail(bottom_m).copy()
    bottom_company_ids = bottom_candidates['company_id'].tolist()
    bottom_algo_scores = bottom_candidates['composite_score'].tolist()
    
    print(f"✅ Selected TOP {len(top_company_ids)} companies (LONG positions)")
    print(f"  Score range: {min(top_algo_scores):.4f} to {max(top_algo_scores):.4f}")
    print(f"  Sample: {top_company_ids[:3]}")
    print()
    
    print(f"✅ Selected BOTTOM {len(bottom_company_ids)} companies (SHORT positions)")
    print(f"  Score range: {min(bottom_algo_scores):.4f} to {max(bottom_algo_scores):.4f}")
    print(f"  Sample: {bottom_company_ids[:3]}")
    print()
    
    top_dict = {
        'company_ids': top_company_ids,
        'algo_scores': top_algo_scores,
        'df': top_candidates
    }
    
    bottom_dict = {
        'company_ids': bottom_company_ids,
        'algo_scores': bottom_algo_scores,
        'df': bottom_candidates
    }
    
    return top_dict, bottom_dict


# =============================================================================
# STEP 3: PREPARE DATA FOR EACH BATCH
# =============================================================================

def step3_prepare_data_batch(company_ids: List[str], batch_name: str, 
                           prediction_year: int, prediction_month: int) -> Optional[str]:
    """
    Prepare filtered OHLCV data for a batch of companies (companies already pre-filtered for look-ahead bias).
    
    Args:
        company_ids: List of company IDs to process (pre-filtered in Step 2)
        batch_name: Name for this batch (e.g., 'TOP_LONG', 'BOTTOM_SHORT')
        prediction_year: Year of prediction (for cutoff)
        prediction_month: Month of prediction (for cutoff)
    
    Returns:
        str: Path to created pkl file, or None if failed
    """
    print(f"[STEP 3] Preparing Data for {batch_name} Batch ({len(company_ids)} companies)")
    print("-" * 80)
    
    if not company_ids:
        print("❌ No company IDs provided")
        return None
    
    # Calculate cutoff date (day before prediction date)
    prediction_date = datetime(prediction_year, prediction_month, 1)
    cutoff_date = prediction_date - timedelta(days=1)
    
    print(f"🔒 LOOK-AHEAD BIAS PROTECTION:")
    print(f"   Prediction date: {prediction_date.strftime('%Y-%m-%d')}")
    print(f"   Data cutoff: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"   ✅ Companies already pre-filtered in Step 2 for sufficient historical data")
    print()
    
    # Create filtered OHLCV directory
    original_ohlcv_dir = Path("inference/company_ohlcv_data")
    filtered_ohlcv_dir = Path(f"temp/filtered_ohlcv_{batch_name}")
    
    if filtered_ohlcv_dir.exists():
        shutil.rmtree(filtered_ohlcv_dir)
    filtered_ohlcv_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating time-filtered OHLCV data...")
    
    processed_count = 0
    
    for company_id in company_ids:
        original_file = original_ohlcv_dir / f"{company_id}_ohlcv.csv"
        filtered_file = filtered_ohlcv_dir / f"{company_id}_ohlcv.csv"
        
        if not original_file.exists():
            print(f"  ⚠️  {company_id}: OHLCV file not found - skipping")
            continue
        
        try:
            # Read and filter CSV
            df = pd.read_csv(original_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df_filtered = df[df['Date'] <= cutoff_date].copy()
            
            # Companies were already pre-filtered in Step 2, so this should have sufficient data
            df_filtered['Date'] = df_filtered['Date'].dt.strftime('%Y-%m-%d')
            df_filtered.to_csv(filtered_file, index=False)
            processed_count += 1
        
        except Exception as e:
            print(f"  ❌ {company_id}: Error processing - {e}")
            continue
    
    print(f"✓ Processed {processed_count}/{len(company_ids)} companies")
    
    if processed_count == 0:
        print("❌ No companies successfully processed!")
        return None
    
    # Call data processing on filtered data
    print(f"📈 Processing filtered data with data.py...")
    
    try:
        from data_filtered import process_specific_companies
        
        # Get list of successfully processed company IDs
        filtered_files = list(filtered_ohlcv_dir.glob("*_ohlcv.csv"))
        filtered_company_ids = [f.stem.replace('_ohlcv', '') for f in filtered_files]
        
        print(f"  Processing {len(filtered_company_ids)} companies with filtered data...")
        
        # Calculate appropriate begin_date for this prediction
        # Use 3 years before prediction date to ensure enough data for moving averages
        prediction_date = datetime(prediction_year, prediction_month, 1)
        begin_date = prediction_date.replace(year=prediction_date.year - 3)
        begin_date_str = begin_date.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"  📅 Using begin_date: {begin_date_str} (3 years before prediction)")
        
        # Process data
        result = process_specific_companies(
            company_ids=filtered_company_ids,
            market_name=batch_name,
            ohlcv_dir=str(filtered_ohlcv_dir),
            begin_date_str=begin_date_str,
            min_data_points=30
        )
        
        pkl_path = Path(f"ml-model/data/{batch_name}_all_features.pkl")
        
        # Cleanup filtered directory
        shutil.rmtree(filtered_ohlcv_dir)
        
        if pkl_path.exists():
            file_size = pkl_path.stat().st_size / (1024**2)  # MB
            print(f"✅ Created {pkl_path}")
            print(f"  File size: {file_size:.1f} MB")
            print(f"  Companies in file: {len(result['all_features'])}")
            print(f"  🔒 Look-ahead bias prevented: No data after {cutoff_date.strftime('%Y-%m-%d')}")
            print()
            return str(pkl_path)
        else:
            print(f"❌ Failed to create {pkl_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if filtered_ohlcv_dir.exists():
            shutil.rmtree(filtered_ohlcv_dir)
        
        return None


# =============================================================================
# STEP 4: RUN ML INFERENCE
# =============================================================================

def step4_run_ml_inference(pkl_path: str, model_path: str, batch_name: str) -> Optional[Dict]:
    """
    Run ML inference on a data batch.
    
    Args:
        pkl_path: Path to the data pkl file
        model_path: Path to the pre-trained model
        batch_name: Name of the batch for logging
    
    Returns:
        Dict with ML predictions and scores, or None if failed
    """
    print(f"[STEP 4] Running ML Inference for {batch_name} Batch")
    print("-" * 80)
    
    if not os.path.exists(pkl_path):
        print(f"❌ PKL file not found: {pkl_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"🧠 Loading ML model: {model_path}")
    print(f"📊 Loading data: {pkl_path}")
    
    try:
        from simple_inference import StockInference
        
        # Initialize and load model
        inference = StockInference(model_path, device='cpu')
        inference.load_model()
        
        # Load data
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        
        all_features = data.get('all_features', {})
        print(f"  Loaded {len(all_features)} companies")
        
        if len(all_features) == 0:
            print("❌ No companies in pkl file!")
            return None
        
        # Run predictions
        print(f"🔮 Running ML predictions...")
        predictions = inference.predict(all_features, days=32)
        
        if not predictions:
            print("❌ No predictions generated!")
            return None
        
        print(f"✅ Generated ML predictions for {len(predictions)} companies")
        
        # Calculate statistics
        ml_scores = list(predictions.values())
        stats = {
            'count': len(ml_scores),
            'mean': np.mean(ml_scores),
            'std': np.std(ml_scores),
            'min': np.min(ml_scores),
            'max': np.max(ml_scores)
        }
        
        print(f"  ML Score Statistics:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Range: {stats['min']:.4f} to {stats['max']:.4f}")
        print()
        
        return {
            'predictions': predictions,
            'statistics': stats,
            'company_count': len(predictions)
        }
        
    except Exception as e:
        print(f"❌ Error during ML inference: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# STEP 5: COMBINE SCORES AND BUILD PORTFOLIO
# =============================================================================

def step5_build_portfolio(top_candidates: Dict, bottom_candidates: Dict, 
                         top_ml_results: Dict, bottom_ml_results: Dict,
                         algo_weight: float = 0.6, ml_weight: float = 0.4) -> Dict:
    """
    Combine algorithm and ML scores to create final portfolio weights.
    
    Args:
        top_candidates: Top candidates with algo scores
        bottom_candidates: Bottom candidates with algo scores
        top_ml_results: ML results for top candidates
        bottom_ml_results: ML results for bottom candidates
        algo_weight: Weight for algorithm scores (default: 0.6)
        ml_weight: Weight for ML scores (default: 0.4)
    
    Returns:
        Dict with final portfolio construction
    """
    print(f"[STEP 5] Building Final Portfolio (Algo: {algo_weight:.1f}, ML: {ml_weight:.1f})")
    print("-" * 80)
    
    portfolio = {
        'long_positions': [],
        'short_positions': [],
        'summary': {},
        'parameters': {
            'algo_weight': algo_weight,
            'ml_weight': ml_weight
        }
    }
    
    # Process LONG positions (top candidates)
    print("📈 Processing LONG positions...")
    
    # Normalize algorithm scores for top candidates
    top_algo_scores = np.array(top_candidates['algo_scores'])
    top_algo_normalized = (top_algo_scores - np.min(top_algo_scores)) / (np.max(top_algo_scores) - np.min(top_algo_scores))
    
    # Get ML scores for companies that have predictions
    top_ml_predictions = top_ml_results['predictions']
    
    for i, company_id in enumerate(top_candidates['company_ids']):
        if company_id in top_ml_predictions:
            algo_score = top_algo_normalized[i]
            ml_score = top_ml_predictions[company_id]
            
            # Normalize ML score to [0, 1] range for combination
            ml_normalized = (ml_score - top_ml_results['statistics']['min']) / \
                          (top_ml_results['statistics']['max'] - top_ml_results['statistics']['min'])
            
            # Combined score
            combined_score = algo_weight * algo_score + ml_weight * ml_normalized
            
            # Get company info
            company_info = top_candidates['df'][top_candidates['df']['company_id'] == company_id].iloc[0]
            
            portfolio['long_positions'].append({
                'company_id': company_id,
                'gvkey': int(float(company_info['gvkey'])),
                'iid': company_info['iid'],
                'algo_score_raw': top_candidates['algo_scores'][i],
                'algo_score_normalized': algo_score,
                'ml_score_raw': ml_score,
                'ml_score_normalized': ml_normalized,
                'combined_score': combined_score,
                'candidate_type': company_info['candidate_type']
            })
    
    # Sort long positions by combined score (descending)
    portfolio['long_positions'].sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Process SHORT positions (bottom candidates)
    print("📉 Processing SHORT positions...")
    
    # Normalize algorithm scores for bottom candidates (lower is better for shorts)
    bottom_algo_scores = np.array(bottom_candidates['algo_scores'])
    bottom_algo_normalized = (np.max(bottom_algo_scores) - bottom_algo_scores) / (np.max(bottom_algo_scores) - np.min(bottom_algo_scores))
    
    # Get ML scores for companies that have predictions
    bottom_ml_predictions = bottom_ml_results['predictions']
    
    for i, company_id in enumerate(bottom_candidates['company_ids']):
        if company_id in bottom_ml_predictions:
            algo_score = bottom_algo_normalized[i]
            ml_score = bottom_ml_predictions[company_id]
            
            # For shorts, lower ML scores are better, so invert normalization
            ml_normalized = (bottom_ml_results['statistics']['max'] - ml_score) / \
                          (bottom_ml_results['statistics']['max'] - bottom_ml_results['statistics']['min'])
            
            # Combined score
            combined_score = algo_weight * algo_score + ml_weight * ml_normalized
            
            # Get company info
            company_info = bottom_candidates['df'][bottom_candidates['df']['company_id'] == company_id].iloc[0]
            
            portfolio['short_positions'].append({
                'company_id': company_id,
                'gvkey': int(float(company_info['gvkey'])),
                'iid': company_info['iid'],
                'algo_score_raw': bottom_candidates['algo_scores'][i],
                'algo_score_normalized': algo_score,
                'ml_score_raw': ml_score,
                'ml_score_normalized': ml_normalized,
                'combined_score': combined_score,
                'candidate_type': company_info['candidate_type']
            })
    
    # Sort short positions by combined score (descending)
    portfolio['short_positions'].sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Calculate portfolio weights based on combined scores
    total_long = len(portfolio['long_positions'])
    total_short = len(portfolio['short_positions'])
    
    # Score-based weighting for LONG positions (sum to +1.0)
    if total_long > 0:
        long_scores = [p['combined_score'] for p in portfolio['long_positions']]
        total_long_score = sum(long_scores)
        
        if total_long_score > 0:
            for position in portfolio['long_positions']:
                # Weight proportional to combined score, normalized to sum to 1.0
                position['portfolio_weight'] = position['combined_score'] / total_long_score
        else:
            # Fallback to equal weights if all scores are zero
            long_weight_each = 1.0 / total_long
            for position in portfolio['long_positions']:
                position['portfolio_weight'] = long_weight_each
    
    # Score-based weighting for SHORT positions (sum to -1.0)
    if total_short > 0:
        short_scores = [p['combined_score'] for p in portfolio['short_positions']]
        total_short_score = sum(short_scores)
        
        if total_short_score > 0:
            for position in portfolio['short_positions']:
                # Weight proportional to combined score, normalized to sum to -1.0
                position['portfolio_weight'] = -position['combined_score'] / total_short_score
        else:
            # Fallback to equal weights if all scores are zero
            short_weight_each = -1.0 / total_short
            for position in portfolio['short_positions']:
                position['portfolio_weight'] = short_weight_each
    
    # Summary statistics
    portfolio['summary'] = {
        'total_long_positions': total_long,
        'total_short_positions': total_short,
        'total_positions': total_long + total_short,
        'long_combined_scores': [p['combined_score'] for p in portfolio['long_positions']],
        'short_combined_scores': [p['combined_score'] for p in portfolio['short_positions']],
        'long_weights_sum': sum([p['portfolio_weight'] for p in portfolio['long_positions']]) if total_long > 0 else 0,
        'short_weights_sum': sum([p['portfolio_weight'] for p in portfolio['short_positions']]) if total_short > 0 else 0,
        'creation_date': datetime.now().isoformat()
    }
    
    print(f"✅ Portfolio constructed:")
    print(f"  Long positions: {total_long}")
    print(f"  Short positions: {total_short}")
    print(f"  Total positions: {total_long + total_short}")
    
    if total_long > 0:
        long_scores = [p['combined_score'] for p in portfolio['long_positions']]
        long_weights = [p['portfolio_weight'] for p in portfolio['long_positions']]
        print(f"  Long combined scores: {np.mean(long_scores):.4f} ± {np.std(long_scores):.4f}")
        print(f"  Long weights sum: {sum(long_weights):.6f} (range: {min(long_weights):.4f} to {max(long_weights):.4f})")
    
    if total_short > 0:
        short_scores = [p['combined_score'] for p in portfolio['short_positions']]
        short_weights = [p['portfolio_weight'] for p in portfolio['short_positions']]
        print(f"  Short combined scores: {np.mean(short_scores):.4f} ± {np.std(short_scores):.4f}")
        print(f"  Short weights sum: {sum(short_weights):.6f} (range: {max(short_weights):.4f} to {min(short_weights):.4f})")
    
    print()
    
    return portfolio


# =============================================================================
# STEP 6: SAVE RESULTS
# =============================================================================

def step6_save_results(portfolio: Dict, output_path: str, sector: str, 
                      year: int, month: int) -> bool:
    """
    Save portfolio results to JSON file.
    
    Args:
        portfolio: Portfolio dictionary
        output_path: Path to save results
        sector: Sector name
        year: Target year
        month: Target month
    
    Returns:
        bool: True if successful
    """
    print(f"[STEP 6] Saving Results")
    print("-" * 80)
    
    # Add metadata
    portfolio['metadata'] = {
        'sector': sector,
        'target_year': year,
        'target_month': month,
        'pipeline_version': '1.0',
        'creation_timestamp': datetime.now().isoformat()
    }
    
    try:
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run_main_pipeline(sector: str, year: int, month: int, top_n: int, bottom_m: int,
                     model_path: str, output_path: str, algo_weight: float, ml_weight: float,
                     skip_algo: bool = False) -> bool:
    """
    Run the complete main pipeline.
    
    Args:
        sector: Sector to analyze
        year: Target year
        month: Target month
        top_n: Number of long positions
        bottom_m: Number of short positions
        model_path: Path to ML model
        output_path: Path to save results
        algo_weight: Weight for algorithm scores
        ml_weight: Weight for ML scores
        skip_algo: Skip algorithm step (assume already run)
    
    Returns:
        bool: True if successful
    """
    print(f"🚀 Starting Main Pipeline for {sector.upper()} {year}/{month:02d}")
    print(f"Target: {top_n} long + {bottom_m} short positions")
    print(f"Model: {model_path}")
    print()
    
    try:
        # Step 1: Run Algorithm (unless skipped)
        if not skip_algo:
            if not step1_run_algo(sector):
                return False
        else:
            print("[STEP 1] Skipping algorithm execution (--skip-algo)")
            print()
        
        # Step 2: Extract Candidates
        top_candidates, bottom_candidates = step2_extract_candidates(
            sector, year, month, top_n, bottom_m
        )
        
        if not top_candidates or not bottom_candidates:
            return False
        
        # Step 3A: Prepare Data for Top Batch
        top_pkl_path = step3_prepare_data_batch(
            top_candidates['company_ids'], 'TOP_LONG', year, month
        )
        
        if not top_pkl_path:
            return False
        
        # Step 3B: Prepare Data for Bottom Batch
        bottom_pkl_path = step3_prepare_data_batch(
            bottom_candidates['company_ids'], 'BOTTOM_SHORT', year, month
        )
        
        if not bottom_pkl_path:
            return False
        
        # Step 4A: ML Inference for Top Batch
        top_ml_results = step4_run_ml_inference(top_pkl_path, model_path, 'TOP_LONG')
        
        if not top_ml_results:
            return False
        
        # Step 4B: ML Inference for Bottom Batch
        bottom_ml_results = step4_run_ml_inference(bottom_pkl_path, model_path, 'BOTTOM_SHORT')
        
        if not bottom_ml_results:
            return False
        
        # Step 5: Build Portfolio
        portfolio = step5_build_portfolio(
            top_candidates, bottom_candidates,
            top_ml_results, bottom_ml_results,
            algo_weight, ml_weight
        )
        
        # Step 6: Save Results
        if not step6_save_results(portfolio, output_path, sector, year, month):
            return False
        
        # Final Summary
        print("=" * 100)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print(f"Sector: {sector.upper()}")
        print(f"Period: {year}/{month:02d}")
        print(f"Portfolio: {len(portfolio['long_positions'])} long + {len(portfolio['short_positions'])} short")
        print(f"Results: {output_path}")
        print("=" * 100)
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Main Quantitative Trading Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--sector', '-s', required=True,
                       choices=list(SECTOR_GICS_MAPPING.keys()),
                       help='Sector to analyze')
    
    parser.add_argument('--year', '-y', type=int, required=True,
                       help='Target year for predictions')
    
    parser.add_argument('--month', '-m', type=int, required=True,
                       choices=range(1, 13),
                       help='Target month for predictions')
    
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of long positions (default: 20)')
    
    parser.add_argument('--bottom-m', type=int, default=15,
                       help='Number of short positions (default: 15)')
    
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                       help=f'Path to ML model (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--output', '-o',
                       help='Output path for results (default: auto-generated)')
    
    parser.add_argument('--algo-weight', type=float, default=0.6,
                       help='Weight for algorithm scores (default: 0.6)')
    
    parser.add_argument('--ml-weight', type=float, default=0.4,
                       help='Weight for ML scores (default: 0.4)')
    
    parser.add_argument('--skip-algo', action='store_true',
                       help='Skip algorithm execution (assume already run)')
    
    parser.add_argument('--list-sectors', action='store_true',
                       help='List available sectors and exit')
    
    args = parser.parse_args()
    
    # List sectors
    if args.list_sectors:
        print("Available sectors:")
        for sector, gics in SECTOR_GICS_MAPPING.items():
            print(f"  {sector:20} (GICS: {gics})")
        return
    
    # Validate weights
    if abs(args.algo_weight + args.ml_weight - 1.0) > 0.01:
        print("❌ Algorithm and ML weights must sum to 1.0")
        sys.exit(1)
    
    # Auto-generate output path if not provided
    if not args.output:
        args.output = f"results/portfolio_{args.year:04d}_{args.month:02d}_{args.sector}.json"
    
    # Run pipeline
    success = run_main_pipeline(
        sector=args.sector,
        year=args.year,
        month=args.month,
        top_n=args.top_n,
        bottom_m=args.bottom_m,
        model_path=args.model,
        output_path=args.output,
        algo_weight=args.algo_weight,
        ml_weight=args.ml_weight,
        skip_algo=args.skip_algo
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()