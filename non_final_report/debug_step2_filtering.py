#!/usr/bin/env python3
"""
Debug the new Step 2 filtering to find appropriate thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def debug_step2_filtering():
    """Debug the Step 2 filtering requirements"""
    
    print("🔍 DEBUGGING STEP 2 FILTERING REQUIREMENTS")
    print("=" * 60)
    
    # Load 2015/01 candidates
    candidates_dir = Path("algo/results/it_parquet/candidates/year=2015/month=1")
    parquet_files = list(candidates_dir.glob("*.parquet"))
    
    all_candidates = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        all_candidates.append(df)
    
    df_candidates = pd.concat(all_candidates, ignore_index=True)
    df_candidates['company_id'] = ('comp_' + 
                                   df_candidates['gvkey'].astype(float).astype(int).astype(str).str.zfill(6) + 
                                   '_' + 
                                   df_candidates['iid'].astype(str))
    
    print(f"📊 Total candidates: {len(df_candidates)}")
    
    # Parameters
    prediction_date = datetime(2015, 1, 1)
    cutoff_date = prediction_date - timedelta(days=1)  # 2014-12-31
    begin_date = prediction_date.replace(year=prediction_date.year - 3)  # 2012-01-01
    
    print(f"📅 Date window: {begin_date.strftime('%Y-%m-%d')} to {cutoff_date.strftime('%Y-%m-%d')}")
    
    original_ohlcv_dir = Path("inference/company_ohlcv_data")
    
    # Test different thresholds
    thresholds = [30, 40, 50, 60, 70, 80]
    
    for threshold in thresholds:
        valid_count = 0
        total_checked = 0
        data_lengths = []
        
        for _, row in df_candidates.head(100).iterrows():  # Sample first 100
            company_id = row['company_id']
            ohlcv_file = original_ohlcv_dir / f"{company_id}_ohlcv.csv"
            
            if ohlcv_file.exists():
                try:
                    df_ohlcv = pd.read_csv(ohlcv_file)
                    df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'])
                    
                    # Apply both filters
                    df_filtered = df_ohlcv[
                        (df_ohlcv['Date'] >= begin_date) & 
                        (df_ohlcv['Date'] <= cutoff_date)
                    ]
                    
                    data_lengths.append(len(df_filtered))
                    total_checked += 1
                    
                    if len(df_filtered) >= threshold:
                        valid_count += 1
                        
                except Exception as e:
                    continue
        
        if data_lengths:
            print(f"Threshold {threshold:2d}: {valid_count:3d}/{total_checked:3d} valid ({valid_count/total_checked*100:5.1f}%), "
                  f"avg length: {np.mean(data_lengths):5.1f}, "
                  f"min: {np.min(data_lengths):3.0f}, max: {np.max(data_lengths):3.0f}")
    
    # Show distribution
    if data_lengths:
        print(f"\n📊 DATA LENGTH DISTRIBUTION (first 100 companies):")
        print(f"   Mean: {np.mean(data_lengths):.1f}")
        print(f"   Median: {np.median(data_lengths):.1f}")
        print(f"   Std: {np.std(data_lengths):.1f}")
        print(f"   Min: {np.min(data_lengths)}")
        print(f"   Max: {np.max(data_lengths)}")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            value = np.percentile(data_lengths, p)
            print(f"   {p:2d}th percentile: {value:.1f}")

if __name__ == "__main__":
    debug_step2_filtering()