#!/usr/bin/env python3
"""
Debug script to check why 2015/01 has no valid candidates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def debug_2015_candidates():
    """Debug the 2015/01 candidate filtering issue"""
    
    print("🔍 DEBUGGING 2015/01 CANDIDATE FILTERING")
    print("=" * 60)
    
    # Load candidates for 2015/01
    candidates_dir = Path("algo/results/it_parquet/candidates/year=2015/month=1")
    parquet_files = list(candidates_dir.glob("*.parquet"))
    
    print(f"📁 Found {len(parquet_files)} candidate files")
    
    # Load candidates
    all_candidates = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        all_candidates.append(df)
    
    df_candidates = pd.concat(all_candidates, ignore_index=True)
    
    # Construct company IDs
    df_candidates['company_id'] = ('comp_' + 
                                   df_candidates['gvkey'].astype(float).astype(int).astype(str).str.zfill(6) + 
                                   '_' + 
                                   df_candidates['iid'].astype(str))
    
    print(f"📊 Total candidates: {len(df_candidates)}")
    print(f"   Candidate types: {df_candidates['candidate_type'].value_counts().to_dict()}")
    print(f"   Sample company IDs: {df_candidates['company_id'].head(3).tolist()}")
    print()
    
    # Calculate cutoff date for 2015/01 prediction
    prediction_date = datetime(2015, 1, 1)
    cutoff_date = prediction_date - timedelta(days=1)  # 2014-12-31
    
    print(f"🗓️  Prediction date: {prediction_date.strftime('%Y-%m-%d')}")
    print(f"🗓️  Data cutoff: {cutoff_date.strftime('%Y-%m-%d')}")
    print()
    
    # Check OHLCV data availability
    original_ohlcv_dir = Path("inference/company_ohlcv_data")
    print(f"📂 OHLCV directory: {original_ohlcv_dir}")
    print(f"   Directory exists: {original_ohlcv_dir.exists()}")
    
    if original_ohlcv_dir.exists():
        ohlcv_files = list(original_ohlcv_dir.glob("*.csv"))
        print(f"   Total OHLCV files: {len(ohlcv_files)}")
        print(f"   Sample OHLCV files: {[f.name for f in ohlcv_files[:3]]}")
    print()
    
    # Check sample companies for data availability
    print("🔬 DETAILED ANALYSIS OF FIRST 10 COMPANIES:")
    print("-" * 60)
    
    valid_count = 0
    total_checked = 0
    
    for i, (_, row) in enumerate(df_candidates.head(10).iterrows()):
        company_id = row['company_id']
        ohlcv_file = original_ohlcv_dir / f"{company_id}_ohlcv.csv"
        
        total_checked += 1
        
        print(f"{i+1:2d}. {company_id}")
        print(f"    File exists: {ohlcv_file.exists()}")
        
        if ohlcv_file.exists():
            try:
                # Load and analyze the data
                df_ohlcv = pd.read_csv(ohlcv_file)
                df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'])
                
                print(f"    Total records: {len(df_ohlcv)}")
                print(f"    Date range: {df_ohlcv['Date'].min()} to {df_ohlcv['Date'].max()}")
                
                # Filter for cutoff date
                df_filtered = df_ohlcv[df_ohlcv['Date'] <= cutoff_date]
                print(f"    Records before cutoff ({cutoff_date.strftime('%Y-%m-%d')}): {len(df_filtered)}")
                
                if len(df_filtered) >= 30:
                    print(f"    ✅ VALID (≥30 records)")
                    valid_count += 1
                else:
                    print(f"    ❌ INSUFFICIENT ({len(df_filtered)} < 30)")
                    
            except Exception as e:
                print(f"    ❌ ERROR: {e}")
        else:
            print(f"    ❌ FILE NOT FOUND")
        
        print()
    
    print("=" * 60)
    print(f"📈 SUMMARY FOR FIRST 10 COMPANIES:")
    print(f"   Total checked: {total_checked}")
    print(f"   Valid companies: {valid_count}")
    print(f"   Success rate: {valid_count/total_checked*100:.1f}%")
    print()
    
    # Now check all companies (but just count, don't print details)
    print("🔍 CHECKING ALL COMPANIES...")
    
    all_valid_count = 0
    all_total_count = 0
    no_file_count = 0
    insufficient_data_count = 0
    error_count = 0
    
    for _, row in df_candidates.iterrows():
        company_id = row['company_id']
        ohlcv_file = original_ohlcv_dir / f"{company_id}_ohlcv.csv"
        
        all_total_count += 1
        
        if not ohlcv_file.exists():
            no_file_count += 1
            continue
            
        try:
            df_ohlcv = pd.read_csv(ohlcv_file)
            df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'])
            df_filtered = df_ohlcv[df_ohlcv['Date'] <= cutoff_date]
            
            if len(df_filtered) >= 30:
                all_valid_count += 1
            else:
                insufficient_data_count += 1
                
        except Exception as e:
            error_count += 1
            
    print("=" * 60)
    print(f"📊 FINAL RESULTS FOR ALL {all_total_count} COMPANIES:")
    print(f"   ✅ Valid companies: {all_valid_count}")
    print(f"   ❌ No OHLCV file: {no_file_count}")
    print(f"   ❌ Insufficient data: {insufficient_data_count}")
    print(f"   ❌ Data errors: {error_count}")
    print(f"   📈 Success rate: {all_valid_count/all_total_count*100:.1f}%")
    print()
    
    if all_valid_count == 0:
        print("🚨 PROBLEM IDENTIFIED: NO VALID COMPANIES!")
        print("   This explains why the pipeline fails for 2015/01")
        print("   The data cutoff (2014-12-31) is too early for available OHLCV data")
        
        # Check what's the earliest date in OHLCV files
        print("\n🔍 CHECKING EARLIEST AVAILABLE DATES...")
        sample_files = list(original_ohlcv_dir.glob("*.csv"))[:5]
        
        for ohlcv_file in sample_files:
            try:
                df = pd.read_csv(ohlcv_file)
                df['Date'] = pd.to_datetime(df['Date'])
                earliest = df['Date'].min()
                latest = df['Date'].max()
                print(f"   {ohlcv_file.name}: {earliest} to {latest}")
            except:
                continue
    else:
        print("✅ VALIDATION: Some companies should be valid")
        print("   The pipeline should work - need to investigate further")

if __name__ == "__main__":
    debug_2015_candidates()