#!/usr/bin/env python3
"""
Debug script to check candidate OHLCV availability
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def check_candidate_ohlcv():
    """Check OHLCV availability for selected candidates"""
    
    candidates_dir = Path("algo/results/it_parquet/candidates/year=2018/month=1")
    ohlcv_dir = Path("inference/company_ohlcv_data")
    cutoff_date = datetime(2017, 12, 31)
    
    print("=" * 60)
    print("CANDIDATE OHLCV AVAILABILITY CHECK")
    print("=" * 60)
    print(f"Candidates dir: {candidates_dir}")
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    print()
    
    if not candidates_dir.exists():
        print(f"❌ Candidates directory not found: {candidates_dir}")
        return
    
    # Read candidate files
    parquet_files = list(candidates_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ No candidate files found")
        return
    
    print(f"✓ Found {len(parquet_files)} candidate files")
    
    # Load candidates
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
    
    print(f"✓ Total candidates: {len(df_candidates)}")
    
    # Check first 20 candidates
    candidates_to_check = df_candidates.head(20)
    
    valid_count = 0
    
    for _, row in candidates_to_check.iterrows():
        company_id = row['company_id']
        ohlcv_file = ohlcv_dir / f"{company_id}_ohlcv.csv"
        
        print(f"\nCandidate: {company_id}")
        print(f"  Score: {row['composite_score']:.4f}")
        print(f"  Type: {row['candidate_type']}")
        
        if ohlcv_file.exists():
            try:
                df_ohlcv = pd.read_csv(ohlcv_file)
                df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'])
                df_filtered = df_ohlcv[df_ohlcv['Date'] <= cutoff_date]
                
                min_date = df_ohlcv['Date'].min()
                max_date = df_ohlcv['Date'].max()
                points_before = len(df_filtered)
                
                print(f"  ✅ OHLCV file exists")
                print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                print(f"  Points before cutoff: {points_before}")
                print(f"  Sufficient: {'✅' if points_before >= 30 else '❌'}")
                
                if points_before >= 30:
                    valid_count += 1
                    
            except Exception as e:
                print(f"  ❌ Error reading OHLCV: {e}")
        else:
            print(f"  ❌ No OHLCV file")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {valid_count}/{len(candidates_to_check)} candidates have sufficient OHLCV data")
    print(f"{'='*60}")

if __name__ == "__main__":
    check_candidate_ohlcv()