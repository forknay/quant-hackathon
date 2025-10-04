#!/usr/bin/env python3
"""
Debug script to check OHLCV data availability
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def check_ohlcv_coverage():
    """Check date coverage of OHLCV files"""
    
    ohlcv_dir = Path("inference/company_ohlcv_data")
    cutoff_date = datetime(2017, 12, 31)
    
    print("=" * 60)
    print("OHLCV DATA COVERAGE ANALYSIS")
    print("=" * 60)
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    print()
    
    # Check first 10 files
    files = list(ohlcv_dir.glob("*.csv"))[:10]
    
    for file in files:
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            
            # Check data before cutoff
            df_before_cutoff = df[df['Date'] <= cutoff_date]
            points_before_cutoff = len(df_before_cutoff)
            
            print(f"File: {file.name}")
            print(f"  Total rows: {len(df)}")
            print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            print(f"  Points before cutoff: {points_before_cutoff}")
            print(f"  Sufficient data (≥30): {'✅' if points_before_cutoff >= 30 else '❌'}")
            print()
            
        except Exception as e:
            print(f"File: {file.name} - Error: {e}")
            print()

if __name__ == "__main__":
    check_ohlcv_coverage()