#!/usr/bin/env python3
"""
Debug the detailed data processing to see where 2015/01 companies are being filtered out
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def debug_data_processing():
    """Debug why companies are being filtered out in detailed processing"""
    
    print("🔍 DEBUGGING DETAILED DATA PROCESSING FOR 2015/01")
    print("=" * 70)
    
    # Test companies that should be valid
    test_companies = ['comp_235762_01W', 'comp_295397_01W', 'comp_287449_01W']
    ohlcv_dir = 'temp/filtered_ohlcv_TOP_LONG'  # This would be created by the pipeline
    
    # Since that directory won't exist, let's use the original directory
    # and manually apply the date filtering
    original_ohlcv_dir = Path("inference/company_ohlcv_data")
    cutoff_date = datetime(2014, 12, 31)
    
    date_format = '%Y-%m-%d %H:%M:%S'
    begin_date_str = '2012-11-19 00:00:00'
    begin_date = datetime.strptime(begin_date_str, date_format)
    min_data_points = 30
    pad_begin = 29
    
    print(f"📅 Processing parameters:")
    print(f"   Begin date filter: {begin_date_str}")
    print(f"   Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"   Min data points: {min_data_points}")
    print(f"   Pad begin: {pad_begin}")
    print()
    
    for company_id in test_companies:
        print(f"🔬 ANALYZING {company_id}")
        print("-" * 50)
        
        # Load original data
        ohlcv_file = original_ohlcv_dir / f"{company_id}_ohlcv.csv"
        if not ohlcv_file.exists():
            print(f"   ❌ File not found: {ohlcv_file}")
            continue
        
        try:
            df = pd.read_csv(ohlcv_file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            print(f"   📊 Original data: {len(df)} records")
            print(f"      Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Apply cutoff filter (pipeline step)
            df_filtered = df[df['Date'] <= cutoff_date].copy()
            print(f"   📊 After cutoff filter: {len(df_filtered)} records")
            print(f"      Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
            
            # Convert to processing format
            single_EOD_str = []
            for _, row in df_filtered.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d') + ' 00:00:00'
                single_EOD_str.append([
                    date_str,
                    str(row['Open']),
                    str(row['High']),
                    str(row['Low']),
                    str(row['Close']),
                    str(row['Volume'])
                ])
            
            single_EOD_str = np.array(single_EOD_str)
            
            # Apply begin_date filter (from data processing)
            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD_str):
                date_str = daily_EOD[0]
                cur_date = datetime.strptime(date_str, date_format)
                if cur_date >= begin_date:
                    begin_date_row = date_index
                    break
            
            if begin_date_row == -1:
                print(f"   ❌ FILTERED OUT: No data after begin_date ({begin_date_str})")
                continue
            
            selected_EOD_str = single_EOD_str[begin_date_row:]
            print(f"   📊 After begin_date filter: {len(selected_EOD_str)} records")
            print(f"      Begin date row: {begin_date_row}")
            
            if len(selected_EOD_str) < min_data_points:
                print(f"   ❌ FILTERED OUT: Insufficient data ({len(selected_EOD_str)} < {min_data_points})")
                continue
            
            print(f"   ✅ Passed min_data_points check")
            
            # Check trading dates requirement
            # For this we need all trading dates, let's simulate
            all_dates_set = set()
            for daily_EOD in single_EOD_str:
                all_dates_set.add(daily_EOD[0])
            trading_dates = np.array(sorted(list(all_dates_set)))
            
            print(f"   📊 Unique trading dates: {len(trading_dates)}")
            
            # Create date mappings
            tra_dates_index = {}
            for index, date in enumerate(trading_dates):
                tra_dates_index[date] = index
            
            # Convert to numeric format
            selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
            valid_rows = 0
            
            for row, daily_EOD in enumerate(selected_EOD_str):
                date_str = daily_EOD[0]
                if date_str in tra_dates_index:
                    selected_EOD[row][0] = tra_dates_index[date_str]
                    for col in range(1, selected_EOD_str.shape[1]):
                        selected_EOD[row][col] = float(daily_EOD[col])
                    valid_rows += 1
            
            print(f"   📊 Valid rows after date mapping: {valid_rows}")
            
            # Check moving average requirement
            begin_date_row_ma = -1
            for row_idx in range(len(selected_EOD)):
                date_index_val = int(selected_EOD[row_idx][0])
                if date_index_val >= pad_begin:
                    begin_date_row_ma = row_idx
                    break
            
            if begin_date_row_ma == -1:
                print(f"   ❌ FILTERED OUT: No data with date_index >= {pad_begin}")
                print(f"      Max date_index: {int(selected_EOD[:, 0].max()) if len(selected_EOD) > 0 else 'N/A'}")
                continue
            
            # Check normalization data
            normalization_data = selected_EOD[begin_date_row_ma:, 1:6]
            if len(normalization_data) == 0:
                print(f"   ❌ FILTERED OUT: No normalization data")
                continue
            
            final_features_length = len(trading_dates) - pad_begin
            print(f"   📊 Final features array length: {final_features_length}")
            
            if final_features_length <= 0:
                print(f"   ❌ FILTERED OUT: No room for features (trading_dates={len(trading_dates)}, pad_begin={pad_begin})")
                continue
            
            print(f"   ✅ SHOULD PASS all checks!")
            print(f"      Records for moving averages: {len(selected_EOD) - begin_date_row_ma}")
            print(f"      Normalization data shape: {normalization_data.shape}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    debug_data_processing()