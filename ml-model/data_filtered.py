"""
Filtered version of data.py processing that only processes specific company IDs.

This is used for end-to-end testing with small datasets to avoid memory issues.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime
from pathlib import Path

def process_specific_companies(company_ids, market_name='TEST', ohlcv_dir='../inference/company_ohlcv_data',
                                begin_date_str='2012-11-19 00:00:00', min_data_points=30, data_output_dir=None):
    """
    Process ONLY specific company IDs to create a small .pkl file.
    
    This is identical to process_company_ohlcv_data() from data.py, but filters
    to only process the provided company_ids list.
    
    Args:
        company_ids: List of company IDs to process (e.g., ['comp_001004_01', ...])
        market_name: Name for output files
        ohlcv_dir: Directory containing OHLCV CSV files
        begin_date_str: Start date for filtering
        min_data_points: Minimum data points required
        data_output_dir: Directory to save output files (default: <script_dir>/data)
    
    Returns:
        dict: save_data with all_features, index_tra_dates, tra_dates_index
    """
    
    # Determine output directory
    if data_output_dir is None:
        # Use ml-model/data directory relative to this script
        script_dir = Path(__file__).parent
        data_path = script_dir / 'data'
    else:
        data_path = Path(data_output_dir)
    
    data_path.mkdir(parents=True, exist_ok=True)
    data_path = str(data_path)  # Convert back to string for compatibility
    
    date_format = '%Y-%m-%d %H:%M:%S'
    begin_date = datetime.strptime(begin_date_str, date_format)
    return_days = 1
    pad_begin = 29
    
    print(f"=" * 80)
    print(f"Processing {len(company_ids)} specific companies")
    print(f"=" * 80)
    
    # Step 1: Build list of valid files for these companies
    print("\nStep 1: Discovering OHLCV files for specified companies...")
    
    valid_files = []
    valid_company_ids = []
    missing_companies = []
    
    for company_id in company_ids:
        csv_file = os.path.join(ohlcv_dir, f'{company_id}_ohlcv.csv')
        if os.path.exists(csv_file):
            try:
                # Quick validation
                df = pd.read_csv(csv_file)
                if len(df) >= min_data_points:
                    valid_files.append(csv_file)
                    valid_company_ids.append(company_id)
            except Exception as e:
                print(f"  Warning: Could not read {company_id}: {e}")
        else:
            missing_companies.append(company_id)
    
    print(f"  Requested: {len(company_ids)}")
    print(f"  Found with OHLCV: {len(valid_company_ids)}")
    if missing_companies:
        print(f"  Missing OHLCV files: {len(missing_companies)}")
        if len(missing_companies) <= 10:
            print(f"    {missing_companies}")
    
    if len(valid_company_ids) == 0:
        raise ValueError("No valid OHLCV files found for any requested company!")
    
    # Step 2: Build trading dates from these files
    print("\nStep 2: Building trading dates list...")
    all_dates_set = set()
    
    for csv_file in valid_files:
        try:
            df = pd.read_csv(csv_file)
            for date_val in df['Date'].values:
                if ' ' in str(date_val):
                    date_str = str(date_val).split(' ')[0] + ' 00:00:00'
                else:
                    date_str = str(date_val) + ' 00:00:00'
                all_dates_set.add(date_str)
        except Exception as e:
            continue
    
    trading_dates = np.array(sorted(list(all_dates_set)))
    print(f"  Found {len(trading_dates)} unique trading dates")
    
    # Create date mappings
    index_tra_dates = {}
    tra_dates_index = {}
    for index, date in enumerate(trading_dates):
        tra_dates_index[date] = index
        index_tra_dates[index] = date
    
    # Step 3: Process each company (IDENTICAL LOGIC TO data.py)
    print("\nStep 3: Computing features...")
    all_features = {}
    processed_count = 0
    skipped_count = 0
    error_reasons = {'insufficient_data': 0, 'date_filter': 0, 'normalization': 0, 'other': 0}
    
    for stock_index, (csv_file, company_id) in enumerate(zip(valid_files, valid_company_ids)):
        try:
            df = pd.read_csv(csv_file)
            
            # Convert to numpy array format
            single_EOD_str = []
            for _, row in df.iterrows():
                date_val = str(row['Date'])
                if ' ' in date_val:
                    date_str = date_val.split(' ')[0] + ' 00:00:00'
                else:
                    date_str = date_val + ' 00:00:00'
                
                single_EOD_str.append([
                    date_str,
                    str(row['Open']),
                    str(row['High']),
                    str(row['Low']),
                    str(row['Close']),
                    str(row['Volume'])
                ])
            
            single_EOD_str = np.array(single_EOD_str)
            
            # === BEGIN: IDENTICAL LOGIC TO data.py process_eod_data() ===
            
            # select data within the begin_date
            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD_str):
                date_str = daily_EOD[0]
                cur_date = datetime.strptime(date_str, date_format)
                if cur_date >= begin_date:
                    begin_date_row = date_index
                    break
            
            if begin_date_row == -1:
                skipped_count += 1
                error_reasons['date_filter'] += 1
                continue
            
            selected_EOD_str = single_EOD_str[begin_date_row:]
            
            if len(selected_EOD_str) < min_data_points:
                skipped_count += 1
                error_reasons['insufficient_data'] += 1
                continue
            
            selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
            
            for row, daily_EOD in enumerate(selected_EOD_str):
                date_str = daily_EOD[0]
                if date_str not in tra_dates_index:
                    continue
                selected_EOD[row][0] = tra_dates_index[date_str]
                for col in range(1, selected_EOD_str.shape[1]):
                    selected_EOD[row][col] = float(daily_EOD[col])
            
            # calculate moving average features
            begin_date_row = -1
            for row_idx in range(len(selected_EOD)):
                date_index_val = int(selected_EOD[row_idx][0])
                if date_index_val >= pad_begin:
                    begin_date_row = row_idx
                    break
            
            if begin_date_row == -1:
                skipped_count += 1
                error_reasons['insufficient_data'] += 1
                continue
            
            mov_aver_features = np.zeros([selected_EOD.shape[0], 4 * 5], dtype=float)
            
            for row in range(begin_date_row, selected_EOD.shape[0]):
                date_index = selected_EOD[row][0]
                aver_5, aver_10, aver_20, aver_30 = [0.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5
                count_5, count_10, count_20, count_30 = 0, 0, 0, 0
                
                for offset in range(30):
                    if row - offset < 0:
                        break
                    date_gap = date_index - selected_EOD[row - offset][0]
                    if date_gap < 5:
                        count_5 += 1
                        for price_index in range(1, 6):
                            aver_5[price_index-1] += selected_EOD[row - offset][price_index]
                    if date_gap < 10:
                        count_10 += 1
                        for price_index in range(1, 6):
                            aver_10[price_index-1] += selected_EOD[row - offset][price_index]
                    if date_gap < 20:
                        count_20 += 1
                        for price_index in range(1, 6):
                            aver_20[price_index-1] += selected_EOD[row - offset][price_index]
                    if date_gap < 30:
                        count_30 += 1
                        for price_index in range(1, 6):
                            aver_30[price_index-1] += selected_EOD[row - offset][price_index]
                
                for price_index in range(5):
                    if count_5 > 0:
                        mov_aver_features[row][4 * price_index + 0] = aver_5[price_index] / count_5
                    if count_10 > 0:
                        mov_aver_features[row][4 * price_index + 1] = aver_10[price_index] / count_10
                    if count_20 > 0:
                        mov_aver_features[row][4 * price_index + 2] = aver_20[price_index] / count_20
                    if count_30 > 0:
                        mov_aver_features[row][4 * price_index + 3] = aver_30[price_index] / count_30
            
            # normalize features
            normalization_data = selected_EOD[begin_date_row:, 1:6]
            if len(normalization_data) == 0:
                skipped_count += 1
                error_reasons['normalization'] += 1
                continue
            
            price_min = np.min(normalization_data, 0)
            price_max = np.max(normalization_data, 0)
            for price_index in range(5):
                if price_max[price_index] > 0:
                    mov_aver_features[:, 4 * price_index: 4 * price_index + 4] = (
                        mov_aver_features[:, 4 * price_index: 4 * price_index + 4] / price_max[price_index])
            
            # build final features array
            features = np.ones([len(trading_dates) - pad_begin, 1+5*5], dtype=float) * -1234
            for row in range(len(trading_dates) - pad_begin):
                features[row][0] = row
            
            for row in range(begin_date_row, selected_EOD.shape[0]):
                cur_index = int(selected_EOD[row][0])
                if cur_index < pad_begin:
                    continue
                for price_index in range(5):
                    features[cur_index - pad_begin][1+5*price_index: 1+5*price_index+4] \
                        = mov_aver_features[row][4*price_index: 4*price_index+4]
                    if row >= return_days:
                        if cur_index - int(selected_EOD[row - return_days][0]) == return_days:
                            if price_max[price_index] > 0:
                                features[cur_index - pad_begin][1+5*price_index+4] \
                                    = selected_EOD[row][1+price_index] / price_max[price_index]
            
            # Store with company ID as key
            all_features[company_id] = features
            processed_count += 1
            
            # === END: IDENTICAL LOGIC ===
            
            if processed_count % 10 == 0:
                print(f"  Processed: {processed_count} / {len(valid_company_ids)}")
        
        except Exception as e:
            skipped_count += 1
            error_reasons['other'] += 1
            if skipped_count <= 5:
                print(f"  Error processing {company_id}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Requested companies:            {len(company_ids):,}")
    print(f"Found with OHLCV:               {len(valid_company_ids):,}")
    print(f"Successfully processed:         {processed_count:,}")
    print(f"Skipped:                        {skipped_count:,}")
    print(f"\nSkip reasons:")
    print(f"  Insufficient data:            {error_reasons['insufficient_data']:,}")
    print(f"  Failed date filter:           {error_reasons['date_filter']:,}")
    print(f"  Normalization issues:         {error_reasons['normalization']:,}")
    print(f"  Other errors:                 {error_reasons['other']:,}")
    print("="*80)
    
    # Save outputs
    print(f"\nSaving {len(all_features)} companies to pickle file...")
    save_data = {'all_features': all_features, 'index_tra_dates': index_tra_dates, 'tra_dates_index': tra_dates_index}
    
    pkl_path = os.path.join(data_path, market_name + '_all_features.pkl')
    try:
        with open(pkl_path, 'wb') as fw:
            pkl.dump(save_data, fw, protocol=4)
        print(f"✅ Successfully saved: {pkl_path}")
        
        # Get file size
        file_size = os.path.getsize(pkl_path) / (1024**2)
        print(f"   File size: {file_size:.1f} MB")
    except Exception as e:
        print(f"❌ Error saving pickle file: {e}")
        raise
    
    # Save trading dates
    with open(os.path.join(data_path, market_name + '_aver_line_dates.csv'), 'w') as f:
        for date_str in trading_dates:
            f.write(f"{date_str}\n")
    
    # Save company IDs list
    with open(os.path.join(data_path, market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'), 'w') as f:
        for company_id in sorted(all_features.keys()):
            f.write(f"{company_id}\n")
    
    print(f"✅ All outputs saved to {data_path}/")
    print()
    
    return save_data


if __name__ == "__main__":
    # Example usage
    test_companies = [
        'comp_001004_01',
        'comp_001045_01',
        'comp_001078_01'
    ]
    
    result = process_specific_companies(
        company_ids=test_companies,
        market_name='TEST_FILTERED',
        ohlcv_dir='../inference/company_ohlcv_data'
    )
    
    print(f"Processed {len(result['all_features'])} companies")
