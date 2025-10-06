"""
Transform cleaned_all.parquet data into the format expected by the ML model.

This script converts the panel data format to the time series format with moving averages
that the TransformerStockPrediction model expects.
"""

import pandas as pd
import numpy as np
import pickle as pkl
import os
from datetime import datetime
import argparse
from pathlib import Path


def create_basic_price_features(df_stock):
    """
    Create realistic OHLCV features from cleaned parquet data using hybrid approach.
    
    Based on analysis:
    - 'prc' contains normalized/log price data (range -4 to +8)
    - 'stock_ret' contains actual returns (range -8 to +8, mean ~0.13)
    - 'turnover_126d' can be used as volume proxy
    """
    
    # Primary approach: Use 'prc' as base price and 'stock_ret' for volatility
    if 'prc' in df_stock.columns and 'stock_ret' in df_stock.columns:
        # Get the data
        log_prices = df_stock['prc'].values  # These are log-normalized prices
        returns = df_stock['stock_ret'].values
        
        # Convert log prices to actual price levels (assuming they're log-normalized)
        # Scale to reasonable stock price range ($10-$200)
        base_price = 50.0  # Starting price level
        close_prices = base_price * np.exp(log_prices * 0.5)  # Scale factor to get reasonable prices
        
        # Create realistic OHLC using returns for intraday volatility
        num_days = len(close_prices)
        open_prices = np.zeros_like(close_prices)
        high_prices = np.zeros_like(close_prices)
        low_prices = np.zeros_like(close_prices)
        
        for i in range(num_days):
            current_close = close_prices[i]
            current_return = returns[i] if not np.isnan(returns[i]) else 0.0
            
            # Generate Open (previous close + overnight gap)
            if i == 0:
                open_prices[i] = current_close * (1 + np.random.normal(0, 0.002))  # Small random gap
            else:
                # Use a portion of the return as overnight gap
                overnight_gap = current_return * 0.3  # 30% of return happens overnight
                open_prices[i] = close_prices[i-1] * (1 + overnight_gap)
            
            # Generate High and Low using intraday volatility
            intraday_return = current_return * 0.7  # 70% of return is intraday
            intraday_vol = abs(intraday_return) * 0.5  # Intraday volatility
            
            # Ensure realistic OHLC relationships
            if intraday_return >= 0:  # Up day
                high_prices[i] = max(open_prices[i], current_close) * (1 + intraday_vol)
                low_prices[i] = min(open_prices[i], current_close) * (1 - intraday_vol * 0.5)
            else:  # Down day
                high_prices[i] = max(open_prices[i], current_close) * (1 + intraday_vol * 0.5)
                low_prices[i] = min(open_prices[i], current_close) * (1 - intraday_vol)
            
            # Ensure OHLC constraints: Low <= Open,Close <= High
            low_prices[i] = min(low_prices[i], open_prices[i], current_close)
            high_prices[i] = max(high_prices[i], open_prices[i], current_close)
        
        # Generate Volume using turnover data
        if 'turnover_126d' in df_stock.columns:
            # turnover_126d is normalized, convert to realistic volume
            turnover_norm = df_stock['turnover_126d'].values
            # Scale to realistic volume range (10K to 10M shares)
            base_volume = 500000  # 500K shares average
            volumes = base_volume * np.exp(turnover_norm * 0.5)
            volumes = np.clip(volumes, 10000, 10000000)  # Reasonable bounds
            
            # Higher volume on days with larger returns
            volume_multiplier = 1 + np.abs(returns) * 0.2
            volumes = volumes * volume_multiplier
        else:
            # Synthetic volume correlated with returns
            base_volume = 500000
            volume_multiplier = 1 + np.abs(returns) * 0.3
            volumes = base_volume * volume_multiplier
        
        volumes = volumes.astype(int)
        
    else:
        # Fallback: basic synthetic data
        num_days = len(df_stock)
        base_price = 50.0
        close_prices = np.full(num_days, base_price)
        open_prices = close_prices * 1.001
        high_prices = close_prices * 1.01
        low_prices = close_prices * 0.99
        volumes = np.full(num_days, 1000000, dtype=int)
    
    return np.column_stack([open_prices, high_prices, low_prices, close_prices, volumes])


def calculate_moving_averages(ohlcv_data, window_sizes=[5, 10, 20, 30]):
    """
    Calculate moving averages for OHLCV data.
    
    Args:
        ohlcv_data: [num_days, 5] array with Open, High, Low, Close, Volume
        window_sizes: List of window sizes for moving averages
    
    Returns:
        features: [num_days, 25] array with moving averages and original values
    """
    num_days, num_features = ohlcv_data.shape
    features = np.full((num_days, 1 + 5 * 5), -1234.0)  # Initialize with missing value indicator
    
    # First column is day index
    features[:, 0] = np.arange(num_days)
    
    # Calculate moving averages for each price/volume column
    for price_idx in range(5):  # Open, High, Low, Close, Volume
        price_data = ohlcv_data[:, price_idx]
        
        # Calculate moving averages for each window size
        for window_idx, window_size in enumerate(window_sizes):
            feature_col = 1 + 5 * price_idx + window_idx
            
            # Calculate moving average
            for day in range(window_size - 1, num_days):
                start_idx = max(0, day - window_size + 1)
                features[day, feature_col] = np.mean(price_data[start_idx:day + 1])
        
        # Add original price as the 5th feature for each price type
        original_col = 1 + 5 * price_idx + 4
        features[:, original_col] = price_data
    
    return features


def normalize_features(features, start_idx=29):
    """
    Normalize features by dividing by the maximum value (following the original approach).
    
    Args:
        features: [num_days, 26] feature array
        start_idx: Index to start normalization from (skip initial padding days)
    
    Returns:
        normalized_features: Normalized feature array
    """
    normalized_features = features.copy()
    
    # Skip day index column (column 0) and normalize price-related features
    for price_idx in range(5):  # Open, High, Low, Close, Volume
        # Get columns for this price type (moving averages + original)
        start_col = 1 + 5 * price_idx
        end_col = start_col + 5
        
        # Find max value for normalization (from start_idx onwards)
        price_data = features[start_idx:, start_col:end_col]
        valid_data = price_data[price_data > -1000]  # Ignore missing values
        
        if len(valid_data) > 0:
            max_val = np.max(valid_data)
            if max_val > 0:
                # Normalize all columns for this price type
                mask = normalized_features[:, start_col:end_col] > -1000
                normalized_features[:, start_col:end_col][mask] /= max_val
    
    return normalized_features


def transform_parquet_to_ml_format(parquet_path, output_path, market_name='NASDAQ', 
                                   start_date='2012-11-19', min_days=100):
    """
    Transform cleaned_all.parquet to the format expected by the ML model.
    
    Args:
        parquet_path: Path to cleaned_all.parquet file
        output_path: Path to save the transformed data
        market_name: Market name for output file
        start_date: Start date for data selection
        min_days: Minimum number of days required for a stock
    """
    print(f"📂 Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"📊 Original data shape: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} -> {df['date'].max()}")
    
    # Filter data from start_date onwards
    start_date = pd.to_datetime(start_date)
    df = df[df['date'] >= start_date].copy()
    
    print(f"📊 After date filtering: {df.shape}")
    
    # Get unique stocks and dates
    unique_stocks = df['id'].unique()
    unique_dates = sorted(df['date'].unique())
    
    print(f"📈 Number of stocks: {len(unique_stocks)}")
    print(f"📅 Number of dates: {len(unique_dates)}")
    
    # Create date mappings
    index_tra_dates = {i: date.strftime('%Y-%m-%d %H:%M:%S') for i, date in enumerate(unique_dates)}
    tra_dates_index = {date.strftime('%Y-%m-%d %H:%M:%S'): i for i, date in enumerate(unique_dates)}
    
    # Process each stock
    all_features = {}
    processed_stocks = 0
    
    for stock_id in unique_stocks:
        df_stock = df[df['id'] == stock_id].sort_values('date').copy()
        
        # Skip stocks with insufficient data
        if len(df_stock) < min_days:
            print(f"⚠️  Skipping stock {stock_id}: only {len(df_stock)} days (need {min_days})")
            continue
        
        try:
            # Create basic OHLCV features from available data
            ohlcv_data = create_basic_price_features(df_stock)
            
            # Calculate moving averages and features
            features = calculate_moving_averages(ohlcv_data)
            
            # Normalize features
            normalized_features = normalize_features(features)
            
            # Store in the expected format
            all_features[stock_id] = normalized_features
            processed_stocks += 1
            
            if processed_stocks % 100 == 0:
                print(f"🔄 Processed {processed_stocks} stocks...")
        
        except Exception as e:
            print(f"⚠️  Error processing stock {stock_id}: {e}")
            continue
    
    print(f"✅ Successfully processed {processed_stocks} stocks")
    
    # Save in the expected pickle format
    save_data = {
        'all_features': all_features,
        'index_tra_dates': index_tra_dates,
        'tra_dates_index': tra_dates_index
    }
    
    output_file = f"{market_name}_all_features.pkl"
    output_full_path = os.path.join(output_path, output_file)
    
    with open(output_full_path, 'wb') as fw:
        pkl.dump(save_data, fw)
    
    print(f"💾 Saved transformed data to: {output_full_path}")
    
    # Create additional files needed by the ML model
    
    # 1. Trading dates CSV
    dates_file = os.path.join(output_path, f"{market_name}_aver_line_dates.csv")
    with open(dates_file, 'w') as f:
        for date_str in [index_tra_dates[i] for i in sorted(index_tra_dates.keys())]:
            f.write(f"{date_str}\n")
    
    print(f"📅 Saved trading dates to: {dates_file}")
    
    # 2. Tickers list
    tickers_file = os.path.join(output_path, f"{market_name}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    with open(tickers_file, 'w') as f:
        for stock_id in all_features.keys():
            f.write(f"{stock_id}\n")
    
    print(f"📈 Saved tickers list to: {tickers_file}")
    
    return save_data


def create_sample_inference_data(parquet_path, output_path, market_name='NASDAQ', 
                                 sample_stocks=50, sample_days=100):
    """
    Create a smaller sample dataset for inference testing.
    """
    print(f"🔬 Creating sample inference data...")
    
    df = pd.read_parquet(parquet_path)
    
    # Get data from a reasonable time range (not just recent)
    all_dates = sorted(df['date'].unique())
    
    # Use a slice from the middle of the dataset to ensure we have enough history
    if len(all_dates) > sample_days * 2:
        start_idx = len(all_dates) // 2  # Start from middle
        end_idx = start_idx + sample_days
        selected_dates = all_dates[start_idx:end_idx]
    else:
        selected_dates = all_dates[-sample_days:]  # Fallback to recent dates
    
    df_selected = df[df['date'].isin(selected_dates)].copy()
    
    # Get stocks with sufficient data in the selected period
    stock_counts = df_selected['id'].value_counts()
    # Only select stocks that have at least 10 data points
    sufficient_stocks = stock_counts[stock_counts >= 10]
    top_stocks = sufficient_stocks.head(sample_stocks).index.tolist()
    
    df_sample = df_selected[df_selected['id'].isin(top_stocks)].copy()
    
    print(f"📊 Sample data shape: {df_sample.shape}")
    print(f"📈 Stocks: {len(top_stocks)}")
    print(f"📅 Days: {len(selected_dates)}")
    print(f"📅 Date range: {min(selected_dates)} -> {max(selected_dates)}")
    
    # Save sample data temporarily and transform it
    temp_file = os.path.join(output_path, 'temp_sample.parquet')
    df_sample.to_parquet(temp_file, index=False)
    
    try:
        # Transform the sample data
        # Get the actual date range from the sample data
        sample_dates = sorted(df_sample['date'].unique())
        sample_start_date = sample_dates[0].strftime('%Y-%m-%d')
        
        result = transform_parquet_to_ml_format(
            temp_file, 
            output_path, 
            f"{market_name}_sample",
            start_date=sample_start_date,  # Use the actual start date from sample
            min_days=5  # Lower minimum for sample data
        )
        
        # Clean up temp file
        os.remove(temp_file)
        return result
        
    except Exception as e:
        # Clean up temp file even if there's an error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e


def main():
    parser = argparse.ArgumentParser(description='Transform parquet data for ML model')
    parser.add_argument('--input', type=str, default='cleaned_all.parquet',
                       help='Input parquet file path')
    parser.add_argument('--output', type=str, default='../ml-model/data/',
                       help='Output directory')
    parser.add_argument('--market', type=str, default='NASDAQ',
                       help='Market name')
    parser.add_argument('--start_date', type=str, default='2012-11-19',
                       help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--min_days', type=int, default=100,
                       help='Minimum number of days required per stock')
    parser.add_argument('--sample_only', action='store_true',
                       help='Create only sample data for testing')
    parser.add_argument('--sample_stocks', type=int, default=50,
                       help='Number of stocks in sample')
    parser.add_argument('--sample_days', type=int, default=100,
                       help='Number of recent days in sample')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    if args.sample_only:
        # Create sample data for inference testing
        create_sample_inference_data(
            args.input, 
            args.output, 
            args.market,
            args.sample_stocks,
            args.sample_days
        )
    else:
        # Transform full dataset
        transform_parquet_to_ml_format(
            args.input,
            args.output,
            args.market,
            args.start_date,
            args.min_days
        )
    
    print("✅ Data transformation completed!")


if __name__ == "__main__":
    main()


# Usage examples:
"""
# Transform full dataset:
python transform_parquet_data.py --input cleaned_all.parquet --output ../ml-model/data/

# Create sample data for testing:
python transform_parquet_data.py --sample_only --sample_stocks 20 --sample_days 50

# Custom parameters:
python transform_parquet_data.py --input cleaned_all.parquet --output ./inference_data/ --market NASDAQ_CUSTOM --start_date 2015-01-01 --min_days 200
"""