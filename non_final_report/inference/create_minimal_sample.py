"""
Simple script to create a minimal working sample from cleaned_all.parquet 
that can be used for testing the inference pipeline.
"""

import pandas as pd
import numpy as np
import pickle as pkl
import os
from datetime import datetime

def create_minimal_sample():
    """Create a minimal sample that works with the ML model."""
    
    print("📂 Loading parquet data...")
    df = pd.read_parquet('../cleaned_all.parquet')
    
    print(f"📊 Original data: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} -> {df['date'].max()}")
    
    # Find stocks with the most data points
    stock_counts = df['id'].value_counts()
    print(f"📈 Top 10 stocks by data count:")
    print(stock_counts.head(10))
    
    # Take top 3 stocks with most data
    top_stocks = stock_counts.head(3).index.tolist()
    df_sample = df[df['id'].isin(top_stocks)].copy()
    
    print(f"📊 Sample data: {df_sample.shape}")
    
    # Create simple features for each stock
    all_features = {}
    
    for stock_id in top_stocks:
        stock_data = df_sample[df_sample['id'] == stock_id].sort_values('date')
        
        print(f"📈 Processing {stock_id}: {len(stock_data)} rows")
        
        # Create a simple time series with basic features
        num_days = len(stock_data)
        
        # Create features matrix: [num_days, 26] (day_index + 25 features)
        features = np.zeros((num_days, 26))
        
        # Day index
        features[:, 0] = np.arange(num_days)
        
        # Use available features from the cleaned data
        # Map some of the existing features to the 25 expected features
        if 'prc' in stock_data.columns:
            price_data = stock_data['prc'].values
        else:
            price_data = stock_data['stock_ret'].values
            
        # Fill the feature matrix with price-based features
        # For simplicity, we'll use the price data to create 25 synthetic features
        for i in range(25):
            if i < 5:  # First 5 features - price-based
                features[:, i+1] = price_data * (1 + i * 0.01)  # Slight variations
            elif i < 10:  # Next 5 features - moving averages (simplified)
                window = min(5 + i, num_days)
                for j in range(num_days):
                    start_idx = max(0, j - window + 1)
                    features[j, i+1] = np.mean(price_data[start_idx:j+1])
            else:  # Remaining features - synthetic data based on price
                features[:, i+1] = price_data * np.sin(i) + np.random.normal(0, 0.1, num_days)
        
        all_features[stock_id] = features
    
    # Create date mappings
    unique_dates = sorted(df_sample['date'].unique())
    index_tra_dates = {i: date.strftime('%Y-%m-%d %H:%M:%S') for i, date in enumerate(unique_dates)}
    tra_dates_index = {date.strftime('%Y-%m-%d %H:%M:%S'): i for i, date in enumerate(unique_dates)}
    
    # Save in the expected format
    save_data = {
        'all_features': all_features,
        'index_tra_dates': index_tra_dates,
        'tra_dates_index': tra_dates_index
    }
    
    # Create output directory
    output_dir = '../ml-model/data-example'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the pickle file
    output_file = os.path.join(output_dir, 'NASDAQ_all_features.pkl')
    with open(output_file, 'wb') as fw:
        pkl.dump(save_data, fw)
    
    print(f"💾 Saved sample data to: {output_file}")
    
    # Create supporting files
    dates_file = os.path.join(output_dir, 'NASDAQ_aver_line_dates.csv')
    with open(dates_file, 'w') as f:
        for i in sorted(index_tra_dates.keys()):
            f.write(f"{index_tra_dates[i]}\n")
    
    tickers_file = os.path.join(output_dir, 'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv')
    with open(tickers_file, 'w') as f:
        for stock_id in all_features.keys():
            f.write(f"{stock_id}\n")
    
    print(f"📅 Created supporting files:")
    print(f"  - {dates_file}")
    print(f"  - {tickers_file}")
    
    # Print sample info
    print(f"\n📊 Sample Summary:")
    print(f"  - Stocks: {len(all_features)}")
    print(f"  - Time periods: {len(unique_dates)}")
    print(f"  - Features per stock: {list(all_features.values())[0].shape}")
    
    return save_data

if __name__ == "__main__":
    create_minimal_sample()