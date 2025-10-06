"""
Create a larger sample dataset from cleaned_all.parquet for comparison
"""

import pandas as pd
import numpy as np
import pickle as pkl
import os
from datetime import datetime

def create_larger_sample():
    """Create a larger sample with more stocks and longer time series."""
    
    print("📂 Loading parquet data...")
    df = pd.read_parquet('../cleaned_all.parquet')
    
    # Find stocks with substantial data (>100 time points)
    stock_counts = df['id'].value_counts()
    stocks_with_data = stock_counts[stock_counts >= 100]
    
    print(f"📊 Found {len(stocks_with_data)} stocks with >=100 data points")
    print(f"📈 Top 20 stocks by data count:")
    print(stocks_with_data.head(20))
    
    # Take top 10 stocks with most data
    top_stocks = stocks_with_data.head(10).index.tolist()
    df_sample = df[df['id'].isin(top_stocks)].copy()
    
    print(f"📊 Sample data: {df_sample.shape}")
    print(f"📅 Date range: {df_sample['date'].min()} -> {df_sample['date'].max()}")
    
    # Create more realistic features for each stock
    all_features = {}
    
    for stock_id in top_stocks:
        stock_data = df_sample[df_sample['id'] == stock_id].sort_values('date')
        
        print(f"📈 Processing {stock_id}: {len(stock_data)} rows")
        
        num_days = len(stock_data)
        
        # Create features matrix: [num_days, 26] (day_index + 25 features)
        features = np.zeros((num_days, 26))
        
        # Day index
        features[:, 0] = np.arange(num_days)
        
        # Use multiple features from the cleaned data to create more realistic patterns
        available_features = ['prc', 'stock_ret', 'me', 'turnover_126d', 'ret_1_0', 
                            'ret_12_1', 'ivol_capm_21d', 'beta_60m', 'mispricing_perf']
        
        # Create 25 features using combinations of available data
        feature_idx = 1
        for i, feat_name in enumerate(available_features):
            if feat_name in stock_data.columns and feature_idx < 26:
                base_data = stock_data[feat_name].values
                
                # Use the base feature
                features[:, feature_idx] = base_data
                feature_idx += 1
                
                if feature_idx < 26:
                    # Create moving average of the feature
                    ma_data = pd.Series(base_data).rolling(window=5, min_periods=1).mean().values
                    features[:, feature_idx] = ma_data
                    feature_idx += 1
                
                if feature_idx < 26:
                    # Create lagged version
                    lagged_data = np.roll(base_data, 1)
                    lagged_data[0] = base_data[0]  # Fill first value
                    features[:, feature_idx] = lagged_data
                    feature_idx += 1
        
        # Fill remaining features with synthetic data based on price patterns
        while feature_idx < 26:
            if 'prc' in stock_data.columns:
                price_data = stock_data['prc'].values
                synthetic_feature = price_data * np.sin(feature_idx) + np.random.normal(0, 0.01, num_days)
                features[:, feature_idx] = synthetic_feature
            else:
                features[:, feature_idx] = np.random.normal(0, 0.1, num_days)
            feature_idx += 1
        
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
    
    # Save as a new file for comparison
    output_file = '../ml-model/data-example/NASDAQ_all_features_large.pkl'
    with open(output_file, 'wb') as fw:
        pkl.dump(save_data, fw)
    
    print(f"💾 Saved larger sample to: {output_file}")
    
    # Print sample info
    print(f"\n📊 Large Sample Summary:")
    print(f"  - Stocks: {len(all_features)}")
    print(f"  - Time periods: {len(unique_dates)}")
    print(f"  - Features per stock: {list(all_features.values())[0].shape}")
    
    for stock_id, data in all_features.items():
        print(f"    {stock_id}: {data.shape}")
    
    return save_data

if __name__ == "__main__":
    create_larger_sample()