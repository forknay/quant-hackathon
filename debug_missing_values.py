#!/usr/bin/env python3
"""
Debug script to analyze missing values in TOP_LONG pkl file
"""

import pickle as pkl
import numpy as np

def debug_missing_values():
    """Debug missing values in the TOP_LONG pkl file"""
    
    pkl_path = "ml-model/data/TOP_LONG_all_features.pkl"
    
    print("=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    
    # Load data
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    
    all_features = data.get('all_features', {})
    print(f"Companies in file: {len(all_features)}")
    
    if all_features:
        days = 32  # Model requirement
        
        for i, (stock_id, features_array) in enumerate(all_features.items()):
            print(f"\n{'='*50}")
            print(f"Company {i+1}: {stock_id}")
            print(f"{'='*50}")
            print(f"Full array shape: {features_array.shape}")
            
            # Check recent data (last 32 days, exclude date column)
            recent_data = features_array[-days:, 1:]  # Last days, exclude date column
            print(f"Recent data shape (last {days} days): {recent_data.shape}")
            
            # Count missing values
            missing_count = (recent_data == -1234).sum()
            total_values = recent_data.size
            missing_ratio = missing_count / total_values
            
            print(f"Missing values (-1234): {missing_count} / {total_values}")
            print(f"Missing ratio: {missing_ratio:.2%}")
            print(f"Threshold (50%): {'❌ SKIP' if missing_ratio > 0.5 else '✅ OK'}")
            
            # Show data sample
            print(f"\nRecent data sample (last 5 days):")
            print("Shape:", recent_data[-5:].shape)
            print("Min/Max:", recent_data.min(), "to", recent_data.max())
            
            # Check if all values are missing
            if missing_count == total_values:
                print("⚠️  ALL recent data is missing!")
            elif missing_count == 0:
                print("✅ No missing values in recent data")
            else:
                print(f"📊 {missing_count} missing out of {total_values} values")
            
            # Show first few rows of recent data
            print(f"\nFirst 3 rows of recent data:")
            for j in range(min(3, len(recent_data))):
                row = recent_data[j]
                missing_in_row = (row == -1234).sum()
                print(f"  Row {j}: {missing_in_row}/{len(row)} missing, range: {row.min():.2f} to {row.max():.2f}")

if __name__ == "__main__":
    debug_missing_values()