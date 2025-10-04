"""
Test CUSTOM_all_features.pkl with inference pipeline

This script:
1. Loads the CUSTOM_all_features.pkl file
2. Validates its structure
3. Runs basic inference test (without actual model)
4. Shows what predictions would look like
"""

import sys
import pickle as pkl
import numpy as np
from pathlib import Path

def test_custom_pkl():
    print("=" * 80)
    print("TESTING CUSTOM_ALL_FEATURES.PKL FOR INFERENCE")
    print("=" * 80)
    
    # Load the pkl file
    pkl_path = "../ml-model/data/CUSTOM_all_features.pkl"
    print(f"\n[STEP 1] Loading {pkl_path}...")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        print("✅ Successfully loaded!")
    except FileNotFoundError:
        print(f"❌ File not found: {pkl_path}")
        print("   Please run: cd ml-model && python data.py")
        return False
    except EOFError:
        print("❌ File is incomplete (EOFError)")
        print("   The file was not fully saved. Please re-run: cd ml-model && python data.py")
        return False
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Validate structure
    print("\n[STEP 2] Validating structure...")
    required_keys = ['all_features', 'index_tra_dates', 'tra_dates_index']
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing required key: {key}")
            return False
    print("✅ All required keys present")
    
    # Check content
    print("\n[STEP 3] Analyzing content...")
    all_features = data['all_features']
    num_stocks = len(all_features)
    num_dates = len(data['index_tra_dates'])
    
    print(f"  Companies: {num_stocks:,}")
    print(f"  Trading dates: {num_dates:,}")
    
    if num_stocks == 0:
        print("❌ No companies in the file!")
        return False
    
    # Sample analysis
    print("\n[STEP 4] Analyzing sample company...")
    sample_id = list(all_features.keys())[0]
    sample_data = all_features[sample_id]
    
    print(f"  Sample company: {sample_id}")
    print(f"  Shape: {sample_data.shape}")
    print(f"  Expected: [num_days, 26]")
    
    if sample_data.shape[1] != 26:
        print(f"❌ Wrong number of columns! Expected 26, got {sample_data.shape[1]}")
        return False
    print("✅ Correct shape")
    
    # Check for valid data
    print("\n[STEP 5] Checking data quality...")
    
    # Count companies with sufficient valid data (need at least 32 days for inference)
    min_sequence_length = 32
    valid_companies = []
    
    for company_id, features in all_features.items():
        # Get recent data (columns 1-25, excluding day index column 0)
        recent_features = features[-min_sequence_length:, 1:]
        
        # Check if all data is valid (not -1234)
        if len(recent_features) >= min_sequence_length:
            if not np.any(recent_features == -1234):
                valid_companies.append(company_id)
    
    print(f"  Total companies: {num_stocks:,}")
    print(f"  Companies with ≥{min_sequence_length} valid days: {len(valid_companies):,}")
    print(f"  Usable for inference: {len(valid_companies):,} ({len(valid_companies)/num_stocks*100:.1f}%)")
    
    if len(valid_companies) == 0:
        print("❌ No companies have enough valid data for inference!")
        print("   Models typically need 32+ consecutive days of data")
        return False
    
    # Show sample companies
    print(f"\n  Sample valid companies (first 10):")
    for i, cid in enumerate(valid_companies[:10], 1):
        print(f"    {i:2d}. {cid}")
    
    # Simulate what inference would return
    print("\n[STEP 6] Simulating inference output...")
    print("  (This is what real inference would look like)")
    
    # Mock predictions for top 5
    top_5 = valid_companies[:5] if len(valid_companies) >= 5 else valid_companies
    mock_predictions = {
        'top_5': {
            'symbols': top_5,
            'predictions': [0.85, 0.72, 0.68, 0.61, 0.55][:len(top_5)],
            'weights': [0.25, 0.22, 0.20, 0.18, 0.15][:len(top_5)]
        }
    }
    
    print("\n  Example prediction output:")
    print("  {")
    print("    'top_5': {")
    print(f"      'symbols': {mock_predictions['top_5']['symbols']}")
    print(f"      'predictions': {mock_predictions['top_5']['predictions']}")
    print(f"      'weights': {mock_predictions['top_5']['weights']}")
    print("    }")
    print("  }")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("✅ DATA IS READY FOR INFERENCE!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  ✓ {num_stocks:,} companies processed")
    print(f"  ✓ {len(valid_companies):,} companies ready for model inference")
    print(f"  ✓ Company IDs preserved (e.g., {valid_companies[0]})")
    print(f"  ✓ Can be matched with algo pipeline")
    
    print(f"\nNext steps:")
    print(f"  1. Run actual inference:")
    print(f"     cd inference")
    print(f"     python stock_inference.py \\")
    print(f"       --model_path '../ml-model/models/.../model_tt_100.ckpt' \\")
    print(f"       --data_path '../ml-model/data/CUSTOM_all_features.pkl'")
    print(f"\n  2. Predictions will use company IDs as keys")
    print(f"  3. Match with algo pipeline using these company IDs")
    
    return True

if __name__ == "__main__":
    success = test_custom_pkl()
    sys.exit(0 if success else 1)

