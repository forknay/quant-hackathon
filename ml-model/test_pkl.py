"""Quick test to check CUSTOM_all_features.pkl content"""
import pickle
import sys

try:
    with open('data/CUSTOM_all_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("=" * 80)
    print("PKL FILE STRUCTURE CHECK")
    print("=" * 80)
    print(f"Keys: {list(data.keys())}")
    print(f"Number of stocks: {len(data['all_features'])}")
    print(f"Number of trading dates: {len(data['index_tra_dates'])}")
    
    if len(data['all_features']) > 0:
        print("\n" + "-" * 80)
        print("SAMPLE STOCK DATA")
        print("-" * 80)
        sample_key = list(data['all_features'].keys())[0]
        sample_data = data['all_features'][sample_key]
        print(f"Sample stock: {sample_key}")
        print(f"Shape: {sample_data.shape}")
        print(f"Columns: {sample_data.shape[1]} (should be 26)")
        print(f"Days: {sample_data.shape[0]}")
        
        # Check for valid data (not all -1234)
        valid_rows = (sample_data[:, 1:] != -1234).any(axis=1).sum()
        print(f"Valid rows (with data): {valid_rows} / {sample_data.shape[0]}")
        
        print(f"\nFirst 5 stocks: {list(data['all_features'].keys())[:5]}")
        print(f"\nDate range:")
        print(f"  First: {data['index_tra_dates'][0]}")
        print(f"  Last: {data['index_tra_dates'][len(data['index_tra_dates'])-1]}")
        
        print("\n" + "=" * 80)
        print("✅ PKL FILE LOOKS GOOD - READY FOR INFERENCE!")
        print("=" * 80)
    else:
        print("\n❌ ERROR: No stocks processed!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error reading pkl file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

