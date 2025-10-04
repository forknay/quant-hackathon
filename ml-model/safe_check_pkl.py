"""Safe check of pkl file - handles partial/corrupted files"""
import pickle
import os

pkl_path = 'data/CUSTOM_all_features.pkl'
file_size = os.path.getsize(pkl_path)
print(f"File size: {file_size / (1024**3):.2f} GB")

if file_size == 0:
    print("❌ File is empty!")
else:
    print("File exists and has content. Attempting to load...")
    try:
        with open(pkl_path, 'rb') as f:
            # Try to load
            data = pickle.load(f)
            print(f"✅ Successfully loaded!")
            print(f"Keys: {list(data.keys())}")
            print(f"Stocks: {len(data.get('all_features', {}))}")
    except EOFError:
        print("❌ File appears incomplete (EOFError) - data.py may have been interrupted")
        print("   Please re-run: python data.py")
    except Exception as e:
        print(f"❌ Error: {e}")

