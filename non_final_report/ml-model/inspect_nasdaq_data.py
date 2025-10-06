import pickle as pkl
import numpy as np
import os

def print_raw_pkl_head():
    """
    Print the raw structure and head of the pickle file
    """
    data_path = os.path.join(os.getcwd(), './data-example')
    file_path = os.path.join(data_path, 'NASDAQ_all_features.pkl')
    
    print("=" * 60)
    print("RAW PICKLE FILE STRUCTURE")
    print("=" * 60)
    
    try:
        with open(file_path, 'rb') as fr:
            data = pkl.load(fr)
        
        print(f"📁 Raw pickle object type: {type(data)}")
        print(f"📁 Raw pickle object size in memory: {len(str(data))} characters")
        print()
        
        # Print raw structure
        print("🔍 Raw Dictionary Keys and Types:")
        for key, value in data.items():
            print(f"   '{key}': {type(value)}")
            if isinstance(value, dict):
                print(f"      └─ Contains {len(value)} items")
                if len(value) > 0:
                    sample_key = list(value.keys())[0]
                    sample_value = value[sample_key]
                    print(f"      └─ Sample item: '{sample_key}' -> {type(sample_value)}")
                    if hasattr(sample_value, 'shape'):
                        print(f"         └─ Shape: {sample_value.shape}")
            elif hasattr(value, '__len__'):
                print(f"      └─ Length: {len(value)}")
        print()
        
        # Print raw head of all_features
        if 'all_features' in data:
            all_features = data['all_features']
            print("📊 RAW all_features dictionary head:")
            stock_keys = list(all_features.keys())
            
            print(f"   First 5 stock keys (raw): {stock_keys[:5]}")
            print()
            
            # Show raw numpy array for first stock
            first_stock = stock_keys[0]
            raw_array = all_features[first_stock]
            
            print(f"🔍 RAW numpy array for '{first_stock}':")
            print(f"   Array type: {type(raw_array)}")
            print(f"   Array dtype: {raw_array.dtype}")
            print(f"   Array shape: {raw_array.shape}")
            print(f"   Array memory usage: {raw_array.nbytes} bytes ({raw_array.nbytes/1024/1024:.2f} MB)")
            print()
            
            print(f"📋 First 3x10 raw values from '{first_stock}' array:")
            print("      ", end="")
            for col in range(10):
                print(f"  Col{col:2d}    ", end="")
            print()
            
            for row in range(min(3, raw_array.shape[0])):
                print(f"Row{row:2d}: ", end="")
                for col in range(min(10, raw_array.shape[1])):
                    val = raw_array[row, col]
                    if val == -1234:
                        print(" MISSING  ", end="")
                    else:
                        print(f"{val:9.6f}", end="")
                print()
            print()
        
        # Print raw head of date mappings
        if 'index_tra_dates' in data:
            index_tra_dates = data['index_tra_dates']
            print("📅 RAW index_tra_dates dictionary head:")
            print(f"   Type: {type(index_tra_dates)}")
            print(f"   Length: {len(index_tra_dates)}")
            
            sample_keys = list(index_tra_dates.keys())[:5]
            print("   First 5 raw key-value pairs:")
            for key in sample_keys:
                print(f"      {key} -> '{index_tra_dates[key]}'")
            print()
        
        if 'tra_dates_index' in data:
            tra_dates_index = data['tra_dates_index']
            print("📅 RAW tra_dates_index dictionary head:")
            print(f"   Type: {type(tra_dates_index)}")
            print(f"   Length: {len(tra_dates_index)}")
            
            sample_keys = list(tra_dates_index.keys())[:5]
            print("   First 5 raw key-value pairs:")
            for key in sample_keys:
                print(f"      '{key}' -> {tra_dates_index[key]}")
            print()
        
        # Memory analysis
        print("💾 Memory Analysis:")
        import sys
        total_size = sys.getsizeof(data)
        print(f"   Total pickle object size: {total_size} bytes ({total_size/1024/1024:.2f} MB)")
        
        if 'all_features' in data:
            features_size = sum(arr.nbytes for arr in data['all_features'].values())
            print(f"   all_features arrays size: {features_size} bytes ({features_size/1024/1024:.2f} MB)")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error reading raw pickle: {str(e)}")

def inspect_nasdaq_features():
    """
    Read and inspect the NASDAQ_all_features.pkl file to understand its structure
    """
    data_path = os.path.join(os.getcwd(), './data-example')
    file_path = os.path.join(data_path, 'NASDAQ_all_features.pkl')
    
    print("=" * 60)
    print("NASDAQ_all_features.pkl Inspector")
    print("=" * 60)
    
    try:
        # Load the pickle file
        with open(file_path, 'rb') as fr:
            data = pkl.load(fr)
        
        print(f"✓ Successfully loaded: {file_path}")
        print()
        
        # Inspect top-level structure
        print("📁 Top-level keys in the pickle file:")
        for key in data.keys():
            print(f"   - {key}: {type(data[key])}")
        print()
        
        # Inspect all_features
        if 'all_features' in data:
            all_features = data['all_features']
            print("📊 All Features Analysis:")
            print(f"   - Type: {type(all_features)}")
            print(f"   - Number of stocks: {len(all_features)}")
            print()
            
            # Get first few stock symbols
            stock_symbols = list(all_features.keys())
            print(f"📈 First 10 Stock Symbols:")
            for i, symbol in enumerate(stock_symbols[:10]):
                print(f"   {i+1:2d}. {symbol}")
            print(f"   ... and {len(stock_symbols)-10} more stocks")
            print()
            
            # Inspect feature structure for first stock
            first_stock = stock_symbols[0]
            features = all_features[first_stock]
            print(f"🔍 Feature Analysis for '{first_stock}':")
            print(f"   - Shape: {features.shape}")
            print(f"   - Data type: {features.dtype}")
            print(f"   - Date range: {features.shape[0]} trading days")
            print(f"   - Features per day: {features.shape[1]} features")
            print()
            
            # Show first few rows and columns
            print(f"📋 Sample Data for '{first_stock}' (first 5 days, first 10 features):")
            print("   Day | Feature 0-9")
            print("   " + "-" * 50)
            
            for day in range(min(5, features.shape[0])):
                row_data = features[day, :10]  # First 10 features
                formatted_row = " | ".join([f"{val:8.4f}" if val != -1234 else "  MISSING" for val in row_data])
                print(f"   {day:3d} | {formatted_row}")
            print()
            
            # Feature interpretation based on data.py analysis
            print("🏷️  Feature Index Mapping (based on data.py):")
            feature_descriptions = [
                "Day Index",
                "5-day MA Open", "10-day MA Open", "20-day MA Open", "30-day MA Open", "Current Open",
                "5-day MA High", "10-day MA High", "20-day MA High", "30-day MA High", "Current High", 
                "5-day MA Low", "10-day MA Low", "20-day MA Low", "30-day MA Low", "Current Low",
                "5-day MA Close", "10-day MA Close", "20-day MA Close", "30-day MA Close", "Current Close",
                "5-day MA Volume", "10-day MA Volume", "20-day MA Volume", "30-day MA Volume", "Current Volume"
            ]
            
            for i, desc in enumerate(feature_descriptions):
                if i < features.shape[1]:
                    sample_val = features[0, i] if features[0, i] != -1234 else "MISSING"
                    print(f"   {i:2d}: {desc:<20} = {sample_val}")
            print()
            
            # Check for missing data
            missing_count = np.sum(features == -1234)
            total_elements = features.size
            missing_pct = (missing_count / total_elements) * 100
            print(f"❗ Missing Data Analysis for '{first_stock}':")
            print(f"   - Missing values: {missing_count:,} / {total_elements:,} ({missing_pct:.2f}%)")
            print()
        
        # Inspect date mappings
        if 'index_tra_dates' in data and 'tra_dates_index' in data:
            index_tra_dates = data['index_tra_dates']
            tra_dates_index = data['tra_dates_index']
            
            print("📅 Date Mapping Analysis:")
            print(f"   - Total trading dates: {len(index_tra_dates)}")
            
            # Show first and last few dates
            print("   - First 5 dates:")
            for i in range(min(5, len(index_tra_dates))):
                print(f"     Index {i}: {index_tra_dates[i]}")
            
            print("   - Last 5 dates:")
            last_indices = list(range(max(0, len(index_tra_dates)-5), len(index_tra_dates)))
            for i in last_indices:
                print(f"     Index {i}: {index_tra_dates[i]}")
            print()
        
        # Summary statistics
        print("📊 Dataset Summary:")
        if 'all_features' in data:
            total_stocks = len(data['all_features'])
            sample_stock = list(data['all_features'].values())[0]
            total_days = sample_stock.shape[0]
            total_features = sample_stock.shape[1]
            
            print(f"   - Total stocks: {total_stocks:,}")
            print(f"   - Trading days per stock: {total_days:,}")
            print(f"   - Features per day: {total_features}")
            print(f"   - Total data points: {total_stocks * total_days * total_features:,}")
            
            # Estimate memory usage
            estimated_size_mb = (total_stocks * total_days * total_features * 8) / (1024 * 1024)  # 8 bytes per float64
            print(f"   - Estimated memory: {estimated_size_mb:.1f} MB")
        
        print("\n" + "=" * 60)
        print("✅ Inspection completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        print("   Make sure you have run 'python data.py' to generate the file.")
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")

def sample_stock_data(stock_symbol, num_days=10):
    """
    Show detailed data for a specific stock
    
    Args:
        stock_symbol (str): Stock symbol to inspect (e.g., 'AAPL')
        num_days (int): Number of recent days to show
    """
    data_path = os.path.join(os.getcwd(), './data-example')
    file_path = os.path.join(data_path, 'NASDAQ_all_features.pkl')
    
    try:
        with open(file_path, 'rb') as fr:
            data = pkl.load(fr)
        
        all_features = data['all_features']
        
        if stock_symbol not in all_features:
            print(f"❌ Stock '{stock_symbol}' not found in dataset")
            print("Available stocks:", list(all_features.keys())[:20], "...")
            return
        
        features = all_features[stock_symbol]
        print(f"\n📈 Detailed data for {stock_symbol}")
        print(f"Shape: {features.shape}")
        print("-" * 80)
        
        # Show last N days of data
        start_idx = max(0, features.shape[0] - num_days)
        
        print("Recent trading days (last {} days):".format(num_days))
        print("Day | Open-MA5  Open-MA10 Open-MA20 Open-MA30 Open-Curr |  Close-Curr")
        print("-" * 80)
        
        for day in range(start_idx, features.shape[0]):
            if features[day, 1] != -1234:  # Check if data exists
                row = features[day]
                print(f"{day:3d} | {row[1]:8.4f}  {row[2]:8.4f}  {row[3]:8.4f}  {row[4]:8.4f}  {row[5]:8.4f} | {row[20]:10.4f}")
            else:
                print(f"{day:3d} | MISSING DATA")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Print raw pickle structure first
    print_raw_pkl_head()
    
    # Run the main inspection
    inspect_nasdaq_features()
    
    # Example: Show detailed data for a specific stock
    print("\n" + "="*60)
    sample_stock_data("AAPL", num_days=5)