import pickle

# Load the original NASDAQ data
with open('../ml-model/data-example/NASDAQ_all_features.pkl', 'rb') as f:
    data = pickle.load(f)

print("Original NASDAQ_all_features.pkl data:")
print(f"Number of stocks: {len(data['all_features'])}")
print(f"Sample stock IDs: {list(data['all_features'].keys())[:10]}")
print(f"Sample stock shape: {list(data['all_features'].values())[0].shape}")
print(f"Number of trading days: {len(data['index_tra_dates'])}")

# Show some sample stock data
sample_stock = list(data['all_features'].keys())[0]
sample_data = data['all_features'][sample_stock]
print(f"\nSample stock '{sample_stock}':")
print(f"Shape: {sample_data.shape}")
print(f"First few time steps:")
print(sample_data[:5, :6])  # First 5 rows, first 6 columns