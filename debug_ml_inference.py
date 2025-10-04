#!/usr/bin/env python3
"""
Debug script for ML inference issues in main_pipeline.py
"""

import sys
import pickle as pkl
import torch
from pathlib import Path
import os

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "ml-model"))
sys.path.insert(0, str(Path(__file__).parent / "inference"))

def debug_ml_inference():
    """Debug the ML inference process step by step"""
    
    print("=" * 60)
    print("ML INFERENCE DEBUG")
    print("=" * 60)
    
    # Check if pkl files exist
    pkl_files = [
        "ml-model/data/TOP_LONG_all_features.pkl",
        "ml-model/data/BOTTOM_SHORT_all_features.pkl"
    ]
    
    for pkl_path in pkl_files:
        if os.path.exists(pkl_path):
            print(f"✓ Found: {pkl_path}")
            
            # Load and examine data
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)
            
            all_features = data.get('all_features', {})
            print(f"  - Companies: {len(all_features)}")
            
            if all_features:
                sample_key = list(all_features.keys())[0]
                sample_data = all_features[sample_key]
                print(f"  - Sample key: {sample_key}")
                print(f"  - Sample shape: {sample_data.shape}")
                print(f"  - Sample data range: {sample_data.min():.2f} to {sample_data.max():.2f}")
                
                # Check for missing values
                missing_count = (sample_data == -1234).sum()
                print(f"  - Missing values (-1234): {missing_count}")
                
        else:
            print(f"❌ Missing: {pkl_path}")
    
    # Test model loading
    model_path = "ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tv_100.ckpt"
    
    if os.path.exists(model_path):
        print(f"✓ Found model: {model_path}")
        
        try:
            from simple_inference import StockInference
            
            # Initialize inference
            inference = StockInference(model_path, device='cpu')
            inference.load_model()
            print("✓ Model loaded successfully")
            
            # Test with sample data
            if os.path.exists("ml-model/data/TOP_LONG_all_features.pkl"):
                with open("ml-model/data/TOP_LONG_all_features.pkl", 'rb') as f:
                    data = pkl.load(f)
                
                all_features = data.get('all_features', {})
                if all_features:
                    print(f"\n🔮 Testing inference with {len(all_features)} companies...")
                    
                    # Run prediction
                    predictions = inference.predict(all_features, days=32)
                    
                    print(f"✓ Predictions returned: {len(predictions)}")
                    if predictions:
                        values = list(predictions.values())
                        print(f"  - Prediction range: {min(values):.4f} to {max(values):.4f}")
                        print(f"  - Sample predictions: {dict(list(predictions.items())[:3])}")
                    else:
                        print("❌ No predictions generated!")
                        
                        # Debug why no predictions
                        print("\n🔍 Debugging empty predictions...")
                        
                        # Check each company individually
                        for i, (stock_id, features_array) in enumerate(list(all_features.items())[:3]):
                            print(f"\n  Testing company {i+1}: {stock_id}")
                            print(f"    Shape: {features_array.shape}")
                            
                            # Check data quality
                            recent_data = features_array[-32:, 1:]  # Last 32 days, exclude date column
                            missing_ratio = (recent_data == -1234).sum() / recent_data.size
                            print(f"    Missing ratio: {missing_ratio:.2%}")
                            
                            if missing_ratio <= 0.5:
                                try:
                                    # Test individual prediction
                                    single_pred = inference.predict({stock_id: features_array}, days=32)
                                    print(f"    Individual prediction: {single_pred}")
                                except Exception as e:
                                    print(f"    Error: {e}")
                
            
        except Exception as e:
            print(f"❌ Error during inference test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Model not found: {model_path}")

if __name__ == "__main__":
    debug_ml_inference()