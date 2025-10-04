"""
Compare inference results between different datasets
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
import pathlib

# Add the current directory to the path for importing stock_inference
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stock_inference import StockSelectionInference, print_inference_results

def run_comparison():
    """Compare inference results between small and large datasets."""
    
    print("🔀 DATASET COMPARISON - STOCK SELECTION INFERENCE")
    print("="*80)
    
    model_path = "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Dataset paths
    small_data = "../ml-model/data-example/NASDAQ_all_features_custom.pkl"  # 3 stocks
    large_data = "../ml-model/data-example/NASDAQ_all_features_large.pkl"   # 10 stocks
    
    datasets = [
        ("Small Dataset (3 stocks)", small_data),
        ("Large Dataset (10 stocks)", large_data)
    ]
    
    results_comparison = {}
    
    for dataset_name, data_path in datasets:
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            continue
            
        print(f"\n🔍 TESTING: {dataset_name}")
        print("="*60)
        
        try:
            # Initialize inference pipeline
            inference = StockSelectionInference(model_path)
            
            # Run inference with multiple top-k values
            results = inference.run_inference(
                data_source=data_path,
                top_k=[1, 3, 5, 10],
                sequence_length=32,
                feature_describe='all'
            )
            
            # Store results for comparison
            results_comparison[dataset_name] = results
            
            # Print detailed results
            print_inference_results(results)
            
            # Print key metrics
            data_info = results['data_info']
            predictions = results['raw_predictions']['values']
            valid_predictions = [p for p, v in zip(predictions, results['raw_predictions']['valid_mask']) if v > 0.5]
            
            print(f"📊 SUMMARY FOR {dataset_name}:")
            print(f"   Total stocks: {data_info['total_stocks']}")
            print(f"   Valid stocks: {data_info['valid_stocks']}")
            print(f"   Mean prediction: {np.mean(valid_predictions):.4f}")
            print(f"   Max prediction: {np.max(valid_predictions):.4f}")
            print(f"   Min prediction: {np.min(valid_predictions):.4f}")
            print(f"   Std deviation: {np.std(valid_predictions):.4f}")
            
        except Exception as e:
            print(f"❌ Error with {dataset_name}: {str(e)}")
            continue
    
    # Compare results side by side
    if len(results_comparison) >= 2:
        print("\n🔀 SIDE-BY-SIDE COMPARISON")
        print("="*80)
        
        dataset_names = list(results_comparison.keys())
        
        print(f"{'Metric':<25} {'Small (3 stocks)':<20} {'Large (10 stocks)':<20} {'Difference':<15}")
        print("-" * 80)
        
        for name in dataset_names:
            results = results_comparison[name]
            predictions = results['raw_predictions']['values']
            valid_predictions = [p for p, v in zip(predictions, results['raw_predictions']['valid_mask']) if v > 0.5]
            
            metrics = {
                'Total Stocks': results['data_info']['total_stocks'],
                'Valid Stocks': results['data_info']['valid_stocks'],
                'Mean Prediction': np.mean(valid_predictions),
                'Max Prediction': np.max(valid_predictions),
                'Min Prediction': np.min(valid_predictions),
                'Std Deviation': np.std(valid_predictions)
            }
            
            if name == dataset_names[0]:
                small_metrics = metrics
            else:
                large_metrics = metrics
        
        # Print comparison table
        for metric in small_metrics.keys():
            small_val = small_metrics[metric]
            large_val = large_metrics[metric]
            
            if isinstance(small_val, (int, float)) and isinstance(large_val, (int, float)):
                if metric in ['Total Stocks', 'Valid Stocks']:
                    diff = large_val - small_val
                    print(f"{metric:<25} {small_val:<20} {large_val:<20} +{diff}")
                else:
                    diff = large_val - small_val
                    pct_change = (diff / small_val * 100) if small_val != 0 else 0
                    print(f"{metric:<25} {small_val:<20.4f} {large_val:<20.4f} {diff:+.4f} ({pct_change:+.1f}%)")
        
        # Compare top stock selections
        print(f"\n🏆 TOP STOCK SELECTIONS COMPARISON:")
        print("-" * 50)
        
        for name in dataset_names:
            results = results_comparison[name]
            top_5 = results['portfolio_selections']['top_5']
            print(f"\n{name} - Top 5 stocks:")
            for i, (symbol, pred, weight) in enumerate(zip(top_5['symbols'], top_5['predictions'], top_5['weights'])):
                print(f"   {i+1}. {symbol:<20} | Score: {pred:8.2f} | Weight: {weight:6.2f}%")

if __name__ == "__main__":
    run_comparison()