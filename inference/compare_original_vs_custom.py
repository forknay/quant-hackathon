"""
Compare inference results between original NASDAQ data and custom data
"""

import sys
import os
import torch
import numpy as np
import pickle
from datetime import datetime
import pathlib

# Add the current directory to the path for importing stock_inference
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stock_inference import StockSelectionInference

def analyze_data_structure(data_path, name):
    """Analyze the structure of a dataset."""
    print(f"\n📊 ANALYZING {name.upper()} DATA STRUCTURE")
    print("="*50)
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded {name} data successfully")
        print(f"   Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"   {key} shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"   {key} length: {len(value)}")
                else:
                    print(f"   {key} type: {type(value)}")
        elif hasattr(data, 'shape'):
            print(f"   Shape: {data.shape}")
        elif isinstance(data, list):
            print(f"   Length: {len(data)}")
            if len(data) > 0:
                print(f"   First element type: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"   First element shape: {data[0].shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading {name} data: {str(e)}")
        return None

def run_original_vs_custom_comparison():
    """Compare original NASDAQ data with custom generated data."""
    
    print("🚀 ORIGINAL vs CUSTOM DATA COMPARISON")
    print("="*80)
    
    model_path = "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    original_data = "../ml-model/data-example/NASDAQ_all_features.pkl"
    custom_data = "../ml-model/data-example/NASDAQ_all_features_custom.pkl"
    
    # Verify all files exist
    files_to_check = [
        (model_path, "Model checkpoint"),
        (original_data, "Original NASDAQ data"),
        (custom_data, "Custom data")
    ]
    
    for file_path, description in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ {description} not found: {file_path}")
            return
        else:
            print(f"✅ {description} found")
    
    # Analyze data structures first
    original_dataset = analyze_data_structure(original_data, "original")
    custom_dataset = analyze_data_structure(custom_data, "custom")
    
    if original_dataset is None or custom_dataset is None:
        print("❌ Cannot proceed without valid datasets")
        return
    
    # Test configurations
    test_configs = [
        {"top_k": [5], "name": "Top 5 Selection"},
        {"top_k": [3], "name": "Conservative (Top 3)"},
        {"top_k": [10], "name": "Aggressive (Top 10)"},
    ]
    
    results_comparison = []
    
    for config in test_configs:
        print(f"\n🔧 TESTING CONFIGURATION: {config['name']}")
        print("="*60)
        
        config_results = {
            'config_name': config['name'],
            'top_k': config['top_k'][0]
        }
        
        # Test with original data
        print(f"\n📈 Testing with ORIGINAL NASDAQ data...")
        try:
            inference_original = StockSelectionInference(model_path)
            results_original = inference_original.run_inference(
                data_source=original_data,
                top_k=config['top_k'],
                sequence_length=32,
                feature_describe='all'
            )
            
            # Extract metrics
            predictions_orig = results_original['raw_predictions']['values']
            valid_predictions_orig = [p for p, v in zip(predictions_orig, results_original['raw_predictions']['valid_mask']) if v > 0.5]
            
            top_k_key = f"top_{config['top_k'][0]}"
            if top_k_key in results_original['portfolio_selections']:
                top_selection_orig = results_original['portfolio_selections'][top_k_key]
                top_stock_orig = top_selection_orig['symbols'][0] if top_selection_orig['symbols'] else "N/A"
                top_score_orig = top_selection_orig['predictions'][0] if top_selection_orig['predictions'] else 0
            else:
                top_stock_orig = "N/A"
                top_score_orig = 0
            
            config_results['original'] = {
                'total_stocks': results_original['data_info']['total_stocks'],
                'valid_stocks': results_original['data_info']['valid_stocks'],
                'mean_prediction': np.mean(valid_predictions_orig),
                'max_prediction': np.max(valid_predictions_orig),
                'min_prediction': np.min(valid_predictions_orig),
                'std_prediction': np.std(valid_predictions_orig),
                'top_stock': top_stock_orig,
                'top_score': top_score_orig,
                'all_predictions': valid_predictions_orig
            }
            
            print(f"✅ Original data results:")
            print(f"   Stocks: {results_original['data_info']['valid_stocks']}/{results_original['data_info']['total_stocks']}")
            print(f"   Mean prediction: {np.mean(valid_predictions_orig):.2f}")
            print(f"   Range: {np.min(valid_predictions_orig):.2f} to {np.max(valid_predictions_orig):.2f}")
            print(f"   Top stock: {top_stock_orig} (score: {top_score_orig:.2f})")
            
        except Exception as e:
            print(f"❌ Error with original data: {str(e)}")
            config_results['original'] = None
        
        # Test with custom data
        print(f"\n📊 Testing with CUSTOM data...")
        try:
            inference_custom = StockSelectionInference(model_path)
            results_custom = inference_custom.run_inference(
                data_source=custom_data,
                top_k=config['top_k'],
                sequence_length=32,
                feature_describe='all'
            )
            
            # Extract metrics
            predictions_custom = results_custom['raw_predictions']['values']
            valid_predictions_custom = [p for p, v in zip(predictions_custom, results_custom['raw_predictions']['valid_mask']) if v > 0.5]
            
            top_k_key = f"top_{config['top_k'][0]}"
            if top_k_key in results_custom['portfolio_selections']:
                top_selection_custom = results_custom['portfolio_selections'][top_k_key]
                top_stock_custom = top_selection_custom['symbols'][0] if top_selection_custom['symbols'] else "N/A"
                top_score_custom = top_selection_custom['predictions'][0] if top_selection_custom['predictions'] else 0
            else:
                top_stock_custom = "N/A"
                top_score_custom = 0
            
            config_results['custom'] = {
                'total_stocks': results_custom['data_info']['total_stocks'],
                'valid_stocks': results_custom['data_info']['valid_stocks'],
                'mean_prediction': np.mean(valid_predictions_custom),
                'max_prediction': np.max(valid_predictions_custom),
                'min_prediction': np.min(valid_predictions_custom),
                'std_prediction': np.std(valid_predictions_custom),
                'top_stock': top_stock_custom,
                'top_score': top_score_custom,
                'all_predictions': valid_predictions_custom
            }
            
            print(f"✅ Custom data results:")
            print(f"   Stocks: {results_custom['data_info']['valid_stocks']}/{results_custom['data_info']['total_stocks']}")
            print(f"   Mean prediction: {np.mean(valid_predictions_custom):.2f}")
            print(f"   Range: {np.min(valid_predictions_custom):.2f} to {np.max(valid_predictions_custom):.2f}")
            print(f"   Top stock: {top_stock_custom} (score: {top_score_custom:.2f})")
            
        except Exception as e:
            print(f"❌ Error with custom data: {str(e)}")
            config_results['custom'] = None
        
        results_comparison.append(config_results)
    
    # Generate comprehensive comparison report
    print(f"\n📊 COMPREHENSIVE ORIGINAL vs CUSTOM COMPARISON")
    print("="*100)
    
    # Summary table header
    print(f"{'Configuration':<20} {'Dataset':<10} {'Stocks':<8} {'Mean':<10} {'Max':<10} {'Min':<10} {'Std':<10} {'Top Stock':<15} {'Score':<8}")
    print("-" * 100)
    
    for result in results_comparison:
        config_name = result['config_name']
        
        # Original data row
        if result['original']:
            orig = result['original']
            print(f"{config_name:<20} {'Original':<10} {orig['valid_stocks']:<8} {orig['mean_prediction']:<10.2f} {orig['max_prediction']:<10.2f} {orig['min_prediction']:<10.2f} {orig['std_prediction']:<10.2f} {orig['top_stock']:<15} {orig['top_score']:<8.2f}")
        
        # Custom data row
        if result['custom']:
            custom = result['custom']
            print(f"{'':<20} {'Custom':<10} {custom['valid_stocks']:<8} {custom['mean_prediction']:<10.2f} {custom['max_prediction']:<10.2f} {custom['min_prediction']:<10.2f} {custom['std_prediction']:<10.2f} {custom['top_stock']:<15} {custom['top_score']:<8.2f}")
        
        # Comparison metrics
        if result['original'] and result['custom']:
            orig, custom = result['original'], result['custom']
            mean_diff = orig['mean_prediction'] - custom['mean_prediction']
            mean_pct = (mean_diff / custom['mean_prediction'] * 100) if custom['mean_prediction'] != 0 else 0
            
            max_diff = orig['max_prediction'] - custom['max_prediction']
            max_pct = (max_diff / custom['max_prediction'] * 100) if custom['max_prediction'] != 0 else 0
            
            print(f"{'  → Difference':<20} {'Orig-Cust':<10} {'':<8} {mean_diff:<10.2f} {max_diff:<10.2f} {'':<10} {'':<10} {'':<15} {'':<8}")
            print(f"{'  → % Change':<20} {'(Orig/Cust)':<10} {'':<8} {mean_pct:<10.1f}% {max_pct:<10.1f}% {'':<10} {'':<10} {'':<15} {'':<8}")
        
        print("-" * 100)
    
    # Statistical analysis
    print(f"\n📈 STATISTICAL ANALYSIS")
    print("="*50)
    
    for result in results_comparison:
        if result['original'] and result['custom']:
            print(f"\n🔍 {result['config_name']}:")
            orig, custom = result['original'], result['custom']
            
            # Dataset size comparison
            print(f"   Dataset sizes: Original({orig['valid_stocks']}) vs Custom({custom['valid_stocks']})")
            
            # Prediction distribution comparison
            print(f"   Prediction ranges:")
            print(f"     Original: {orig['min_prediction']:.2f} to {orig['max_prediction']:.2f} (span: {orig['max_prediction'] - orig['min_prediction']:.2f})")
            print(f"     Custom:   {custom['min_prediction']:.2f} to {custom['max_prediction']:.2f} (span: {custom['max_prediction'] - custom['min_prediction']:.2f})")
            
            # Top stock comparison
            if orig['top_stock'] == custom['top_stock']:
                print(f"   🎯 Same top stock identified: {orig['top_stock']}")
            else:
                print(f"   🔄 Different top stocks: {orig['top_stock']} vs {custom['top_stock']}")
            
            # Performance metrics
            score_diff = orig['top_score'] - custom['top_score']
            print(f"   📊 Top score difference: {score_diff:+.2f} (Original: {orig['top_score']:.2f}, Custom: {custom['top_score']:.2f})")

if __name__ == "__main__":
    run_original_vs_custom_comparison()