"""
Advanced comparison with different model configurations
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

from stock_inference import StockSelectionInference

def run_advanced_comparison():
    """Run advanced comparison with different configurations."""
    
    print("🚀 ADVANCED INFERENCE CONFIGURATION COMPARISON")
    print("="*80)
    
    model_path = "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    large_data = "../ml-model/data-example/NASDAQ_all_features_large.pkl"   # 10 stocks
    
    if not os.path.exists(model_path) or not os.path.exists(large_data):
        print("❌ Required files not found")
        return
    
    # Test different configurations
    configurations = [
        {"name": "Standard Configuration", "top_k": [5], "seq_len": 32, "features": "all"},
        {"name": "Conservative Selection", "top_k": [3], "seq_len": 32, "features": "all"},
        {"name": "Aggressive Selection", "top_k": [8], "seq_len": 32, "features": "all"},
        {"name": "Close-Only Features", "top_k": [5], "seq_len": 32, "features": "close_only"},
    ]
    
    results_table = []
    
    for config in configurations:
        print(f"\n🔧 TESTING: {config['name']}")
        print("="*50)
        
        try:
            # Create custom config for close_only features
            if config['features'] == 'close_only':
                model_config = {
                    'input_size': 5,  # Only 5 features for close_only
                    'num_class': 1,
                    'hidden_size': 128,
                    'num_feat_att_layers': 1,
                    'num_pre_att_layers': 1,
                    'num_heads': 4,
                    'days': 32,
                    'dropout': 0.1,
                    'market_name': 'NASDAQ',
                    'feature_describe': 'close_only'
                }
                inference = StockSelectionInference(model_path, config=model_config)
            else:
                inference = StockSelectionInference(model_path)
            
            # Run inference
            results = inference.run_inference(
                data_source=large_data,
                top_k=config['top_k'],
                sequence_length=config['seq_len'],
                feature_describe=config['features']
            )
            
            # Extract key metrics
            predictions = results['raw_predictions']['values']
            valid_predictions = [p for p, v in zip(predictions, results['raw_predictions']['valid_mask']) if v > 0.5]
            
            top_k_key = f"top_{config['top_k'][0]}"
            if top_k_key in results['portfolio_selections']:
                top_selection = results['portfolio_selections'][top_k_key]
                top_stock = top_selection['symbols'][0] if top_selection['symbols'] else "N/A"
                top_score = top_selection['predictions'][0] if top_selection['predictions'] else 0
            else:
                top_stock = "N/A"
                top_score = 0
            
            result_summary = {
                'Configuration': config['name'],
                'Top K': config['top_k'][0],
                'Features': config['features'],
                'Total Stocks': results['data_info']['total_stocks'],
                'Valid Stocks': results['data_info']['valid_stocks'],
                'Mean Prediction': np.mean(valid_predictions),
                'Max Prediction': np.max(valid_predictions),
                'Min Prediction': np.min(valid_predictions),
                'Std Deviation': np.std(valid_predictions),
                'Top Stock': top_stock,
                'Top Score': top_score
            }
            
            results_table.append(result_summary)
            
            print(f"✅ {config['name']} completed")
            print(f"   Top stock: {top_stock} (score: {top_score:.2f})")
            print(f"   Mean prediction: {np.mean(valid_predictions):.2f}")
            print(f"   Prediction range: {np.min(valid_predictions):.2f} to {np.max(valid_predictions):.2f}")
            
        except Exception as e:
            print(f"❌ Error with {config['name']}: {str(e)}")
            continue
    
    # Print comprehensive comparison table
    if results_table:
        print(f"\n📊 COMPREHENSIVE CONFIGURATION COMPARISON")
        print("="*100)
        
        # Header
        print(f"{'Configuration':<25} {'Top K':<6} {'Features':<10} {'Stocks':<7} {'Mean Pred':<10} {'Max Pred':<10} {'Top Stock':<18} {'Top Score':<10}")
        print("-" * 100)
        
        # Data rows
        for result in results_table:
            print(f"{result['Configuration']:<25} "
                  f"{result['Top K']:<6} "
                  f"{result['Features']:<10} "
                  f"{result['Valid Stocks']:<7} "
                  f"{result['Mean Prediction']:<10.2f} "
                  f"{result['Max Prediction']:<10.2f} "
                  f"{result['Top Stock']:<18} "
                  f"{result['Top Score']:<10.2f}")
        
        # Analysis
        print(f"\n📈 ANALYSIS:")
        print("-" * 50)
        
        # Find best performing configuration
        best_config = max(results_table, key=lambda x: x['Max Prediction'])
        worst_config = min(results_table, key=lambda x: x['Max Prediction'])
        
        print(f"🏆 Best performing: {best_config['Configuration']}")
        print(f"   Max prediction: {best_config['Max Prediction']:.2f}")
        print(f"   Top stock: {best_config['Top Stock']}")
        
        print(f"⚠️  Lowest performing: {worst_config['Configuration']}")
        print(f"   Max prediction: {worst_config['Max Prediction']:.2f}")
        print(f"   Top stock: {worst_config['Top Stock']}")
        
        # Feature comparison
        all_features = [r for r in results_table if r['Features'] == 'all']
        close_features = [r for r in results_table if r['Features'] == 'close_only']
        
        if all_features and close_features:
            all_mean = np.mean([r['Mean Prediction'] for r in all_features])
            close_mean = np.mean([r['Mean Prediction'] for r in close_features])
            
            print(f"\n🔍 Feature Set Analysis:")
            print(f"   All features mean: {all_mean:.2f}")
            print(f"   Close-only mean: {close_mean:.2f}")
            print(f"   Difference: {all_mean - close_mean:.2f} ({((all_mean - close_mean) / close_mean * 100):+.1f}%)")

if __name__ == "__main__":
    run_advanced_comparison()