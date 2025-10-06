"""
Example Usage of Stock Selection Inference
==========================================

This script demonstrates how to use the stock_inference.py module for 
predicting stock performance and selecting top-performing stocks.

Run this script after training your model to see inference in action.
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

# Import the inference class
try:
    from stock_inference import StockSelectionInference, print_inference_results
except ImportError:
    print("❌ Could not import stock_inference.py. Make sure it's in the same directory.")
    sys.exit(1)


def example_1_basic_inference():
    """Example 1: Basic inference with default settings."""
    print("\n" + "="*80)
    print("📝 EXAMPLE 1: Basic Inference")
    print("="*80)
    
    # Check if required files exist
    model_path = "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    data_path = "../ml-model/data-example/NASDAQ_all_features.pkl"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("   Please ensure NASDAQ_all_features.pkl exists in the ./data/ directory")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please train a model first or adjust the model_path")
        return None
    
    try:
        # Initialize inference pipeline
        print("🔧 Initializing inference pipeline...")
        inference = StockSelectionInference(model_path)
        
        # Run complete inference
        print("🚀 Running inference...")
        results = inference.run_inference(data_path)
        
        # Print results
        print_inference_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Example 1 failed: {str(e)}")
        return None


def example_2_custom_configuration():
    """Example 2: Custom configuration and step-by-step processing."""
    print("\n" + "="*80)
    print("📝 EXAMPLE 2: Custom Configuration")
    print("="*80)
    
    model_path = r"../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    data_path = "../ml-model/data-example/NASDAQ_all_features.pkl"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("❌ Required files not found. Skipping this example.")
        return None
    
    try:
        # Custom configuration - KEEP same architecture as training
        # Note: Can't change 'days' for pre-trained model due to positional encoding
        custom_config = {
            'input_size': 25,
            'num_class': 1,
            'hidden_size': 128,
            'num_feat_att_layers': 1,
            'num_pre_att_layers': 1,
            'num_heads': 4,
            'days': 32,  # Must match training (32 days)
            'dropout': 0.1,
            'market_name': 'NASDAQ',
            'feature_describe': 'all'
        }
        
        print("🔧 Initializing with custom config...")
        print(f"   - Sequence length: {custom_config['days']} days (must match training)")
        print("   - Note: Pre-trained models can't change sequence length")
        
        inference = StockSelectionInference(model_path, custom_config)
        
        # Step-by-step processing
        print("📊 Step 1: Preprocessing data...")
        input_tensor, valid_mask, symbols = inference.preprocess_data(
            data_path,
            sequence_length=32,  # Use same as training
            feature_describe='all'
        )
        
        print("🤖 Step 2: Running predictions...")
        predictions = inference.predict(input_tensor, valid_mask)
        
        print("🏆 Step 3: Selecting top stocks...")
        portfolios = inference.select_top_stocks(
            predictions, valid_mask, symbols, top_k=[3, 7, 15]
        )
        
        # Display results
        print("\n📈 CUSTOM INFERENCE RESULTS:")
        for portfolio_key, portfolio in portfolios.items():
            k = portfolio['top_k']
            print(f"\n🏆 Top {k} stocks:")
            for i, (symbol, pred, weight) in enumerate(zip(
                portfolio['symbols'], portfolio['predictions'], portfolio['weights']
            )):
                print(f"   {i+1}. {symbol:<6} | Score: {pred:7.4f} | Weight: {weight:6.2%}")
        
        return portfolios
        
    except Exception as e:
        print(f"❌ Example 2 failed: {str(e)}")
        return None


def example_3_programmatic_usage():
    """Example 3: Programmatic usage for integration."""
    print("\n" + "="*80)
    print("📝 EXAMPLE 3: Programmatic Integration")
    print("="*80)
    
    model_path = r"../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    data_path = "../ml-model/data-example/NASDAQ_all_features.pkl"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("❌ Required files not found. Skipping this example.")
        return None
    
    try:
        # Initialize once (would be done at startup in production)
        inference = StockSelectionInference(model_path)
        
        # Simulate daily trading workflow
        print("📅 Simulating daily trading workflow...")
        
        # Run inference
        results = inference.run_inference(
            data_source=data_path,
            top_k=[5, 10],
            sequence_length=32
        )
        
        # Extract actionable information
        top_5_portfolio = results['portfolio_selections']['top_5']
        top_10_portfolio = results['portfolio_selections']['top_10']
        
        print("💼 TRADING SIGNALS:")
        print(f"   📊 Analysis timestamp: {results['timestamp']}")
        print(f"   📈 Market: {results['model_config']['market_name']}")
        print(f"   🔢 Valid stocks analyzed: {results['data_info']['valid_stocks']}")
        
        print("\n🎯 RECOMMENDED PORTFOLIO (Top 5):")
        for i, (symbol, weight, prediction) in enumerate(zip(
            top_5_portfolio['symbols'],
            top_5_portfolio['weights'],
            top_5_portfolio['predictions']
        )):
            print(f"   {i+1}. {symbol:<6} - Allocate: {weight:6.2%} (Score: {prediction:6.3f})")
        
        # Risk assessment
        all_predictions = [p for p, v in zip(
            results['raw_predictions']['values'],
            results['raw_predictions']['valid_mask']
        ) if v > 0.5]
        
        mean_pred = np.mean(all_predictions)
        std_pred = np.std(all_predictions)
        
        print(f"\n📊 MARKET SENTIMENT:")
        print(f"   Mean prediction: {mean_pred:6.3f}")
        print(f"   Volatility (std): {std_pred:6.3f}")
        
        if mean_pred > 0.1:
            print("   🟢 Market outlook: BULLISH")
        elif mean_pred < -0.1:
            print("   🔴 Market outlook: BEARISH")
        else:
            print("   🟡 Market outlook: NEUTRAL")
        
        # Confidence assessment
        top_5_scores = top_5_portfolio['predictions']
        confidence = np.mean(top_5_scores) / std_pred if std_pred > 0 else 0
        
        print(f"   🎯 Selection confidence: {confidence:6.3f}")
        if confidence > 1.0:
            print("   ✅ HIGH confidence in top selections")
        elif confidence > 0.5:
            print("   ⚠️  MEDIUM confidence in top selections")
        else:
            print("   🔴 LOW confidence - consider market timing")
        
        return {
            'portfolio': top_5_portfolio,
            'market_sentiment': mean_pred,
            'volatility': std_pred,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"❌ Example 3 failed: {str(e)}")
        return None


def example_4_feature_comparison():
    """Example 4: Compare different feature sets."""
    print("\n" + "="*80)
    print("📝 EXAMPLE 4: Feature Set Comparison")
    print("="*80)
    
    model_path = r"../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    data_path = "../ml-model/data-example/NASDAQ_all_features.pkl"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("❌ Required files not found. Skipping this example.")
        return None
    
    try:
        inference = StockSelectionInference(model_path)
        
        # Test different top-k values instead of feature sets
        # Note: Model was trained with all 25 features, so we can only use 'all'
        print("🔍 Testing different top-k selections (model trained with all features)")
        
        top_k_values = [3, 5, 10, 20]
        comparison_results = {}
        
        for k in top_k_values:
            print(f"\n🔍 Testing top-{k} selection:")
            
            results = inference.run_inference(
                data_source=data_path,
                top_k=[k],
                feature_describe='all'
            )
            
            top_k_result = results['portfolio_selections'][f'top_{k}']
            comparison_results[f'top_{k}'] = {
                'symbols': top_k_result['symbols'],
                'predictions': top_k_result['predictions'],
                'mean_score': np.mean(top_k_result['predictions']),
                'min_score': np.min(top_k_result['predictions']),
                'max_score': np.max(top_k_result['predictions'])
            }
            
            print(f"   Top {k} stocks: {top_k_result['symbols']}")
            print(f"   Score range: {np.min(top_k_result['predictions']):.3f} to {np.max(top_k_result['predictions']):.3f}")
        
        # Compare results
        print("\n📊 TOP-K COMPARISON:")
        for k_key, result in comparison_results.items():
            k = k_key.replace('top_', '')
            print(f"\n🏆 TOP-{k} SELECTION:")
            print(f"   Best stock: {result['symbols'][0]} (score: {result['max_score']:.3f})")
            print(f"   Worst in top-{k}: {result['symbols'][-1]} (score: {result['min_score']:.3f})")
            print(f"   Average score: {result['mean_score']:.3f}")
        
        return comparison_results
        
    except Exception as e:
        print(f"❌ Example 4 failed: {str(e)}")
        return None


def main():
    """Run all examples."""
    print("🚀 STOCK SELECTION INFERENCE EXAMPLES")
    print("="*80)
    print("This script demonstrates various ways to use the stock selection model.")
    print("Make sure you have:")
    print("  ✅ Trained model checkpoint (./models/model_tt_50.ckpt)")
    print("  ✅ NASDAQ data file (./data/NASDAQ_all_features.pkl)")
    print("  ✅ Required Python packages (torch, numpy, etc.)")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("  ⚠️  CUDA not available - using CPU")
    
    # Run examples
    examples = [
        ("Basic Inference", example_1_basic_inference),
        ("Custom Configuration", example_2_custom_configuration),
        ("Programmatic Usage", example_3_programmatic_usage),
        ("Feature Comparison", example_4_feature_comparison)
    ]
    
    results = {}
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} Running {name} {'='*20}")
            result = example_func()
            results[name] = result
        except Exception as e:
            print(f"❌ {name} failed with error: {str(e)}")
            results[name] = None
    
    # Summary
    print("\n" + "="*80)
    print("📋 EXECUTION SUMMARY")
    print("="*80)
    
    for name, result in results.items():
        if result is not None:
            print(f"  ✅ {name}: SUCCESS")
        else:
            print(f"  ❌ {name}: FAILED")
    
    print("\n🎯 Next Steps:")
    print("  1. Integrate this code into your trading system")
    print("  2. Set up automated daily inference")
    print("  3. Add risk management and position sizing")
    print("  4. Monitor model performance over time")
    print("  5. Retrain model periodically with new data")


if __name__ == "__main__":
    main()