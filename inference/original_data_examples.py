"""
Run detailed inference examples with original NASDAQ data
"""

import sys
import pathlib

# Add the current directory to the path for importing stock_inference
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stock_inference import StockSelectionInference

def original_data_inference():
    """Run inference with original NASDAQ data and show detailed results."""
    
    print("🚀 ORIGINAL NASDAQ DATA INFERENCE EXAMPLES")
    print("="*80)
    
    model_path = "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    original_data = "../ml-model/data-example/NASDAQ_all_features.pkl"
    
    print("📊 Example 1: Multi-K Stock Selection with Original Data")
    print("-" * 50)
    
    try:
        # Initialize inference with original data
        inference = StockSelectionInference(model_path)
        
        # Run comprehensive inference
        results = inference.run_inference(
            data_source=original_data,
            top_k=[3, 5, 10, 20],  # Multiple K values
            sequence_length=32,
            feature_describe='all'
        )
        
        print(f"🎯 DATA SUMMARY:")
        print(f"   Total stocks analyzed: {results['data_info']['total_stocks']}")
        print(f"   Valid stocks: {results['data_info']['valid_stocks']}")
        print(f"   Data shape: {results['data_info']['preprocessed_shape']}")
        
        print(f"\n📈 PREDICTION STATISTICS:")
        predictions = results['raw_predictions']['values']
        valid_mask = results['raw_predictions']['valid_mask']
        valid_predictions = [p for p, v in zip(predictions, valid_mask) if v > 0.5]
        
        print(f"   Mean prediction: {sum(valid_predictions)/len(valid_predictions):.4f}")
        print(f"   Max prediction: {max(valid_predictions):.4f}")
        print(f"   Min prediction: {min(valid_predictions):.4f}")
        print(f"   Standard deviation: {(sum([(p - sum(valid_predictions)/len(valid_predictions))**2 for p in valid_predictions])/len(valid_predictions))**0.5:.4f}")
        
        print(f"\n🏆 TOP STOCK SELECTIONS:")
        for k_value in [3, 5, 10, 20]:
            key = f"top_{k_value}"
            if key in results['portfolio_selections']:
                selection = results['portfolio_selections'][key]
                print(f"\n   📊 Top {k_value} Stocks:")
                for i, (symbol, score) in enumerate(zip(selection['symbols'], selection['predictions'])):
                    print(f"      {i+1:2d}. {symbol:<8} (Score: {score:8.4f})")
        
        print(f"\n🔍 BOTTOM 10 PERFORMERS:")
        # Get all predictions with symbols
        stock_predictions = []
        for i, (pred, valid) in enumerate(zip(predictions, valid_mask)):
            if valid > 0.5:
                # We need to map index back to stock symbol - this is simplified
                stock_predictions.append((f"Stock_{i}", pred))
        
        # Sort and show bottom 10
        stock_predictions.sort(key=lambda x: x[1])
        for i, (symbol, score) in enumerate(stock_predictions[:10]):
            print(f"      {i+1:2d}. {symbol:<12} (Score: {score:8.4f})")
        
        print(f"\n📊 PREDICTION DISTRIBUTION:")
        # Create simple histogram
        ranges = [(-30, -20), (-20, -10), (-10, 0), (0, 5), (5, 10), (10, 15)]
        for min_val, max_val in ranges:
            count = sum(1 for p in valid_predictions if min_val <= p < max_val)
            percentage = count / len(valid_predictions) * 100
            bar = "█" * int(percentage / 2)  # Scale bar
            print(f"   {min_val:3d} to {max_val:3d}: {count:4d} stocks ({percentage:5.1f}%) {bar}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def compare_top_stocks():
    """Compare top stocks from original vs custom data."""
    
    print(f"\n🆚 TOP STOCKS COMPARISON")
    print("="*50)
    
    model_path = "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt"
    original_data = "../ml-model/data-example/NASDAQ_all_features.pkl"
    custom_data = "../ml-model/data-example/NASDAQ_all_features_custom.pkl"
    
    try:
        # Original data top 5
        inference_orig = StockSelectionInference(model_path)
        results_orig = inference_orig.run_inference(
            data_source=original_data,
            top_k=[5],
            sequence_length=32,
            feature_describe='all'
        )
        
        print("🥇 ORIGINAL DATA - TOP 5 STOCKS:")
        orig_top5 = results_orig['portfolio_selections']['top_5']
        for i, (symbol, score) in enumerate(zip(orig_top5['symbols'], orig_top5['predictions'])):
            print(f"   {i+1}. {symbol:<8} (Score: {score:8.4f})")
        
        # Custom data top 5 (actually top 3 since only 3 stocks)
        inference_custom = StockSelectionInference(model_path)
        results_custom = inference_custom.run_inference(
            data_source=custom_data,
            top_k=[3],
            sequence_length=32,
            feature_describe='all'
        )
        
        print("\n🥈 CUSTOM DATA - ALL 3 STOCKS:")
        custom_top3 = results_custom['portfolio_selections']['top_3']
        for i, (symbol, score) in enumerate(zip(custom_top3['symbols'], custom_top3['predictions'])):
            print(f"   {i+1}. {symbol:<20} (Score: {score:8.4f})")
        
        print(f"\n📊 SCORE COMPARISON:")
        print(f"   Original data highest score: {max(orig_top5['predictions']):.4f}")
        print(f"   Custom data highest score:   {max(custom_top3['predictions']):.4f}")
        print(f"   Difference: {max(custom_top3['predictions']) - max(orig_top5['predictions']):+.4f}")
        print(f"   Custom is {max(custom_top3['predictions']) / max(orig_top5['predictions']):.1f}x higher")
        
    except Exception as e:
        print(f"❌ Error in comparison: {str(e)}")

if __name__ == "__main__":
    original_data_inference()
    compare_top_stocks()