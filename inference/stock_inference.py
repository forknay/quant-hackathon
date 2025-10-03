"""
NASDAQ Stock Selection Model - Inference Pipeline
=================================================

This file provides a complete inference pipeline for the pre-trained TransformerStockPrediction model.
It includes precise data processing instructions and expected output formats.

Requirements:
- Trained model checkpoint (.ckpt file)
- Input data in the same format as training data
- PyTorch with CUDA support (optional but recommended)

Usage Example:
    python stock_inference.py --model_path "path/to/model.ckpt" --data_path "path/to/data.pkl"
"""

import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import sys
import pathlib

# Add the ml-model directory to the path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ML_MODEL_PATH = PROJECT_ROOT / "ml-model"
if str(ML_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_PATH))

# Import model and utilities from ml-model directory
try:
    from model import TransformerStockPrediction
    from utils import get_args
    from metrics import return_eval
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure model.py, utils.py, and metrics.py are in {ML_MODEL_PATH}")
    sys.exit(1)


class StockSelectionInference:
    """
    Complete inference pipeline for stock selection using the trained Transformer model.
    
    This class handles:
    1. Data preprocessing and validation
    2. Model loading and initialization
    3. Batch inference
    4. Top-K stock selection
    5. Performance metrics calculation
    """
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path (str): Path to the trained model checkpoint (.ckpt file)
            config (dict, optional): Model configuration. If None, uses default NASDAQ config.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Default configuration for NASDAQ model
        if config is None:
            config = {
                'input_size': 25,           # Features per time step (excluding day index)
                'num_class': 1,             # Single regression output
                'hidden_size': 128,         # Transformer hidden dimension
                'num_feat_att_layers': 1,   # Feature attention layers
                'num_pre_att_layers': 1,    # Prediction attention layers  
                'num_heads': 4,             # Multi-head attention heads
                'days': 32,                 # Sequence length (lookback window)
                'dropout': 0.1,             # Dropout rate
                'market_name': 'NASDAQ',    # Market identifier
                'feature_describe': 'all'   # Use all 25 features
            }
        
        self.config = config
        self.model = None
        self.model_path = model_path
        
        # Load the model
        self._load_model()
        
        print("✅ StockSelectionInference initialized successfully!")
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        try:
            print(f"📂 Loading model from: {self.model_path}")
            
            # Initialize model architecture
            self.model = TransformerStockPrediction(
                input_size=self.config['input_size'],
                num_class=self.config['num_class'],
                hidden_size=self.config['hidden_size'],
                num_feat_att_layers=self.config['num_feat_att_layers'],
                num_pre_att_layers=self.config['num_pre_att_layers'],
                num_heads=self.config['num_heads'],
                days=self.config['days'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Filter out pre-training layers that aren't needed for inference
            if isinstance(checkpoint, dict):
                # Remove pre-training specific layers
                pretrain_keys = [
                    'pretrain_outlayers.stock.weight', 'pretrain_outlayers.stock.bias',
                    'pretrain_outlayers.sector.weight', 'pretrain_outlayers.sector.bias', 
                    'pretrain_outlayers.mask_avg_price.weight', 'pretrain_outlayers.mask_avg_price.bias'
                ]
                
                filtered_checkpoint = {k: v for k, v in checkpoint.items() if k not in pretrain_keys}
                
                print(f"🔍 Filtered out {len(checkpoint) - len(filtered_checkpoint)} pre-training layers")
                print(f"   - Original keys: {len(checkpoint)}")
                print(f"   - Filtered keys: {len(filtered_checkpoint)}")
                
                # Load the filtered state dict
                self.model.load_state_dict(filtered_checkpoint, strict=False)
            else:
                # Fallback for direct model loading
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            print(f"   - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   - Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def preprocess_data(self, data_source: Union[str, Dict], 
                       sequence_length: int = 32,
                       feature_describe: str = 'all') -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Preprocess input data for model inference.
        
        Data Input Formats Supported:
        1. Path to NASDAQ_all_features.pkl file (str)
        2. Dictionary with preprocessed data (Dict)
        3. Raw market data dictionary (Dict)
        
        Args:
            data_source: Input data (file path or dictionary)
            sequence_length: Number of historical days to use (default: 32)
            feature_describe: Feature set to use ('all' or 'close_only')
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[str]]: 
                - input_tensor: [num_stocks, sequence_length, num_features]
                - valid_mask: [num_stocks] - 1.0 for valid stocks, 0.0 for invalid
                - stock_symbols: List of stock symbols
        """
        print("🔄 Preprocessing data for inference...")
        
        if isinstance(data_source, str):
            # Load from pickle file
            return self._preprocess_from_pickle(data_source, sequence_length, feature_describe)
        elif isinstance(data_source, dict):
            # Process dictionary data
            return self._preprocess_from_dict(data_source, sequence_length, feature_describe)
        else:
            raise ValueError("data_source must be a file path (str) or dictionary")
    
    def _preprocess_from_pickle(self, pickle_path: str, sequence_length: int, 
                              feature_describe: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Preprocess data from NASDAQ_all_features.pkl file."""
        try:
            with open(pickle_path, 'rb') as fr:
                data = pkl.load(fr)
            
            all_stock_features = data['all_features']
            
            print(f"📊 Loaded data for {len(all_stock_features)} stocks")
            
            # Convert to the same format as training data
            # Remove day index (column 0) and last row, keep features 1-25
            all_features = [all_stock_features[stock][:-1, 1:] for stock in all_stock_features.keys()]
            all_features = np.stack(all_features)  # [num_stocks, num_days, 25]
            
            stock_symbols = list(all_stock_features.keys())
            
            # Select feature dimensions
            if feature_describe == 'all':
                feature_dim = list(range(25))  # All 25 features
            elif feature_describe == 'close_only':
                feature_dim = [15, 16, 17, 18, 19]  # Close-related features only
            else:
                feature_dim = feature_describe
            
            # Use the most recent sequence_length days for inference
            if all_features.shape[1] < sequence_length:
                raise ValueError(f"Not enough historical data. Need {sequence_length} days, got {all_features.shape[1]}")
            
            # Take the last sequence_length days
            recent_data = all_features[:, -sequence_length:, feature_dim]  # [num_stocks, seq_len, num_features]
            
            # Create validity mask (check for missing data marked as -1234)
            valid_mask = np.ones(len(stock_symbols))
            for i, stock_data in enumerate(recent_data):
                if np.any(stock_data == -1234):
                    valid_mask[i] = 0.0
                    print(f"⚠️  Stock {stock_symbols[i]} has missing data, will be excluded")
            
            # Convert to tensors
            input_tensor = torch.FloatTensor(recent_data).to(self.device)
            valid_mask_tensor = torch.FloatTensor(valid_mask).to(self.device)
            
            print(f"✅ Preprocessed data shape: {input_tensor.shape}")
            print(f"   - Valid stocks: {int(valid_mask.sum())}/{len(stock_symbols)}")
            
            return input_tensor, valid_mask_tensor, stock_symbols
            
        except Exception as e:
            print(f"❌ Error preprocessing pickle data: {str(e)}")
            raise
    
    def _preprocess_from_dict(self, data_dict: Dict, sequence_length: int, 
                            feature_describe: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Preprocess data from dictionary format."""
        print("📝 Processing dictionary data (custom implementation needed)")
        # This would be implemented based on your specific dictionary format
        raise NotImplementedError("Dictionary preprocessing not implemented. Use pickle file format.")
    
    def predict(self, input_tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Run model inference on preprocessed data.
        
        Args:
            input_tensor: [num_stocks, sequence_length, num_features]
            valid_mask: [num_stocks] - validity mask
            
        Returns:
            torch.Tensor: Predictions [num_stocks] - normalized price predictions
        """
        print("🤖 Running model inference...")
        
        self.model.eval()
        with torch.no_grad():
            # Ensure model is in stock selection mode (not pre-training)
            self.model.pretrain_task = ''
            
            # Run inference
            predictions = self.model(input_tensor)  # [num_stocks, 1]
            predictions = predictions.squeeze()     # [num_stocks]
            
            # Apply validity mask
            masked_predictions = predictions * valid_mask
            
        print(f"✅ Inference completed. Predictions shape: {predictions.shape}")
        return masked_predictions
    
    def select_top_stocks(self, predictions: torch.Tensor, valid_mask: torch.Tensor, 
                         stock_symbols: List[str], top_k: List[int] = None) -> Dict:
        """
        Select top-K stocks based on model predictions.
        
        Args:
            predictions: [num_stocks] - Model predictions
            valid_mask: [num_stocks] - Validity mask  
            stock_symbols: List of stock symbols
            top_k: List of K values (default: [1, 5, 10])
            
        Returns:
            Dict: Portfolio selections for each K value
        """
        if top_k is None:
            top_k = [1, 5, 10]
        
        print(f"📈 Selecting top stocks for K values: {top_k}")
        
        # Mask invalid stocks with -inf
        masked_predictions = predictions.clone()
        masked_predictions[valid_mask == 0] = -float('inf')
        
        portfolio_selections = {}
        
        for k in top_k:
            # Get top-k indices
            top_indices = torch.argsort(masked_predictions, descending=True)[:k]
            
            # Filter out invalid stocks
            valid_top_indices = []
            for idx in top_indices:
                if valid_mask[idx] > 0.5:  # Valid stock
                    valid_top_indices.append(idx.item())
                if len(valid_top_indices) == k:
                    break
            
            # Create portfolio
            portfolio = {
                'top_k': k,
                'selected_stocks': [],
                'predictions': [],
                'symbols': [],
                'weights': []
            }
            
            total_score = 0
            for idx in valid_top_indices:
                score = predictions[idx].item()
                portfolio['selected_stocks'].append(idx)
                portfolio['predictions'].append(score)
                portfolio['symbols'].append(stock_symbols[idx])
                total_score += max(score, 0)  # Avoid negative weights
            
            # Calculate weights (proportional to prediction scores)
            if total_score > 0:
                weights = [max(score, 0) / total_score for score in portfolio['predictions']]
            else:
                # Equal weights if all predictions are negative/zero
                weights = [1.0 / len(valid_top_indices)] * len(valid_top_indices)
            
            portfolio['weights'] = weights
            portfolio_selections[f'top_{k}'] = portfolio
        
        return portfolio_selections
    
    def run_inference(self, data_source: Union[str, Dict], 
                     top_k: List[int] = None,
                     sequence_length: int = None,
                     feature_describe: str = None) -> Dict:
        """
        Complete inference pipeline: preprocess → predict → select stocks.
        
        Args:
            data_source: Input data (file path or dictionary)
            top_k: List of K values for top-K selection (default: [1, 5, 10])
            sequence_length: Sequence length (default: from config)
            feature_describe: Feature set (default: from config)
            
        Returns:
            Dict: Complete inference results with portfolio selections
        """
        # Use config defaults if not specified
        if sequence_length is None:
            sequence_length = self.config['days']
        if feature_describe is None:
            feature_describe = self.config['feature_describe']
        if top_k is None:
            top_k = [1, 5, 10]
        
        print("🚀 Starting complete inference pipeline...")
        print(f"   - Sequence length: {sequence_length}")
        print(f"   - Feature set: {feature_describe}")
        print(f"   - Top-K values: {top_k}")
        
        # Step 1: Preprocess data
        input_tensor, valid_mask, stock_symbols = self.preprocess_data(
            data_source, sequence_length, feature_describe
        )
        
        # Step 2: Run predictions
        predictions = self.predict(input_tensor, valid_mask)
        
        # Step 3: Select top stocks
        portfolio_selections = self.select_top_stocks(
            predictions, valid_mask, stock_symbols, top_k
        )
        
        # Step 4: Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_config': self.config,
            'data_info': {
                'total_stocks': len(stock_symbols),
                'valid_stocks': int(valid_mask.sum().item()),
                'sequence_length': sequence_length,
                'feature_set': feature_describe
            },
            'raw_predictions': {
                'values': predictions.cpu().numpy().tolist(),
                'symbols': stock_symbols,
                'valid_mask': valid_mask.cpu().numpy().tolist()
            },
            'portfolio_selections': portfolio_selections,
            'model_path': self.model_path
        }
        
        print("✅ Inference pipeline completed successfully!")
        return results


def print_inference_results(results: Dict):
    """Pretty print inference results."""
    print("\n" + "="*80)
    print("📊 STOCK SELECTION INFERENCE RESULTS")
    print("="*80)
    
    # Data info
    data_info = results['data_info']
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"📈 Market: {results['model_config']['market_name']}")
    print(f"🔢 Total stocks analyzed: {data_info['total_stocks']}")
    print(f"✅ Valid stocks: {data_info['valid_stocks']}")
    print(f"📊 Sequence length: {data_info['sequence_length']} days")
    print(f"🏷️  Feature set: {data_info['feature_set']}")
    print()
    
    # Portfolio selections
    portfolio_selections = results['portfolio_selections']
    for portfolio_key, portfolio in portfolio_selections.items():
        k = portfolio['top_k']
        print(f"🏆 TOP {k} STOCK SELECTION:")
        print("-" * 40)
        
        for i, (symbol, pred, weight) in enumerate(zip(
            portfolio['symbols'], portfolio['predictions'], portfolio['weights']
        )):
            print(f"  {i+1:2d}. {symbol:<6} | Prediction: {pred:8.4f} | Weight: {weight:6.2%}")
        
        print()
    
    # Summary statistics
    all_preds = results['raw_predictions']['values']
    valid_preds = [p for p, v in zip(all_preds, results['raw_predictions']['valid_mask']) if v > 0.5]
    
    print("📊 PREDICTION STATISTICS:")
    print(f"   Mean prediction: {np.mean(valid_preds):8.4f}")
    print(f"   Std prediction:  {np.std(valid_preds):8.4f}")
    print(f"   Min prediction:  {np.min(valid_preds):8.4f}")
    print(f"   Max prediction:  {np.max(valid_preds):8.4f}")
    print()


def main():
    """Command-line interface for stock selection inference."""
    parser = argparse.ArgumentParser(description='NASDAQ Stock Selection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.ckpt file)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NASDAQ_all_features.pkl file')
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 5, 10],
                       help='Top-K values for stock selection (default: 1 5 10)')
    parser.add_argument('--sequence_length', type=int, default=32,
                       help='Sequence length for inference (default: 32)')
    parser.add_argument('--feature_describe', type=str, default='all',
                       choices=['all', 'close_only'],
                       help='Feature set to use (default: all)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Save results to JSON file (optional)')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference pipeline
        inference = StockSelectionInference(
            model_path=args.model_path,
            config={'days': args.sequence_length, 'feature_describe': args.feature_describe}
        )
        
        # Run inference
        results = inference.run_inference(
            data_source=args.data_path,
            top_k=args.top_k,
            sequence_length=args.sequence_length,
            feature_describe=args.feature_describe
        )
        
        # Print results
        print_inference_results(results)
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {args.output_file}")
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
USAGE EXAMPLES:

1. Basic inference with default settings:
   python stock_inference.py --model_path "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt" --data_path "../ml-model/data-example/NASDAQ_all_features.pkl"

2. Custom top-K selection:
   python stock_inference.py --model_path "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt" --data_path "../ml-model/data-example/NASDAQ_all_features.pkl" --top_k 5 10 20

3. Different sequence length:
   python stock_inference.py --model_path "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt" --data_path "../ml-model/data-example/NASDAQ_all_features.pkl" --sequence_length 16

4. Close-only features:
   python stock_inference.py --model_path "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt" --data_path "../ml-model/data-example/NASDAQ_all_features.pkl" --feature_describe close_only

5. Save results to file:
   python stock_inference.py --model_path "../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt" --data_path "../ml-model/data-example/NASDAQ_all_features.pkl" --output_file results.json

PROGRAMMATIC USAGE:

from stock_inference import StockSelectionInference

# Initialize
inference = StockSelectionInference("../ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tt_100.ckpt")

# Run inference
results = inference.run_inference("./data/NASDAQ_all_features.pkl")

# Access top 10 stocks
top_10 = results['portfolio_selections']['top_10']
print("Top 10 stocks:", top_10['symbols'])
"""