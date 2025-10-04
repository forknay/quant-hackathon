"""
Simple inference module for stock predictions using pretrained models.

This is a lightweight wrapper around the full run_task.py infrastructure,
designed for easy integration with the end-to-end pipeline test.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from pathlib import Path
import sys

# Import model
from model import TransformerStockPrediction

class StockInference:
    """
    Simple inference class for loading a pretrained model and making predictions.
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the pretrained model checkpoint (.ckpt file)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model_path = model_path
        self.model = None
        
        print(f"Using device: {self.device}")
        
    def load_model(self, input_size=25, days=32, hidden_size=128, num_feat_att_layers=1,
                   num_pre_att_layers=1, num_heads=8, dropout=0.1):
        """
        Load the pretrained model.
        
        Default parameters match the NASDAQ pretrained models:
        - hidden_size=128
        - num_feat_att_layers=1
        - num_pre_att_layers=1
        
        Args:
            input_size: Number of input features (default: 25 for OHLCV features)
            days: Sequence length (default: 32)
            hidden_size: Hidden dimension size (default: 128)
            num_feat_att_layers: Number of feature attention layers (default: 1)
            num_pre_att_layers: Number of prediction attention layers (default: 1)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout rate (default: 0.1)
        """
        print(f"Loading model from {self.model_path}...")
        print(f"  Architecture: hidden_size={hidden_size}, layers=({num_feat_att_layers},{num_pre_att_layers}), heads={num_heads}")
        
        # Initialize model architecture
        self.model = TransformerStockPrediction(
            input_size=input_size,
            num_class=1,  # Regression task
            hidden_size=hidden_size,
            num_feat_att_layers=num_feat_att_layers,
            num_pre_att_layers=num_pre_att_layers,
            num_heads=num_heads,
            days=days,
            dropout=dropout
        ).to(self.device)
        
        # Load weights
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Load with strict=False to ignore pretrain layers (not needed for inference)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            
            if missing_keys:
                # Filter out expected missing keys (prediction layer)
                critical_missing = [k for k in missing_keys if not k.startswith('fc.')]
                if critical_missing:
                    print(f"⚠️  Warning: Missing critical keys: {critical_missing}")
            
            if unexpected_keys:
                # Filter out expected unexpected keys (pretrain layers)
                non_pretrain_unexpected = [k for k in unexpected_keys if 'pretrain' not in k]
                if non_pretrain_unexpected:
                    print(f"⚠️  Warning: Unexpected non-pretrain keys: {non_pretrain_unexpected}")
            
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def prepare_data(self, features_array, days=32):
        """
        Prepare data for inference.
        
        Args:
            features_array: numpy array of shape (num_days, num_features)
                            Expected format from data.py:
                            - Column 0: date index
                            - Columns 1-25: features
            days: Sequence length for model (default: 32)
        
        Returns:
            torch tensor ready for model input
        """
        # Remove date index column (column 0)
        if features_array.shape[1] == 26:
            features = features_array[:, 1:]  # Columns 1-25
        else:
            features = features_array
        
        # Get the last 'days' rows
        if len(features) < days:
            # Pad if insufficient data
            padding = np.zeros((days - len(features), features.shape[1]))
            features = np.vstack([padding, features])
        else:
            features = features[-days:]
        
        # Replace missing values (-1234) with 0
        features = np.where(features == -1234, 0, features)
        
        # Convert to tensor: shape (1, days, features)
        tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict(self, features_dict, days=32, return_all=False):
        """
        Run inference on a dict of stocks.
        
        Args:
            features_dict: Dict of {stock_id: features_array}
                          Each features_array is shape (num_days, 26)
            days: Sequence length (default: 32)
            return_all: If True, return all predictions; if False, only valid ones
        
        Returns:
            dict: {stock_id: prediction_score}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        predictions = {}
        skipped = 0
        
        self.model.eval()
        with torch.no_grad():
            for stock_id, features_array in features_dict.items():
                try:
                    # Check if enough recent data is valid
                    recent_data = features_array[-days:, 1:]  # Last 'days', exclude date column
                    
                    # Skip if too many missing values
                    missing_ratio = (recent_data == -1234).sum() / recent_data.size
                    if missing_ratio > 0.5:  # Skip if > 50% missing
                        skipped += 1
                        if not return_all:
                            continue
                    
                    # Prepare input
                    input_tensor = self.prepare_data(features_array, days=days)
                    
                    # Run inference
                    output = self.model(input_tensor)
                    
                    # Get prediction (regression output)
                    pred_value = output.item()
                    predictions[stock_id] = pred_value
                    
                except Exception as e:
                    skipped += 1
                    print(f"  Warning: Skipped {stock_id}: {e}")
                    continue
        
        print(f"  Predicted: {len(predictions)} stocks | Skipped: {skipped}")
        return predictions
    
    def rank_predictions(self, predictions, top_k=10, bottom_k=10):
        """
        Rank predictions and return top/bottom stocks.
        
        Args:
            predictions: Dict of {stock_id: score}
            top_k: Number of top stocks to return
            bottom_k: Number of bottom stocks to return
        
        Returns:
            dict with 'top' and 'bottom' rankings
        """
        if not predictions:
            return {'top': [], 'bottom': []}
        
        # Sort by score
        sorted_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        top_stocks = sorted_stocks[:top_k]
        bottom_stocks = sorted_stocks[-bottom_k:]
        
        return {
            'top': {
                'stocks': [s[0] for s in top_stocks],
                'scores': [s[1] for s in top_stocks]
            },
            'bottom': {
                'stocks': [s[0] for s in bottom_stocks],
                'scores': [s[1] for s in bottom_stocks]
            }
        }


def load_features_from_pkl(pkl_path):
    """
    Load features from a .pkl file.
    
    Args:
        pkl_path: Path to the .pkl file
    
    Returns:
        dict: {'all_features': {stock_id: features}, ...}
    """
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    return data


def run_inference(model_path, pkl_path, top_k=10, bottom_k=10, days=32, device='cpu'):
    """
    Convenience function to run end-to-end inference.
    
    Args:
        model_path: Path to model checkpoint
        pkl_path: Path to features pkl file
        top_k: Number of top stocks to return
        bottom_k: Number of bottom stocks to return
        days: Sequence length
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: Rankings with top and bottom stocks
    """
    print("=" * 80)
    print("RUNNING STOCK INFERENCE")
    print("=" * 80)
    
    # Load features
    print(f"\n[1/3] Loading features from {pkl_path}...")
    data = load_features_from_pkl(pkl_path)
    all_features = data.get('all_features', {})
    print(f"      Loaded {len(all_features)} stocks")
    
    # Initialize inference
    print(f"\n[2/3] Loading model...")
    inference = StockInference(model_path, device=device)
    inference.load_model(input_size=25, days=days)
    
    # Run predictions
    print(f"\n[3/3] Running inference...")
    predictions = inference.predict(all_features, days=days)
    
    # Rank results
    rankings = inference.rank_predictions(predictions, top_k=top_k, bottom_k=bottom_k)
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    
    return rankings


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stock inference')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', required=True, help='Path to features pkl file')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top stocks')
    parser.add_argument('--bottom-k', type=int, default=10, help='Number of bottom stocks')
    parser.add_argument('--days', type=int, default=32, help='Sequence length')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    results = run_inference(
        model_path=args.model,
        pkl_path=args.data,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        days=args.days,
        device=args.device
    )
    
    print("\nTOP STOCKS:")
    for i, (stock, score) in enumerate(zip(results['top']['stocks'], results['top']['scores']), 1):
        print(f"  {i:2d}. {stock:<20} Score: {score:7.4f}")
    
    print("\nBOTTOM STOCKS:")
    for i, (stock, score) in enumerate(zip(results['bottom']['stocks'], results['bottom']['scores']), 1):
        print(f"  {i:2d}. {stock:<20} Score: {score:7.4f}")

