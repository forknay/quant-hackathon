"""
Lightning.ai Simple FinBERT - Works with Any Text Input

Ultra-simple version: Just upload ANY CSV with 'text' and 'stock_identifier' columns
and this will do complete FinBERT processing with normalization.

Perfect for algorithm testing on Lightning.ai Studio.
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime

def simple_lightning_finbert(csv_file=None):
    """
    Simple complete FinBERT processing on Lightning.ai
    
    Args:
        csv_file: Name of uploaded CSV file (auto-detects if None)
    """
    print("🚀 SIMPLE LIGHTNING.AI FINBERT PROCESSING")
    print("=" * 50)
    
    # Auto-detect input file
    if csv_file is None:
        import os
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("Upload a CSV file with 'text' and 'stock_identifier' columns")
        csv_file = csv_files[0]
        print(f"Using: {csv_file}")
    
    # Load data
    df = pd.read_csv(csv_file)
    required_columns = ['text', 'stock_identifier']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    print(f"Loaded {len(df)} texts for {df['stock_identifier'].nunique()} stocks")
    
    # Initialize FinBERT
    print("Initializing FinBERT...")
    finbert = pipeline("text-classification", model="ProsusAI/finbert", device=0)
    
    # Process texts
    print("Running FinBERT processing...")
    results = []
    
    for idx, row in df.iterrows():
        if idx % 5 == 0:
            print(f"Processing {idx+1}/{len(df)}...")
        
        result = finbert(str(row['text'])[:512])  # FinBERT max length
        
        results.append({
            'stock_identifier': row['stock_identifier'],
            'confidence_score': result[0]['score'],
            'sentiment_label': result[0]['label']
        })
    
    results_df = pd.DataFrame(results)
    
    # Normalize scores (Min-Max method, sum=1.0)
    print("Applying normalization...")
    stock_scores = results_df.groupby('stock_identifier')['confidence_score'].mean().reset_index()
    
    # Min-Max normalization
    min_score = stock_scores['confidence_score'].min()
    max_score = stock_scores['confidence_score'].max()
    
    if max_score == min_score:
        stock_scores['normalized_score'] = 1.0 / len(stock_scores)
    else:
        scaled = (stock_scores['confidence_score'] - min_score) / (max_score - min_score)
        stock_scores['normalized_score'] = scaled / scaled.sum()
    
    # Sort by sentiment
    final_rankings = stock_scores.sort_values('normalized_score', ascending=False)
    
    # Verify normalization
    total = final_rankings['normalized_score'].sum()
    print(f"✅ Normalized scores sum: {total:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'sentiment_rankings_{timestamp}.csv'
    final_rankings.to_csv(output_file, index=False)
    
    # Display results
    print(f"\n🎯 FINAL SENTIMENT RANKINGS:")
    print(final_rankings[['stock_identifier', 'normalized_score']])
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return final_rankings

# Run automatically when script starts
if __name__ == "__main__":
    try:
        rankings = simple_lightning_finbert()
        print("\n🎉 SUCCESS: Complete processing finished on Lightning.ai!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📋 REQUIREMENTS:")
        print("1. Upload CSV file with columns: 'text', 'stock_identifier'")
        print("2. Run this script")
        print("\nExample CSV format:")
        print("stock_identifier,text")
        print("1004:01,\"Risk factors include market volatility...\"")
        print("1045:01,\"Management believes the outlook is positive...\"")