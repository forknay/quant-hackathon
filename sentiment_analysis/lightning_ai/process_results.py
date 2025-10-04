"""
Process FinBERT Sentiment Results from Lightning.ai

This script processes the sentiment results and creates final stock rankings.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def process_finbert_results(results_file: str, original_data_file: str) -> pd.DataFrame:
    """
    Process FinBERT sentiment results and create stock rankings
    
    Args:
        results_file: Path to CSV with sentiment results from Lightning.ai
        original_data_file: Path to original input CSV with stock info
        
    Returns:
        DataFrame with final stock sentiment rankings
    """
    print(f"Processing FinBERT results from {results_file}...")
    
    # Load data
    results_df = pd.read_csv(results_file)
    original_df = pd.read_csv(original_data_file)
    
    # Merge results with original data
    merged_df = results_df.merge(original_df[['text_id', 'stock_id', 'stock_category']], 
                                on='text_id', how='inner')
    
    print(f"Successfully merged {len(merged_df)} sentiment results")
    
    # Calculate sentiment metrics per text
    merged_df['net_sentiment'] = merged_df['positive'] - merged_df['negative']
    merged_df['sentiment_strength'] = np.maximum(merged_df['positive'], merged_df['negative'])
    
    # Aggregate by stock
    stock_sentiment = merged_df.groupby(['stock_id', 'stock_category']).agg({
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean',
        'net_sentiment': 'mean',
        'sentiment_strength': 'mean',
        'text_id': 'count'
    }).round(4)
    
    stock_sentiment.columns = [
        'avg_positive', 'avg_negative', 'avg_neutral', 
        'net_sentiment', 'avg_strength', 'text_count'
    ]
    stock_sentiment = stock_sentiment.reset_index()
    
    # Calculate final weighted sentiment score
    stock_sentiment['weighted_sentiment'] = (
        stock_sentiment['net_sentiment'] * stock_sentiment['avg_strength']
    )
    
    # Create rankings
    stock_sentiment = stock_sentiment.sort_values('weighted_sentiment', ascending=False)
    stock_sentiment['sentiment_rank'] = range(1, len(stock_sentiment) + 1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"stock_sentiment_rankings_{timestamp}.csv"
    stock_sentiment.to_csv(output_file, index=False)
    
    print(f"\n=== STOCK SENTIMENT RANKINGS ===")
    display_cols = ['stock_id', 'stock_category', 'net_sentiment', 
                   'weighted_sentiment', 'sentiment_rank', 'text_count']
    print(stock_sentiment[display_cols].to_string(index=False))
    
    print(f"\nDetailed results saved to: {output_file}")
    return stock_sentiment


def main():
    """
    Main function - update file paths and run
    """
    print("FinBERT Results Processor")
    print("=" * 40)
    
    # UPDATE THESE PATHS:
    results_file = "finbert_sentiment_results.csv"  # From Lightning.ai
    original_data_file = "finbert_input_YYYYMMDD_HHMMSS.csv"  # Your input file
    
    print(f"Looking for results file: {results_file}")
    print(f"Looking for original data: {original_data_file}")
    
    if Path(results_file).exists() and Path(original_data_file).exists():
        rankings = process_finbert_results(results_file, original_data_file)
        
        print(f"\nSUCCESS! Processed sentiment analysis for {len(rankings)} stocks")
        
        # Show top and bottom performers
        print(f"\nTop 3 by sentiment:")
        print(rankings.head(3)[['stock_id', 'weighted_sentiment', 'sentiment_rank']].to_string(index=False))
        
        print(f"\nBottom 3 by sentiment:")  
        print(rankings.tail(3)[['stock_id', 'weighted_sentiment', 'sentiment_rank']].to_string(index=False))
        
    else:
        print("ERROR: Could not find input files!")
        print("Make sure to:")
        print("1. Download sentiment results from Lightning.ai")
        print("2. Update file paths in this script")
        print("3. Run the script again")


if __name__ == "__main__":
    main()
