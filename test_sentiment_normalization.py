"""
Simple Stock Sentiment Ranking Processor

Processes FinBERT sentiment results and generates rankings with normalized scores.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def process_sentiment_rankings_simple(results_file):
    """Process FinBERT results and create sentiment rankings with normalized scores"""
    print("PROCESSING STOCK SENTIMENT RANKINGS WITH NORMALIZATION")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        return None
    
    print(f"Loading results from: {results_file}")
    
    # Load FinBERT results
    if results_file.endswith('.json'):
        # Load JSON results and convert to DataFrame
        with open(results_file, 'r') as f:
            json_data = json.load(f)
        
        # Convert to DataFrame
        if isinstance(json_data, list):
            df_results = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            if 'results' in json_data:
                df_results = pd.DataFrame(json_data['results'])
            else:
                # Assume the dict values are the results
                df_results = pd.DataFrame(list(json_data.values()))
        else:
            print(f"ERROR: Unexpected JSON format")
            return None
    else:
        # Load CSV results
        try:
            df_results = pd.read_csv(results_file)
        except Exception as e:
            print(f"ERROR loading CSV: {e}")
            return None
    
    print(f"Loaded {len(df_results)} sentiment results")
    print(f"Columns: {list(df_results.columns)}")
    
    # Check required columns
    required_cols = ['stock_identifier', 'sentiment_label', 'confidence_score']
    missing_cols = [col for col in required_cols if col not in df_results.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    # Convert sentiment labels to numeric scores
    def sentiment_to_numeric(label):
        label_lower = str(label).lower()
        if 'positive' in label_lower:
            return 1.0
        elif 'negative' in label_lower:
            return -1.0
        else:  # neutral
            return 0.0
    
    df_results['numeric_sentiment'] = df_results['sentiment_label'].apply(sentiment_to_numeric)
    
    # Show sentiment distribution
    sentiment_dist = df_results['sentiment_label'].value_counts()
    print("\nSentiment Distribution:")
    for label, count in sentiment_dist.items():
        pct = (count / len(df_results)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Aggregate by stock
    print("\nAggregating sentiment by stock...")
    
    stock_sentiment = df_results.groupby('stock_identifier').agg({
        'numeric_sentiment': ['mean', 'std', 'count'],
        'confidence_score': 'mean',
        'sentiment_label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'
    }).round(4)
    
    # Flatten column names
    stock_sentiment.columns = ['avg_numeric_sentiment', 'std_numeric_sentiment', 'text_count', 
                              'avg_confidence', 'dominant_sentiment']
    stock_sentiment = stock_sentiment.reset_index()
    
    # Fill NaN std values (for stocks with only one text)
    stock_sentiment['std_numeric_sentiment'] = stock_sentiment['std_numeric_sentiment'].fillna(0)
    
    print(f"Processed {len(stock_sentiment)} unique stocks")
    
    # NORMALIZATION: Create normalized scores
    print("\nCalculating normalized sentiment scores...")
    
    sentiment_scores = stock_sentiment['avg_numeric_sentiment'].values
    
    # Method 1: Min-Max Normalization (0 to 1 scale, then normalize to sum=1)
    min_sent = sentiment_scores.min()
    max_sent = sentiment_scores.max()
    
    if max_sent == min_sent:
        # All stocks have same sentiment - equal weights
        stock_sentiment['normalized_minmax'] = 1.0 / len(stock_sentiment)
    else:
        # Scale to 0-1 range
        normalized_01 = (sentiment_scores - min_sent) / (max_sent - min_sent)
        # Normalize so they sum to 1
        stock_sentiment['normalized_minmax'] = normalized_01 / normalized_01.sum()
    
    # Method 2: Softmax Normalization (probability distribution)
    exp_values = np.exp(sentiment_scores - np.max(sentiment_scores))  # Numerical stability
    stock_sentiment['normalized_softmax'] = exp_values / np.sum(exp_values)
    
    # Method 3: Linear Shift and Scale
    if sentiment_scores.std() == 0:
        stock_sentiment['normalized_linear'] = 1.0 / len(stock_sentiment)
    else:
        shifted = sentiment_scores - min_sent + 0.1  # Ensure positive
        stock_sentiment['normalized_linear'] = shifted / shifted.sum()
    
    # Sort by average numeric sentiment (best to worst)
    stock_sentiment = stock_sentiment.sort_values('avg_numeric_sentiment', ascending=False).reset_index(drop=True)
    stock_sentiment['rank'] = range(1, len(stock_sentiment) + 1)
    
    # Display results
    print(f"\nFINAL SENTIMENT RANKINGS WITH NORMALIZED SCORES")
    print("=" * 80)
    
    display_cols = ['rank', 'stock_identifier', 'avg_numeric_sentiment', 'normalized_minmax', 
                   'normalized_softmax', 'normalized_linear', 'dominant_sentiment', 'text_count']
    
    for col in display_cols:
        if col in stock_sentiment.columns:
            continue
    
    # Format and display
    for i, row in stock_sentiment.iterrows():
        print(f"{row['rank']:<4} {row['stock_identifier']:<12} "
              f"{row['avg_numeric_sentiment']:<8.3f} {row['normalized_minmax']:<8.3f} "
              f"{row['normalized_softmax']:<8.3f} {row['normalized_linear']:<8.3f} "
              f"{row['dominant_sentiment']:<10} {row['text_count']:<6}")
    
    # Validation
    print(f"\nNORMALIZATION VALIDATION:")
    print(f"  Min-Max normalized sum: {stock_sentiment['normalized_minmax'].sum():.6f}")
    print(f"  Softmax normalized sum: {stock_sentiment['normalized_softmax'].sum():.6f}")
    print(f"  Linear normalized sum: {stock_sentiment['normalized_linear'].sum():.6f}")
    
    # Score ranges
    print(f"\nSCORE RANGES:")
    print(f"  Numeric sentiment: {stock_sentiment['avg_numeric_sentiment'].min():.3f} to {stock_sentiment['avg_numeric_sentiment'].max():.3f}")
    print(f"  Min-Max normalized: {stock_sentiment['normalized_minmax'].min():.3f} to {stock_sentiment['normalized_minmax'].max():.3f}")
    print(f"  Softmax normalized: {stock_sentiment['normalized_softmax'].min():.3f} to {stock_sentiment['normalized_softmax'].max():.3f}")
    print(f"  Linear normalized: {stock_sentiment['normalized_linear'].min():.3f} to {stock_sentiment['normalized_linear'].max():.3f}")
    
    # Save results
    output_file = f"sentiment_rankings_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    stock_sentiment.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return stock_sentiment

if __name__ == "__main__":
    # Test with the mock FinBERT results
    results_file = r"C:\Users\positive\Documents\GitHub\quant-hackathon\mock_finbert_results_real_stocks.csv"
    final_rankings = process_sentiment_rankings_simple(results_file)