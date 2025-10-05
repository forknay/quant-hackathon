"""
Complete FinBERT Sentiment Analysis - Lightning.ai Studio Version

This script runs entirely on Lightning.ai Studio and does everything:
1. Takes algorithm stock inputs
2. Extracts SEC filing data (if TextData is available on Lightning.ai)
3. Runs FinBERT processing 
4. Applies enhanced normalization (sum=1.0)
5. Returns final sentiment rankings

Upload this script to Lightning.ai Studio and run it there.
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime
import json
import os

def normalize_sentiment_scores(df, method='minmax', score_column='confidence_score'):
    """
    Apply enhanced normalization ensuring scores sum to exactly 1.0
    
    Args:
        df: DataFrame with sentiment scores
        method: 'minmax', 'softmax', or 'linear'
        score_column: Column name containing sentiment scores
    """
    print(f"Applying {method} normalization...")
    
    # Group by stock and aggregate scores
    stock_scores = df.groupby('stock_identifier')[score_column].mean().reset_index()
    
    if method == 'minmax':
        # Min-Max normalization to [0,1] range, then normalize to sum=1
        min_score = stock_scores[score_column].min()
        max_score = stock_scores[score_column].max()
        
        if max_score == min_score:
            # All scores are the same, equal distribution
            stock_scores['normalized_score'] = 1.0 / len(stock_scores)
        else:
            # Scale to [0,1] range
            scaled = (stock_scores[score_column] - min_score) / (max_score - min_score)
            # Normalize to sum=1
            stock_scores['normalized_score'] = scaled / scaled.sum()
            
    elif method == 'softmax':
        # Softmax normalization (natural probability distribution)
        exp_scores = np.exp(stock_scores[score_column] - stock_scores[score_column].max())
        stock_scores['normalized_score'] = exp_scores / exp_scores.sum()
        
    elif method == 'linear':
        # Linear normalization: shift negative, then proportional scaling
        shifted = stock_scores[score_column] - stock_scores[score_column].min() + 0.001
        stock_scores['normalized_score'] = shifted / shifted.sum()
    
    # Verify normalization
    total = stock_scores['normalized_score'].sum()
    print(f"Normalized scores sum: {total:.6f}")
    
    return stock_scores.sort_values('normalized_score', ascending=False)

def lightning_finbert_complete(stock_identifiers, textdata_path=None):
    """
    Complete FinBERT processing on Lightning.ai
    
    Args:
        stock_identifiers: List of stocks from algorithm ['1004:01', '1045:01']
        textdata_path: Path to TextData on Lightning.ai (if available)
    """
    print("🚀 COMPLETE FINBERT PROCESSING ON LIGHTNING.AI")
    print("=" * 55)
    print(f"Processing {len(stock_identifiers)} stocks: {stock_identifiers}")
    
    # Option 1: If TextData is available on Lightning.ai
    if textdata_path and os.path.exists(textdata_path):
        print("\n📁 STEP 1: Extracting SEC filing data...")
        texts_data = extract_textdata_lightning(stock_identifiers, textdata_path)
    else:
        # Option 2: Use uploaded CSV file (recommended for Lightning.ai)
        print("\n📁 STEP 1: Loading uploaded FinBERT input file...")
        # Look for uploaded file
        input_files = [f for f in os.listdir('.') if f.startswith('finbert_input_') and f.endswith('.csv')]
        if not input_files:
            raise FileNotFoundError("No FinBERT input file found. Upload finbert_input_*.csv to Lightning.ai Studio")
        
        input_file = input_files[0]  # Use the first found file
        print(f"Using input file: {input_file}")
        texts_data = pd.read_csv(input_file)
    
    print(f"Loaded {len(texts_data)} text segments for processing")
    
    # STEP 2: Initialize FinBERT
    print("\n⚡ STEP 2: Initializing FinBERT model...")
    finbert = pipeline("text-classification", 
                      model="ProsusAI/finbert", 
                      device=0)  # Use GPU on Lightning.ai
    
    # STEP 3: Process all texts through FinBERT
    print("\n🧠 STEP 3: Running FinBERT sentiment analysis...")
    results = []
    
    for idx, row in texts_data.iterrows():
        if idx % 10 == 0:
            print(f"Processing text {idx+1}/{len(texts_data)}...")
        
        # Run FinBERT on the text
        result = finbert(row['text'][:512])  # Limit to FinBERT max length
        
        results.append({
            'text_id': row.get('text_id', f"text_{idx}"),
            'stock_identifier': row.get('stock_identifier', f"{row.get('gvkey', 'UNK')}:01"),
            'gvkey': row.get('gvkey', 'UNK'),
            'confidence_score': result[0]['score'],
            'label': result[0]['label'],
            'text_type': row.get('text_type', 'unknown')
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # STEP 4: Apply enhanced normalization (all 3 methods)
    print("\n📊 STEP 4: Applying enhanced normalization...")
    
    normalized_minmax = normalize_sentiment_scores(results_df, method='minmax')
    normalized_softmax = normalize_sentiment_scores(results_df, method='softmax') 
    normalized_linear = normalize_sentiment_scores(results_df, method='linear')
    
    # STEP 5: Generate final results
    print("\n✅ STEP 5: Generating final sentiment rankings...")
    
    # Create comprehensive results
    final_results = {
        'processing_summary': {
            'input_stocks': stock_identifiers,
            'texts_processed': len(results_df),
            'processing_time': datetime.now().isoformat(),
            'normalization_sums': {
                'minmax': float(normalized_minmax['normalized_score'].sum()),
                'softmax': float(normalized_softmax['normalized_score'].sum()),
                'linear': float(normalized_linear['normalized_score'].sum())
            }
        },
        'sentiment_rankings': {
            'minmax_method': normalized_minmax.to_dict('records'),
            'softmax_method': normalized_softmax.to_dict('records'),
            'linear_method': normalized_linear.to_dict('records')
        }
    }
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw FinBERT results
    results_df.to_csv(f'finbert_results_{timestamp}.csv', index=False)
    
    # Save normalized rankings
    normalized_minmax.to_csv(f'sentiment_rankings_minmax_{timestamp}.csv', index=False)
    normalized_softmax.to_csv(f'sentiment_rankings_softmax_{timestamp}.csv', index=False) 
    normalized_linear.to_csv(f'sentiment_rankings_linear_{timestamp}.csv', index=False)
    
    # Save summary JSON
    with open(f'sentiment_analysis_summary_{timestamp}.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Display results
    print("\n🎯 FINAL SENTIMENT RANKINGS (Min-Max Normalized):")
    print(normalized_minmax[['stock_identifier', 'normalized_score']].head())
    
    print(f"\n📁 Results saved:")
    print(f"- finbert_results_{timestamp}.csv")
    print(f"- sentiment_rankings_minmax_{timestamp}.csv")
    print(f"- sentiment_rankings_softmax_{timestamp}.csv") 
    print(f"- sentiment_rankings_linear_{timestamp}.csv")
    print(f"- sentiment_analysis_summary_{timestamp}.json")
    
    return final_results

def extract_textdata_lightning(stock_identifiers, textdata_path):
    """
    Extract TextData on Lightning.ai (if TextData is uploaded there)
    """
    print(f"Extracting TextData from: {textdata_path}")
    
    # Convert stock identifiers to GVKEYs
    target_gvkeys = []
    for stock_id in stock_identifiers:
        gvkey, iid = stock_id.split(':')
        try:
            target_gvkeys.append(int(gvkey))
        except ValueError:
            target_gvkeys.append(gvkey)
    
    all_texts = []
    
    # Check available years
    for year in [2023, 2024, 2025]:
        year_path = os.path.join(textdata_path, str(year), f'text_us_{year}.parquet')
        if os.path.exists(year_path):
            print(f"  Loading {year} data...")
            df = pd.read_parquet(year_path)
            
            # Filter for target stocks
            for gvkey in target_gvkeys:
                stock_data = df[df['gvkey'] == gvkey].copy()
                if not stock_data.empty:
                    print(f"    Found {len(stock_data)} texts for GVKEY {gvkey} in {year}")
                    
                    # Add stock identifier
                    stock_data['stock_identifier'] = f"{gvkey}:01"
                    all_texts.append(stock_data)
    
    if all_texts:
        combined_texts = pd.concat(all_texts, ignore_index=True)
        print(f"Extracted {len(combined_texts)} total texts")
        return combined_texts
    else:
        raise ValueError("No TextData found for specified stocks")

# Example usage for Lightning.ai Studio
if __name__ == "__main__":
    # Example 1: Using uploaded FinBERT input file (RECOMMENDED)
    print("=== LIGHTNING.AI COMPLETE FINBERT PROCESSING ===")
    print("Upload your finbert_input_*.csv file to Lightning.ai Studio first!")
    
    # Your algorithm stocks
    algorithm_stocks = ['1004:01', '1045:01', '1050:01']
    
    try:
        # Run complete processing
        results = lightning_finbert_complete(algorithm_stocks)
        print("\n🎉 SUCCESS: Complete FinBERT processing finished!")
        print("All results are saved and ready for download.")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📋 TO FIX:")
        print("1. Upload finbert_input_*.csv file to Lightning.ai Studio")
        print("2. Run this script again")
        print("\nThe input file should contain columns: text_id, stock_identifier, gvkey, text")