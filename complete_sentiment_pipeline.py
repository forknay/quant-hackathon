"""
COMPLETE SENTIMENT ANALYSIS PIPELINE
=====================================

All-in-one script for complete FinBERT sentiment analysis from stock symbol    print(f"   [SUCCESS] Extracted {len(result_df)} total texts")
        print(f"   [SUCCESS] Successfully processed {len(results_df)} texts")
       print(f"\n[COMPLETE] PIPELINE COMPLETE!")
    print(f"[PROCESSED] Processed {len(results_df)} texts from {len(results_df['stock_identifier'].unique())} stocks")
    print(f"[SAVED] Results saved with timestamp: {timestamp}")int(f"   [STATS] Average confidence: {results_df['confidence_score'].mean():.3f}")int(f"   [STATS] Texts per stock: {result_df['stock_identifier'].value_counts().to_dict()}")to final rankings.

This script does EVERYTHING:
1. Takes stock symbols as input (from algorithm or manual input)
2. Extracts real SEC filing texts from TextData
3. Runs FinBERT sentiment analysis 
4. Applies 3 enhanced normalization methods
5. Generates final sentiment rankings
6. Saves all results with timestamps

USAGE:
------
Local: python complete_sentiment_pipeline.py
Lightning.ai: Upload this file and run it in Studio

INPUT OPTIONS:
- Manual: Edit STOCK_SYMBOLS list below
- Algorithm: Pass stocks to process_stocks_sentiment() function
- CSV: Upload finbert_input_*.csv file (auto-detected)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Default stock symbols (modify these or pass to function)
STOCK_SYMBOLS = ['1004:01', '1045:01', '1050:01']

def normalize_sentiment_scores(results_df, method='minmax', score_column='confidence_score'):
    """
    Apply enhanced normalization ensuring scores sum to exactly 1.0
    
    Args:
        results_df: DataFrame with sentiment results
        method: 'minmax', 'softmax', or 'linear' 
        score_column: Column containing sentiment scores
        
    Returns:
        DataFrame: Stock-level normalized sentiment rankings
    """
    print(f"[NORMALIZE] Applying {method} normalization...")
    
    # Aggregate scores by stock
    stock_scores = results_df.groupby('stock_identifier')[score_column].mean().reset_index()
    
    if method == 'minmax':
        # Min-Max normalization to [0,1], then normalize to sum=1
        min_score = stock_scores[score_column].min()
        max_score = stock_scores[score_column].max()
        
        if max_score == min_score:
            stock_scores['normalized_score'] = 1.0 / len(stock_scores)
        else:
            scaled = (stock_scores[score_column] - min_score) / (max_score - min_score)
            stock_scores['normalized_score'] = scaled / scaled.sum()
            
    elif method == 'softmax':
        # Softmax normalization (probability distribution)
        exp_scores = np.exp(stock_scores[score_column] - stock_scores[score_column].max())
        stock_scores['normalized_score'] = exp_scores / exp_scores.sum()
        
    elif method == 'linear':
        # Linear shift and scale to sum=1
        shifted = stock_scores[score_column] - stock_scores[score_column].min() + 0.001
        stock_scores['normalized_score'] = shifted / shifted.sum()
    
    # Verify normalization (mathematical guarantee)
    total = stock_scores['normalized_score'].sum()
    print(f"   [OK] Normalized scores sum: {total:.6f}")
    
    return stock_scores.sort_values('normalized_score', ascending=False)

def extract_textdata_for_stocks(stock_identifiers, year_range=(2023, 2025), max_texts_per_stock=50):
    """
    Extract real SEC filing texts from TextData parquet files
    
    Args:
        stock_identifiers: List of stock IDs like ['1004:01', '1045:01']
        year_range: Tuple of (start_year, end_year)
        max_texts_per_stock: Maximum texts to extract per stock
        
    Returns:
        DataFrame: Extracted texts with stock identifiers
    """
    print(f"[EXTRACT] Extracting SEC filing texts for {len(stock_identifiers)} stocks...")
    
    all_texts = []
    
    # Check if we're on Lightning.ai or local
    textdata_dirs = ['TextData', '../TextData', '../../TextData']
    textdata_path = None
    
    for dir_path in textdata_dirs:
        if os.path.exists(dir_path):
            textdata_path = dir_path
            break
    
    if not textdata_path:
        raise FileNotFoundError("TextData directory not found. Upload TextData to Lightning.ai or ensure it's in the workspace.")
    
    print(f"   [PATH] Using TextData path: {textdata_path}")
    
    # Extract GVKEYs from stock identifiers
    gvkeys = [stock.split(':')[0] for stock in stock_identifiers]
    
    for year in range(year_range[0], year_range[1] + 1):
        year_path = os.path.join(textdata_path, str(year))
        if not os.path.exists(year_path):
            print(f"   [WARN] Year {year} not found, skipping...")
            continue
        
        # Find parquet files for this year
        parquet_files = [f for f in os.listdir(year_path) if f.endswith('.parquet')]
        
        for file in parquet_files:
            try:
                file_path = os.path.join(year_path, file)
                df = pd.read_parquet(file_path)
                
                # Filter for our stocks
                if 'gvkey' in df.columns:
                    matched_data = df[df['gvkey'].astype(str).isin(gvkeys)]
                    
                    if len(matched_data) > 0:
                        print(f"   [FOUND] Found {len(matched_data)} texts in {year}/{file}")
                        
                        for gvkey in gvkeys:
                            stock_data = matched_data[matched_data['gvkey'].astype(str) == gvkey]
                            
                            if len(stock_data) > 0:
                                # Limit texts per stock
                                stock_data = stock_data.head(max_texts_per_stock)
                                
                                for _, row in stock_data.iterrows():
                                    all_texts.append({
                                        'text_id': f"{gvkey}_{year}_{len(all_texts)}",
                                        'stock_identifier': f"{gvkey}:01",
                                        'gvkey': gvkey,
                                        'year': year,
                                        'text': str(row.get('text', row.get('content', ''))),
                                        'filename': file
                                    })
                
            except Exception as e:
                print(f"   [ERROR] Error processing {file}: {str(e)}")
                continue
    
    if not all_texts:
        raise ValueError(f"No texts found for stocks {stock_identifiers} in years {year_range}")
    
    result_df = pd.DataFrame(all_texts)
    print(f"   ✅ Extracted {len(result_df)} total texts")
    print(f"   📊 Texts per stock: {result_df['stock_identifier'].value_counts().to_dict()}")
    
    return result_df

def run_finbert_analysis(texts_df):
    """
    Run FinBERT sentiment analysis on extracted texts
    
    Args:
        texts_df: DataFrame with text column and stock identifiers
        
    Returns:
        DataFrame: Results with sentiment scores and labels
    """
    print(f"[FINBERT] Running FinBERT analysis on {len(texts_df)} texts...")
    
    try:
        from transformers import pipeline
        print("   [OK] Transformers library loaded")
    except ImportError:
        raise ImportError("Install transformers: pip install transformers torch")
    
    # Initialize FinBERT with GPU/CPU fallback
    try:
        print("   [GPU] Initializing FinBERT model (GPU)...")
        finbert = pipeline("text-classification", model="ProsusAI/finbert", device=0)
        print("   [OK] GPU initialization successful")
    except:
        print("   [FALLBACK] GPU failed, falling back to CPU...")
        finbert = pipeline("text-classification", model="ProsusAI/finbert", device=-1)
        print("   [OK] CPU initialization successful")
    
    # Process all texts
    results = []
    total_texts = len(texts_df)
    
    for idx, row in texts_df.iterrows():
        if idx % 10 == 0:
            print(f"   Processing {idx+1}/{total_texts} ({(idx+1)/total_texts*100:.1f}%)...")
        
        try:
            # Truncate text to FinBERT's max length
            text = str(row['text'])[:512]
            
            # Skip empty texts
            if not text.strip():
                continue
            
            result = finbert(text)
            
            results.append({
                'text_id': row['text_id'],
                'stock_identifier': row['stock_identifier'],
                'gvkey': row['gvkey'],
                'year': row['year'],
                'text_length': len(text),
                'confidence_score': result[0]['score'],
                'label': result[0]['label'],
                'text_preview': text[:100] + "..." if len(text) > 100 else text
            })
            
        except Exception as e:
            print(f"   [ERROR] Error processing text {idx}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No texts were successfully processed by FinBERT")
    
    results_df = pd.DataFrame(results)
    print(f"   ✅ Successfully processed {len(results_df)} texts")
    print(f"   📈 Average confidence: {results_df['confidence_score'].mean():.3f}")
    
    return results_df

def process_stocks_sentiment(stock_identifiers=None, input_csv=None, max_texts_per_stock=50):
    """
    COMPLETE PIPELINE: Process stocks from symbols to final sentiment rankings
    
    Args:
        stock_identifiers: List of stocks ['1004:01', '1045:01'] (optional)
        input_csv: Path to CSV with pre-extracted texts (optional)
        max_texts_per_stock: Maximum texts to process per stock
    
    Returns:
        dict: Complete results with all normalization methods
    """
    print("[PIPELINE] COMPLETE SENTIMENT ANALYSIS PIPELINE")
    print("=" * 50)
    print(f"[START] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine input method
    if input_csv and os.path.exists(input_csv):
        print(f"[INPUT] Using pre-extracted CSV: {input_csv}")
        texts_df = pd.read_csv(input_csv)
        
    elif stock_identifiers:
        print(f"[STOCKS] Processing stocks: {stock_identifiers}")
        texts_df = extract_textdata_for_stocks(stock_identifiers, max_texts_per_stock=max_texts_per_stock)
        
    else:
        # Auto-detect uploaded CSV files
        csv_files = [f for f in os.listdir('.') if f.startswith('finbert_input_') and f.endswith('.csv')]
        if csv_files:
            input_csv = csv_files[0]
            print(f"[AUTO] Auto-detected CSV: {input_csv}")
            texts_df = pd.read_csv(input_csv)
        else:
            print(f"[DEFAULT] Using default stocks: {STOCK_SYMBOLS}")
            texts_df = extract_textdata_for_stocks(STOCK_SYMBOLS, max_texts_per_stock=max_texts_per_stock)
    
    # Run FinBERT analysis
    results_df = run_finbert_analysis(texts_df)
    
    # Apply all normalization methods
    print("[RANKINGS] Generating sentiment rankings with 3 normalization methods...")
    
    normalized_minmax = normalize_sentiment_scores(results_df, method='minmax')
    normalized_softmax = normalize_sentiment_scores(results_df, method='softmax') 
    normalized_linear = normalize_sentiment_scores(results_df, method='linear')
    
    # Save all results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results
    results_df.to_csv(f'finbert_results_{timestamp}.csv', index=False)
    
    # Save normalized rankings
    normalized_minmax.to_csv(f'sentiment_rankings_minmax_{timestamp}.csv', index=False)
    normalized_softmax.to_csv(f'sentiment_rankings_softmax_{timestamp}.csv', index=False)
    normalized_linear.to_csv(f'sentiment_rankings_linear_{timestamp}.csv', index=False)
    
    # Create summary report
    summary = {
        'timestamp': timestamp,
        'total_texts_processed': len(results_df),
        'unique_stocks': len(results_df['stock_identifier'].unique()),
        'average_confidence': float(results_df['confidence_score'].mean()),
        'top_stocks_minmax': normalized_minmax.head(3).to_dict('records'),
        'top_stocks_softmax': normalized_softmax.head(3).to_dict('records'),
        'top_stocks_linear': normalized_linear.head(3).to_dict('records')
    }
    
    with open(f'sentiment_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Display results
    print(f"\n✅ PIPELINE COMPLETE!")
    print(f"📊 Processed {len(results_df)} texts from {len(results_df['stock_identifier'].unique())} stocks")
    print(f"📁 Results saved with timestamp: {timestamp}")
    print("\n[RESULTS] TOP SENTIMENT STOCKS (All Methods):")
    print("\nMin-Max Normalization:")
    print(normalized_minmax[['stock_identifier', 'confidence_score', 'normalized_score']].head())
    print(f"Sum: {normalized_minmax['normalized_score'].sum():.6f}")
    
    print("\nSoftmax Normalization:")
    print(normalized_softmax[['stock_identifier', 'confidence_score', 'normalized_score']].head())
    print(f"Sum: {normalized_softmax['normalized_score'].sum():.6f}")
    
    print("\nLinear Normalization:")
    print(normalized_linear[['stock_identifier', 'confidence_score', 'normalized_score']].head())
    print(f"Sum: {normalized_linear['normalized_score'].sum():.6f}")
    
    return {
        'raw_results': results_df,
        'minmax_rankings': normalized_minmax,
        'softmax_rankings': normalized_softmax,
        'linear_rankings': normalized_linear,
        'summary': summary,
        'timestamp': timestamp
    }

def validate_stock_input(stocks):
    """
    Validate and format stock identifiers
    
    Args:
        stocks: List of stock symbols or identifiers
        
    Returns:
        List: Validated and formatted stock identifiers
    """
    validated = []
    
    for stock in stocks:
        stock = str(stock).strip()
        
        # Convert different formats to standard GVKEY:IID format
        if ':' in stock:
            # Already in GVKEY:IID format
            validated.append(stock)
        elif stock.isdigit():
            # Just GVKEY, add default IID
            validated.append(f"{stock}:01")
        else:
            # Unknown format, try to use as-is
            print(f"[WARN] Warning: Unknown stock format '{stock}', using as-is")
            validated.append(stock)
    
    print(f"Validated {len(validated)} stock identifiers: {validated}")
    return validated

# MAIN EXECUTION
if __name__ == "__main__":
    """
    Main execution - handles command line arguments and direct calls
    """
    
    print("[MAIN] COMPLETE SENTIMENT ANALYSIS PIPELINE")
    print("========================================")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Command line mode: python complete_sentiment_pipeline.py STOCK1 STOCK2 STOCK3
        input_stocks = sys.argv[1:]
        stocks = validate_stock_input(input_stocks)
        print(f"[CMDLINE] Command line input: {stocks}")
        
        results = process_stocks_sentiment(stock_identifiers=stocks)
        
    else:
        # Default mode: use STOCK_SYMBOLS or auto-detect CSV
        print("[AUTO] Auto-detection mode...")
        results = process_stocks_sentiment()
    
    print(f"\n[DONE] All done! Check the output files for your sentiment rankings.")
    print(f"[END] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ALGORITHM INTEGRATION FUNCTION
def get_sentiment_rankings(algorithm_stocks, normalization_method='minmax'):
    """
    ALGORITHM INTEGRATION: Get sentiment rankings for stocks from an external algorithm
    
    This is the function your algorithm should call to get sentiment rankings.
    
    Args:
        algorithm_stocks: List of stock identifiers from your algorithm
        normalization_method: 'minmax', 'softmax', or 'linear'
    
    Returns:
        DataFrame: Sentiment rankings ready for your algorithm
    """
    print(f"[ALGORITHM] ALGORITHM INTEGRATION: Processing {len(algorithm_stocks)} stocks...")
    
    # Validate input
    stocks = validate_stock_input(algorithm_stocks)
    
    # Run complete pipeline
    results = process_stocks_sentiment(stock_identifiers=stocks)
    
    # Return requested normalization method
    if normalization_method == 'minmax':
        return results['minmax_rankings']
    elif normalization_method == 'softmax':
        return results['softmax_rankings']
    elif normalization_method == 'linear':
        return results['linear_rankings']
    else:
        print(f"[WARN] Unknown normalization method '{normalization_method}', using minmax")
        return results['minmax_rankings']

# EXAMPLE USAGE FOR ALGORITHMS:
"""
# Example: How to integrate this with your trading algorithm

from complete_sentiment_pipeline import get_sentiment_rankings

# Your algorithm generates a list of stocks to analyze
algorithm_stocks = ['1004:01', '1045:01', '1050:01', '2034:01', '3567:01']

# Get sentiment rankings (normalized to sum=1.0)
sentiment_rankings = get_sentiment_rankings(algorithm_stocks, normalization_method='softmax')

# Use the rankings in your algorithm
for _, row in sentiment_rankings.iterrows():
    stock = row['stock_identifier']
    score = row['normalized_score']
    print(f"Stock {stock}: Sentiment weight = {score:.4f}")

# Rankings are already sorted by sentiment (best first)
top_stock = sentiment_rankings.iloc[0]['stock_identifier']
print(f"Top sentiment stock: {top_stock}")
"""