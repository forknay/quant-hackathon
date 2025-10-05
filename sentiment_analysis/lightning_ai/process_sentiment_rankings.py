"""
Stock Sentiment Ranking Processor

Processes FinBERT sentiment results from real stock TextData
and generates final sentiment-based rankings.

For Lightning.ai: Use lightning_finbert_complete() function
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

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
    print(f"Applying {method} normalization...")
    
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
    
    # Verify normalization
    total = stock_scores['normalized_score'].sum()
    print(f"Normalized scores sum: {total:.6f}")
    
    return stock_scores.sort_values('normalized_score', ascending=False)

def lightning_finbert_complete(stock_identifiers=None, input_csv=None):
    """
    COMPLETE Lightning.ai FinBERT processing - everything in one function
    
    Args:
        stock_identifiers: List of stocks ['1004:01', '1045:01'] (optional)
        input_csv: Path to uploaded CSV file on Lightning.ai (optional, auto-detects)
    
    Returns:
        dict: Complete results with normalized sentiment rankings
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("Install transformers: pip install transformers torch")
    
    print("🚀 COMPLETE LIGHTNING.AI FINBERT PROCESSING")
    print("=" * 55)
    
    # Auto-detect input file if not specified
    if input_csv is None:
        csv_files = [f for f in os.listdir('.') if f.startswith('finbert_input_') and f.endswith('.csv')]
        if csv_files:
            input_csv = csv_files[0]
            print(f"📁 Using input file: {input_csv}")
        else:
            raise FileNotFoundError("Upload finbert_input_*.csv file to Lightning.ai Studio")
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"📊 Loaded {len(df)} texts for processing")
    
    # Initialize FinBERT
    print("⚡ Initializing FinBERT model...")
    finbert = pipeline("text-classification", model="ProsusAI/finbert", device=0)
    
    # Process all texts
    print("🧠 Running FinBERT sentiment analysis...")
    results = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing {idx+1}/{len(df)}...")
        
        result = finbert(str(row['text'])[:512])  # FinBERT max length
        
        results.append({
            'text_id': row.get('text_id', f"text_{idx}"),
            'stock_identifier': row.get('stock_identifier', f"{row.get('gvkey', 'UNK')}:01"),
            'gvkey': row.get('gvkey', 'UNK'),
            'confidence_score': result[0]['score'],
            'label': result[0]['label']
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply all 3 normalization methods
    print("📈 Applying enhanced normalization...")
    
    normalized_minmax = normalize_sentiment_scores(results_df, method='minmax')
    normalized_softmax = normalize_sentiment_scores(results_df, method='softmax')
    normalized_linear = normalize_sentiment_scores(results_df, method='linear')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_df.to_csv(f'finbert_results_{timestamp}.csv', index=False)
    normalized_minmax.to_csv(f'sentiment_rankings_minmax_{timestamp}.csv', index=False)
    normalized_softmax.to_csv(f'sentiment_rankings_softmax_{timestamp}.csv', index=False)
    normalized_linear.to_csv(f'sentiment_rankings_linear_{timestamp}.csv', index=False)
    
    print(f"\n✅ COMPLETE! Results saved:")
    print(f"- finbert_results_{timestamp}.csv")
    print(f"- sentiment_rankings_minmax_{timestamp}.csv") 
    print(f"- sentiment_rankings_softmax_{timestamp}.csv")
    print(f"- sentiment_rankings_linear_{timestamp}.csv")
    
    print(f"\n🎯 TOP SENTIMENT STOCKS (Min-Max):")
    print(normalized_minmax[['stock_identifier', 'normalized_score']].head())
    
    return {
        'raw_results': results_df,
        'minmax_rankings': normalized_minmax,
        'softmax_rankings': normalized_softmax,
        'linear_rankings': normalized_linear,
        'timestamp': timestamp
    }

def extract_real_textdata_for_stocks(stock_identifiers=['1004:01', '1045:01', '1050:01'], year_range=(2023, 2025)):
    """
    Extract real TextData for the specified stocks from the parquet files.
    This replaces mock data with actual SEC filing texts.
    
    Args:
        stock_identifiers: List of stock IDs in format 'GVKEY:IID' (e.g., ['1004:01', '1045:01'])
        year_range: Tuple of (start_year, end_year) to search (default: 2023-2025)
    """
    print(f"[INFO] Extracting real TextData for stocks: {stock_identifiers}")
    print(f"[INFO] Searching years: {year_range[0]} to {year_range[1]}")
    
    # Convert stock identifiers to GVKEY (TextData doesn't have IID, just GVKEY)
    target_gvkeys = []
    for stock_id in stock_identifiers:
        gvkey, iid = stock_id.split(':')
        # Try to convert to int, but keep as string if it fails (for BOT_01, TOP_01, etc.)
        try:
            target_gvkeys.append(int(gvkey))
        except ValueError:
            target_gvkeys.append(gvkey)
    
    print(f"  Target GVKEYs: {target_gvkeys}")
    
    all_texts = []
    
    # Extract from all years in range (2023, 2024, 2025)
    for year in range(year_range[0], year_range[1] + 1):
        parquet_path = f"TextData/{year}/text_us_{year}.parquet"
        if os.path.exists(parquet_path):
            print(f"  Loading {parquet_path}...")
            df = pd.read_parquet(parquet_path)
            
            # Filter for our target stocks
            for gvkey in target_gvkeys:
                stock_data = df[df['gvkey'] == gvkey].copy()
                if not stock_data.empty:
                    # Add IID column (default to '01' since TextData doesn't have IID)
                    stock_data['iid'] = '01'
                    # Add year column if it doesn't exist
                    if 'year' not in stock_data.columns:
                        stock_data['year'] = year
                    print(f"    Found {len(stock_data)} texts for GVKEY {gvkey} in {year}")
                    all_texts.append(stock_data)
        else:
            print(f"  [WARNING] File not found: {parquet_path}")
    
    if all_texts:
        combined_texts = pd.concat(all_texts, ignore_index=True)
        print(f"[SUCCESS] Extracted {len(combined_texts)} total texts from real TextData")
        print(f"  Stocks: {sorted(combined_texts['gvkey'].unique())}")
        print(f"  Years: {sorted(combined_texts['year'].unique())}")
        return combined_texts
    else:
        print("[WARNING] No real TextData found for specified stocks")
        return pd.DataFrame()

def process_sentiment_rankings(results_file=None, use_real_data=True, stock_identifiers=None):
    """
    Process FinBERT results and create sentiment rankings with enhanced normalization.
    
    Args:
        results_file: Path to FinBERT results CSV (if None, will look for default)
        use_real_data: If True, extract real TextData; if False, use existing files
        stock_identifiers: List of stock IDs from algorithm (e.g., ['1004:01', '1045:01'])
    """
    print("PROCESSING REAL STOCK SENTIMENT RANKINGS")
    print("=" * 60)
    
    # Load or prepare data for FinBERT processing
    if results_file is None:
        if use_real_data:
            print("[INFO] No FinBERT results provided. Extracting real TextData for processing...")
            # Use provided stock identifiers or default test stocks
            stocks_to_process = stock_identifiers if stock_identifiers is not None else ['1004:01', '1045:01', '1050:01']
            print(f"[INPUT] Processing stocks from algorithm: {stocks_to_process}")
            real_texts = extract_real_textdata_for_stocks(stock_identifiers=stocks_to_process)
            if not real_texts.empty:
                # Create input file for FinBERT processing on Lightning.ai
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                input_file = f"finbert_input_real_stocks_{timestamp}.csv"
                
                # Prepare texts for FinBERT (handles both rf/mgmt format and single text format)
                finbert_input = []
                for idx, row in real_texts.iterrows():
                    stock_id = f"{row['gvkey']}:{row['iid']}"
                    
                    # Handle 2023/2025 format with separate rf and mgmt columns
                    if 'rf' in row and pd.notna(row['rf']) and str(row['rf']).strip():
                        finbert_input.append({
                            'text_id': f"REAL_{row['gvkey']}_{row['iid']}_rf_{row['date']}_{len(finbert_input)+1:04d}",
                            'gvkey': row['gvkey'],
                            'iid': row['iid'],
                            'stock_identifier': stock_id,
                            'text': str(row['rf'])[:50000],  # Limit text length for FinBERT
                            'text_type': 'rf',
                            'date': row['date'],
                            'year': row['year'] if 'year' in row else row['date']//10000,
                            'filing_type': row.get('filing_type', row.get('file_type', '10K')),
                            'source': 'real_textdata'
                        })
                    
                    if 'mgmt' in row and pd.notna(row['mgmt']) and str(row['mgmt']).strip():
                        finbert_input.append({
                            'text_id': f"REAL_{row['gvkey']}_{row['iid']}_mgmt_{row['date']}_{len(finbert_input)+1:04d}",
                            'gvkey': row['gvkey'],
                            'iid': row['iid'], 
                            'stock_identifier': stock_id,
                            'text': str(row['mgmt'])[:50000],  # Limit text length for FinBERT
                            'text_type': 'mgmt',
                            'date': row['date'],
                            'year': row['year'] if 'year' in row else row['date']//10000,
                            'filing_type': row.get('filing_type', row.get('file_type', '10K')),
                            'source': 'real_textdata'
                        })
                    
                    # Handle 2024 format with single text column
                    elif 'text' in row and pd.notna(row['text']) and str(row['text']).strip():
                        finbert_input.append({
                            'text_id': f"REAL_{row['gvkey']}_{row['iid']}_text_{row['date']}_{len(finbert_input)+1:04d}",
                            'gvkey': row['gvkey'],
                            'iid': row['iid'],
                            'stock_identifier': stock_id,
                            'text': str(row['text'])[:50000],  # Limit text length for FinBERT
                            'text_type': 'general',
                            'date': row['date'],
                            'year': row['year'] if 'year' in row else row['date']//10000,
                            'filing_type': row.get('filing_type', row.get('file_type', '10K')),
                            'source': 'real_textdata'
                        })
                
                if finbert_input:
                    df_input = pd.DataFrame(finbert_input)
                    df_input.to_csv(input_file, index=False)
                    print(f"[SUCCESS] Created FinBERT input file: {input_file}")
                    print(f"[INFO] Ready for FinBERT processing: {len(df_input)} real texts from {df_input['stock_identifier'].nunique()} stocks")
                    print(f"[NEXT] Upload this file to Lightning.ai and run FinBERT, then process results with this function")
                    return df_input
                else:
                    print("[ERROR] No valid texts found in real TextData")
                    return None
            else:
                print("[ERROR] Could not extract real TextData")
                return None
        else:
            results_file = "finbert_results_real_stocks_20251004_185131.csv"
    
    # Try to find corresponding metadata file
    if results_file.endswith('.json'):
        metadata_file = results_file
        # Look for CSV file with similar name
        csv_name = results_file.replace('.json', '.csv')
        if 'finbert_results' in results_file:
            results_file = csv_name
        else:
            # This is metadata, we need to find the CSV
            print(f"Looking for CSV results file...")
            results_file = "finbert_results_real_stocks_20251004_185131.csv"
    else:
        # It's a CSV file, look for corresponding metadata
        metadata_file = results_file.replace('.csv', '.json').replace('finbert_results', 'metadata')
        if not metadata_file.endswith('.json'):
            metadata_file = "metadata_real_stocks_20251004_185131.json"
    
    try:
        df_results = pd.read_csv(results_file)
        print(f"[SUCCESS] Loaded {len(df_results)} sentiment results")
    except FileNotFoundError:
        print(f"[ERROR] Results file not found: {results_file}")
        print("Make sure you've run the FinBERT processing first!")
        return
    
    # Load metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"[SUCCESS] Loaded metadata for {len(metadata.get('input_stocks', []))} input stocks")
    except FileNotFoundError:
        print(f"[WARNING] Metadata file not found: {metadata_file}")
        metadata = {}
    
    # Convert FinBERT labels to numerical scores
    print("\n[INFO] Converting sentiment labels to scores...")
    
    df_results['positive_score'] = 0.0
    df_results['negative_score'] = 0.0
    df_results['neutral_score'] = 0.0
    
    # Handle flexible column naming for confidence/sentiment scores
    confidence_col = 'confidence_score' if 'confidence_score' in df_results.columns else 'sentiment_score'
    
    for idx, row in df_results.iterrows():
        label = row['sentiment_label'].lower()
        score = row.get(confidence_col, 0.5)  # Default to 0.5 if column missing
        
        if label == 'positive':
            df_results.at[idx, 'positive_score'] = score
            df_results.at[idx, 'neutral_score'] = (1 - score) * 0.6
            df_results.at[idx, 'negative_score'] = (1 - score) * 0.4
        elif label == 'negative':
            df_results.at[idx, 'negative_score'] = score
            df_results.at[idx, 'neutral_score'] = (1 - score) * 0.6
            df_results.at[idx, 'positive_score'] = (1 - score) * 0.4
        else:  # neutral
            df_results.at[idx, 'neutral_score'] = score
            df_results.at[idx, 'positive_score'] = (1 - score) * 0.5
            df_results.at[idx, 'negative_score'] = (1 - score) * 0.5
    
    # Calculate net sentiment and numeric sentiment score
    df_results['net_sentiment'] = df_results['positive_score'] - df_results['negative_score']
    
    # Create numeric sentiment score (-1 to +1 scale)
    # This gives a continuous numeric value where:
    # -1.0 = Most Negative, 0.0 = Neutral, +1.0 = Most Positive
    df_results['numeric_sentiment'] = df_results['net_sentiment']
    
    print(f"Sentiment distribution:")
    print(f"  Positive: {(df_results['sentiment_label'] == 'Positive').sum()}")
    print(f"  Negative: {(df_results['sentiment_label'] == 'Negative').sum()}")
    print(f"  Neutral: {(df_results['sentiment_label'] == 'Neutral').sum()}")
    
    # Aggregate by stock
    print("\n[INFO] Aggregating sentiment by stock...")
    
    stock_sentiment = df_results.groupby('stock_identifier').agg({
        'gvkey': 'first',
        'iid': 'first',
        'positive_score': 'mean',
        'negative_score': 'mean', 
        'neutral_score': 'mean',
        'net_sentiment': 'mean',
        'numeric_sentiment': 'mean',
        confidence_col: 'mean',
        'text_id': 'count',
        'text_type': lambda x: list(x.unique()),
        'year': lambda x: list(sorted(x.unique()))
    }).round(4)
    
    stock_sentiment.columns = ['gvkey', 'iid', 'avg_positive', 'avg_negative', 'avg_neutral',
                              'net_sentiment', 'numeric_sentiment', 'avg_confidence', 'text_count', 'text_types', 'years']
    stock_sentiment = stock_sentiment.reset_index()
    
    # Calculate weighted sentiment score
    # Weight by confidence and log of text count
    stock_sentiment['weighted_sentiment'] = (
        stock_sentiment['net_sentiment'] * 
        stock_sentiment['avg_confidence'] * 
        np.log1p(stock_sentiment['text_count'])
    )
    
    # NORMALIZATION: Create normalized scores that sum to 1 across all stocks
    print("\n[INFO] Calculating normalized sentiment scores...")
    
    # Method 1: Min-Max Normalization (0 to 1 scale)
    min_sentiment = stock_sentiment['numeric_sentiment'].min()
    max_sentiment = stock_sentiment['numeric_sentiment'].max()
    
    if max_sentiment == min_sentiment:
        # All stocks have same sentiment - equal weights
        stock_sentiment['normalized_score_minmax'] = 1.0 / len(stock_sentiment)
    else:
        # Scale to 0-1 range
        stock_sentiment['normalized_score_minmax'] = (
            (stock_sentiment['numeric_sentiment'] - min_sentiment) / 
            (max_sentiment - min_sentiment)
        )
        # Normalize so they sum to 1
        total = stock_sentiment['normalized_score_minmax'].sum()
        stock_sentiment['normalized_score_minmax'] = stock_sentiment['normalized_score_minmax'] / total
    
    # Method 2: Softmax Normalization (probability distribution)
    # This emphasizes differences between stocks
    sentiment_values = stock_sentiment['numeric_sentiment'].values
    exp_values = np.exp(sentiment_values - np.max(sentiment_values))  # Subtract max for numerical stability
    stock_sentiment['normalized_score_softmax'] = exp_values / np.sum(exp_values)
    
    # Method 3: Simple Linear Shift and Scale (centered around mean)
    mean_sentiment = stock_sentiment['numeric_sentiment'].mean()
    std_sentiment = stock_sentiment['numeric_sentiment'].std()
    
    if std_sentiment == 0:
        # All stocks have same sentiment
        stock_sentiment['normalized_score_linear'] = 1.0 / len(stock_sentiment)
    else:
        # Shift to positive range and normalize
        shifted_sentiment = stock_sentiment['numeric_sentiment'] - min_sentiment + 0.1  # Add small positive offset
        stock_sentiment['normalized_score_linear'] = shifted_sentiment / shifted_sentiment.sum()
    
    print(f"  [SUCCESS] Min-Max normalized scores (sum = {stock_sentiment['normalized_score_minmax'].sum():.3f})")
    print(f"  [SUCCESS] Softmax normalized scores (sum = {stock_sentiment['normalized_score_softmax'].sum():.3f})")
    print(f"  [SUCCESS] Linear normalized scores (sum = {stock_sentiment['normalized_score_linear'].sum():.3f})")
    
    # Add sentiment strength category
    def categorize_sentiment(net_sent):
        if net_sent > 0.2:
            return 'Strong Positive'
        elif net_sent > 0.05:
            return 'Positive'
        elif net_sent > -0.05:
            return 'Neutral'
        elif net_sent > -0.2:
            return 'Negative'
        else:
            return 'Strong Negative'
    
    stock_sentiment['sentiment_category'] = stock_sentiment['net_sentiment'].apply(categorize_sentiment)
    
    # Rank by weighted sentiment (best to worst)
    stock_sentiment = stock_sentiment.sort_values('weighted_sentiment', ascending=False)
    stock_sentiment['sentiment_rank'] = range(1, len(stock_sentiment) + 1)
    
    # Save detailed results
    output_file = f"stock_sentiment_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    stock_sentiment.to_csv(output_file, index=False)
    
    print(f"\n[FINAL] FINAL SENTIMENT RANKINGS WITH NORMALIZED SCORES")
    print("=" * 80)
    display_cols = ['sentiment_rank', 'stock_identifier', 'numeric_sentiment', 'normalized_score_minmax', 
                   'normalized_score_softmax', 'normalized_score_linear', 'sentiment_category', 'text_count']
    print(stock_sentiment[display_cols].to_string(index=False))
    
    print(f"\n[VALIDATION] NORMALIZATION VALIDATION:")
    print(f"  • Min-Max normalized sum: {stock_sentiment['normalized_score_minmax'].sum():.6f}")
    print(f"  • Softmax normalized sum: {stock_sentiment['normalized_score_softmax'].sum():.6f}")
    print(f"  • Linear normalized sum: {stock_sentiment['normalized_score_linear'].sum():.6f}")
    
    print(f"\n[INFO] SCORE RANGES:")
    print(f"  • Numeric sentiment: {stock_sentiment['numeric_sentiment'].min():.3f} to {stock_sentiment['numeric_sentiment'].max():.3f}")
    print(f"  • Min-Max normalized: {stock_sentiment['normalized_score_minmax'].min():.3f} to {stock_sentiment['normalized_score_minmax'].max():.3f}")
    print(f"  • Softmax normalized: {stock_sentiment['normalized_score_softmax'].min():.3f} to {stock_sentiment['normalized_score_softmax'].max():.3f}")
    print(f"  • Linear normalized: {stock_sentiment['normalized_score_linear'].min():.3f} to {stock_sentiment['normalized_score_linear'].max():.3f}")
    
    # Detailed analysis by text type
    print(f"\n📋 SENTIMENT BY TEXT TYPE")
    print("=" * 60)
    
    text_type_analysis = df_results.groupby(['stock_identifier', 'text_type']).agg({
        'net_sentiment': 'mean',
        'sentiment_score': 'mean',
        'text_id': 'count'
    }).round(4)
    
    print("Stock-wise sentiment by text type:")
    for stock_identifier in stock_sentiment['stock_identifier']:
        print(f"\n{stock_identifier}:")
        stock_texts = text_type_analysis.loc[stock_identifier] if stock_identifier in text_type_analysis.index else pd.DataFrame()
        if not stock_texts.empty:
            for text_type, row in stock_texts.iterrows():
                print(f"  {text_type}: {row['net_sentiment']:.4f} (confidence: {row['sentiment_score']:.4f}, texts: {row['text_id']})")
        else:
            print("  No text type breakdown available")
    
    # Summary statistics
    print(f"\n[INFO] SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total stocks analyzed: {len(stock_sentiment)}")
    print(f"Average net sentiment: {stock_sentiment['net_sentiment'].mean():.4f}")
    print(f"Sentiment range: {stock_sentiment['net_sentiment'].min():.4f} to {stock_sentiment['net_sentiment'].max():.4f}")
    print(f"Most positive stock: {stock_sentiment.iloc[0]['stock_identifier']} ({stock_sentiment.iloc[0]['net_sentiment']:.4f})")
    print(f"Most negative stock: {stock_sentiment.iloc[-1]['stock_identifier']} ({stock_sentiment.iloc[-1]['net_sentiment']:.4f})")
    
    # Sentiment distribution
    sentiment_dist = stock_sentiment['sentiment_category'].value_counts()
    print(f"\nSentiment distribution:")
    for category, count in sentiment_dist.items():
        print(f"  {category}: {count} stocks")
    
    print(f"\n[SUCCESS] Results saved to: {output_file}")
    
    # Create summary report
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'input_stocks': metadata.get('input_stocks', []),
        'total_stocks_analyzed': len(stock_sentiment),
        'total_texts_processed': len(df_results),
        'sentiment_rankings': stock_sentiment[['sentiment_rank', 'stock_identifier', 'net_sentiment', 'sentiment_category']].to_dict('records'),
        'summary_statistics': {
            'avg_sentiment': float(stock_sentiment['net_sentiment'].mean()),
            'sentiment_range': [float(stock_sentiment['net_sentiment'].min()), float(stock_sentiment['net_sentiment'].max())],
            'most_positive': stock_sentiment.iloc[0]['stock_identifier'],
            'most_negative': stock_sentiment.iloc[-1]['stock_identifier']
        },
        'sentiment_distribution': sentiment_dist.to_dict(),
        'output_file': output_file
    }
    
    summary_file = f"sentiment_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {summary_file}")
    print("\n[COMPLETE] Stock sentiment analysis complete!")
    
    return output_file, summary_file

def process_sentiment_rankings_complete(stock_identifiers):
    """
    Complete end-to-end sentiment processing: Algorithm stocks → SEC filings → FinBERT → Rankings
    
    Args:
        stock_identifiers: List of stock IDs from your algorithm (e.g., ['1004:01', '1045:01'])
        
    Returns:
        dict: Complete results with normalized sentiment rankings
    """
    print("END-TO-END SENTIMENT PROCESSING")
    print("=" * 50)
    print(f"Input from algorithm: {len(stock_identifiers)} stocks")
    print(f"Stocks to process: {stock_identifiers}")
    
    # Step 1: Extract SEC filings for algorithm stocks
    print("\nSTEP 1: Finding SEC filings for algorithm stocks...")
    real_texts = extract_real_textdata_for_stocks(stock_identifiers=stock_identifiers)
    
    if real_texts.empty:
        print("ERROR: No SEC filings found for provided stocks")
        return None
    
    # Step 2: Prepare for FinBERT processing  
    print(f"\nSTEP 2: Found {len(real_texts)} SEC filing texts")
    print("Preparing FinBERT input file...")
    
    # Generate FinBERT input file (for Lightning.ai or local processing)
    result = process_sentiment_rankings(stock_identifiers=stock_identifiers, use_real_data=True)
    
    print(f"\nCOMPLETE: FinBERT input file ready for processing")
    print("Next: Process on Lightning.ai or run local FinBERT")
    print("Final step: Apply normalization to get sentiment rankings (sum=1.0)")
    
    return {
        'input_stocks': stock_identifiers,
        'sec_filings_found': len(real_texts),
        'finbert_input_file': f"finbert_input_real_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        'status': 'ready_for_finbert_processing'
    }

if __name__ == "__main__":
    # Default test with 3 stocks
    test_stocks = ['1004:01', '1045:01', '1050:01']
    process_sentiment_rankings_complete(test_stocks)
