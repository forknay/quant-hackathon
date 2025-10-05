"""
COMPLETE SENTIMENT ANALYSIS PIPELINE - REAL TEXTDATA EXTRACTION
================================================================

⚠️  IMPORTANT: This script ALWAYS extracts from real TextData - NO pre-generated datasets!

All-in-one script for complete FinBERT sentiment analysis that works with ANY stock GVKEY:IID.

This script does EVERYTHING:
1. Takes stock symbols as input (from algorithm or manual input)
2. **ALWAYS extracts real SEC filing texts from TextData directory**
3. Runs FinBERT sentiment analysis with GPU/CPU fallback
4. Applies 3 enhanced normalization methods (all sum to 1.0)
5. Generates final sentiment rankings
6. Saves all results with timestamps

✅ WORKS WITH ANY STOCK: Just provide any GVKEY:IID and it will search TextData
✅ NO PREDETERMINED DATASETS: Always extracts fresh data from SEC filings
✅ LIGHTNING.AI READY: Upload TextData directory and this script

USAGE:
------
Local: python complete_sentiment_pipeline.py 1234:01 5678:01
Lightning.ai: Upload TextData/ directory + this script, then run

REQUIRED DATA:
- TextData/ directory with parquet files containing SEC filings
- Structure: TextData/YYYY/*.parquet (e.g., TextData/2023/batch_1.parquet)

INPUT OPTIONS:
- Command line: python complete_sentiment_pipeline.py STOCK1 STOCK2
- Algorithm: get_sentiment_rankings(['STOCK1:01', 'STOCK2:01'])
- Default: Uses STOCK_SYMBOLS list below if no stocks specified
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
    
    # Extract GVKEYs from stock identifiers and preserve original IID
    stock_mapping = {}
    gvkeys = []
    for stock_id in stock_identifiers:
        parts = stock_id.split(':')
        gvkey = parts[0]
        iid = parts[1] if len(parts) > 1 else '01'
        stock_mapping[gvkey] = f"{gvkey}:{iid}"
        gvkeys.append(gvkey)
    
    print(f"   [SEARCH] Searching for GVKEYs: {gvkeys} across years {year_range}")
    
    stocks_found = set()
    
    for year in range(year_range[0], year_range[1] + 1):
        year_path = os.path.join(textdata_path, str(year))
        if not os.path.exists(year_path):
            print(f"   [WARN] Year {year} not found, skipping...")
            continue
        
        # Find ALL parquet files for this year (comprehensive search)
        parquet_files = []
        for root, dirs, files in os.walk(year_path):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        print(f"   [SCAN] Scanning {len(parquet_files)} parquet files in {year}...")
        
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                
                # Check different possible column names for GVKEY
                gvkey_col = None
                for col in ['gvkey', 'GVKEY', 'gvkey_id', 'stock_id']:
                    if col in df.columns:
                        gvkey_col = col
                        break
                
                if gvkey_col is None:
                    continue  # Skip files without GVKEY column
                
                # Handle both string and float GVKEYs
                if df[gvkey_col].dtype == 'float64':
                    # Convert target GVKEYs to float for comparison
                    target_gvkeys_float = []
                    for gvkey in gvkeys:
                        try:
                            target_gvkeys_float.append(float(gvkey))
                        except ValueError:
                            target_gvkeys_float.append(gvkey)
                    matched_data = df[df[gvkey_col].isin(target_gvkeys_float)]
                else:
                    # Convert to string for comparison
                    df[gvkey_col] = df[gvkey_col].astype(str)
                    matched_data = df[df[gvkey_col].isin(gvkeys)]
                
                if len(matched_data) > 0:
                    print(f"   [FOUND] Found {len(matched_data)} texts in {os.path.basename(file_path)}")
                    
                    for gvkey in gvkeys:
                        # Handle both string and float comparison
                        if df[gvkey_col].dtype == 'float64':
                            try:
                                gvkey_val = float(gvkey)
                            except ValueError:
                                gvkey_val = gvkey
                        else:
                            gvkey_val = gvkey
                        stock_data = matched_data[matched_data[gvkey_col] == gvkey_val]
                        
                        if len(stock_data) > 0:
                            stocks_found.add(gvkey)
                            # Limit texts per stock to avoid overwhelming the system
                            stock_data = stock_data.head(max_texts_per_stock)
                            
                            # Extract text from multiple columns (rf = Risk Factors, mgmt = Management Discussion)
                            text_columns = ['rf', 'mgmt', 'text', 'content', 'filing_text', 'TEXT', 'CONTENT']
                            available_text_cols = [col for col in text_columns if col in stock_data.columns]
                            
                            if not available_text_cols:
                                print(f"   [WARN] No text columns found in {os.path.basename(file_path)}")
                                continue
                            
                            for _, row in stock_data.iterrows():
                                # Combine text from all available text columns
                                combined_text = ""
                                for text_col in available_text_cols:
                                    text_content = str(row.get(text_col, ''))
                                    if text_content and text_content != 'nan' and len(text_content.strip()) > 10:
                                        combined_text += text_content + " "
                                
                                if len(combined_text.strip()) > 50:  # Ensure we have meaningful text
                                    all_texts.append({
                                        'text_id': f"{gvkey_val}_{year}_{len(all_texts)}",
                                        'stock_identifier': stock_mapping[gvkey],
                                        'gvkey': str(gvkey_val),
                                        'year': year,
                                        'text': combined_text.strip(),
                                        'filename': os.path.basename(file_path),
                                        'file_path': file_path,
                                        'text_columns_used': available_text_cols
                                    })
                
            except Exception as e:
                print(f"   [ERROR] Error processing {os.path.basename(file_path)}: {str(e)}")
                continue
    
    print(f"   [RESULTS] Found data for {len(stocks_found)} out of {len(gvkeys)} requested stocks")
    print(f"   [STOCKS_FOUND] {list(stocks_found)}")
    
    if len(stocks_found) == 0:
        available_stocks = []
        # Quick scan to show available stocks
        try:
            sample_file = parquet_files[0] if parquet_files else None
            if sample_file:
                sample_df = pd.read_parquet(sample_file)
                if 'gvkey' in sample_df.columns:
                    available_stocks = sample_df['gvkey'].astype(str).unique()[:10].tolist()
        except:
            pass
        
        error_msg = f"No texts found for any of the requested stocks {gvkeys} in years {year_range}"
        if available_stocks:
            error_msg += f"\nSample available stocks in TextData: {available_stocks}"
        raise ValueError(error_msg)
    
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
    
    # ALWAYS extract from real TextData - NO pre-generated datasets
    if stock_identifiers:
        print(f"[STOCKS] Processing stocks: {stock_identifiers}")
        texts_df = extract_textdata_for_stocks(stock_identifiers, max_texts_per_stock=max_texts_per_stock)
    else:
        print(f"[DEFAULT] Using default stocks: {STOCK_SYMBOLS}")
        texts_df = extract_textdata_for_stocks(STOCK_SYMBOLS, max_texts_per_stock=max_texts_per_stock)
    
    print(f"[EXTRACTED] Successfully extracted {len(texts_df)} texts from real TextData")
    print(f"[STOCKS_FOUND] Texts per stock: {dict(texts_df['stock_identifier'].value_counts())}")
    
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

def process_dual_stock_lists(positive_outlook_stocks, negative_outlook_stocks, max_texts_per_stock=50):
    """
    DUAL-LIST PIPELINE: Process two separate lists of stocks (positive vs negative outlook)
    
    Args:
        positive_outlook_stocks: List of stocks with positive outlook ['31846:01', '62169:01']
        negative_outlook_stocks: List of stocks with negative outlook ['10349:01', '7906:01']
        max_texts_per_stock: Maximum texts to process per stock
    
    Returns:
        dict: Complete results for both lists with separate rankings
    """
    print("[DUAL-LIST PIPELINE] PROCESSING POSITIVE & NEGATIVE OUTLOOK STOCKS")
    print("=" * 65)
    print(f"[START] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'positive_outlook': None,
        'negative_outlook': None,
        'combined_summary': {},
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # Process Positive Outlook Stocks
    if positive_outlook_stocks:
        print(f"\n🟢 PROCESSING POSITIVE OUTLOOK STOCKS: {positive_outlook_stocks}")
        print("-" * 50)
        
        pos_texts_df = extract_textdata_for_stocks(positive_outlook_stocks, max_texts_per_stock=max_texts_per_stock)
        print(f"[POSITIVE] Extracted {len(pos_texts_df)} texts from {len(pos_texts_df['stock_identifier'].unique())} positive outlook stocks")
        
        pos_results_df = run_finbert_analysis(pos_texts_df)
        
        # Apply normalization for positive stocks
        print("[POSITIVE] Generating sentiment rankings...")
        pos_minmax = normalize_sentiment_scores(pos_results_df, method='minmax')
        pos_softmax = normalize_sentiment_scores(pos_results_df, method='softmax') 
        pos_linear = normalize_sentiment_scores(pos_results_df, method='linear')
        
        results['positive_outlook'] = {
            'raw_results': pos_results_df,
            'minmax_rankings': pos_minmax,
            'softmax_rankings': pos_softmax,
            'linear_rankings': pos_linear,
            'stocks_processed': positive_outlook_stocks,
            'texts_extracted': len(pos_texts_df),
            'average_confidence': float(pos_results_df['confidence_score'].mean())
        }
        
        # Save positive results
        timestamp = results['timestamp']
        pos_results_df.to_csv(f'finbert_results_positive_{timestamp}.csv', index=False)
        pos_minmax.to_csv(f'sentiment_rankings_positive_minmax_{timestamp}.csv', index=False)
        pos_softmax.to_csv(f'sentiment_rankings_positive_softmax_{timestamp}.csv', index=False)
        pos_linear.to_csv(f'sentiment_rankings_positive_linear_{timestamp}.csv', index=False)
    
    # Process Negative Outlook Stocks
    if negative_outlook_stocks:
        print(f"\n🔴 PROCESSING NEGATIVE OUTLOOK STOCKS: {negative_outlook_stocks}")
        print("-" * 50)
        
        neg_texts_df = extract_textdata_for_stocks(negative_outlook_stocks, max_texts_per_stock=max_texts_per_stock)
        print(f"[NEGATIVE] Extracted {len(neg_texts_df)} texts from {len(neg_texts_df['stock_identifier'].unique())} negative outlook stocks")
        
        neg_results_df = run_finbert_analysis(neg_texts_df)
        
        # Apply normalization for negative stocks
        print("[NEGATIVE] Generating sentiment rankings...")
        neg_minmax = normalize_sentiment_scores(neg_results_df, method='minmax')
        neg_softmax = normalize_sentiment_scores(neg_results_df, method='softmax')
        neg_linear = normalize_sentiment_scores(neg_results_df, method='linear')
        
        results['negative_outlook'] = {
            'raw_results': neg_results_df,
            'minmax_rankings': neg_minmax,
            'softmax_rankings': neg_softmax,
            'linear_rankings': neg_linear,
            'stocks_processed': negative_outlook_stocks,
            'texts_extracted': len(neg_texts_df),
            'average_confidence': float(neg_results_df['confidence_score'].mean())
        }
        
        # Save negative results
        timestamp = results['timestamp']
        neg_results_df.to_csv(f'finbert_results_negative_{timestamp}.csv', index=False)
        neg_minmax.to_csv(f'sentiment_rankings_negative_minmax_{timestamp}.csv', index=False)
        neg_softmax.to_csv(f'sentiment_rankings_negative_softmax_{timestamp}.csv', index=False)
        neg_linear.to_csv(f'sentiment_rankings_negative_linear_{timestamp}.csv', index=False)
    
    # Create combined summary
    results['combined_summary'] = {
        'timestamp': timestamp,
        'positive_outlook_stocks': positive_outlook_stocks if positive_outlook_stocks else [],
        'negative_outlook_stocks': negative_outlook_stocks if negative_outlook_stocks else [],
        'positive_texts_processed': results['positive_outlook']['texts_extracted'] if results['positive_outlook'] else 0,
        'negative_texts_processed': results['negative_outlook']['texts_extracted'] if results['negative_outlook'] else 0,
        'positive_avg_confidence': results['positive_outlook']['average_confidence'] if results['positive_outlook'] else 0,
        'negative_avg_confidence': results['negative_outlook']['average_confidence'] if results['negative_outlook'] else 0
    }
    
    # Save combined summary
    with open(f'dual_sentiment_summary_{timestamp}.json', 'w') as f:
        json.dump(results['combined_summary'], f, indent=2)
    
    # Display results
    print(f"\n✅ DUAL-LIST PIPELINE COMPLETE!")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['positive_outlook']:
        print(f"\n🟢 POSITIVE OUTLOOK RESULTS:")
        print(f"📊 Processed {results['positive_outlook']['texts_extracted']} texts from {len(positive_outlook_stocks)} stocks")
        print(f"📈 Average confidence: {results['positive_outlook']['average_confidence']:.3f}")
        print("Top sentiment stocks (Min-Max):")
        print(results['positive_outlook']['minmax_rankings'][['stock_identifier', 'normalized_score']].head())
        print(f"Sum: {results['positive_outlook']['minmax_rankings']['normalized_score'].sum():.6f}")
    
    if results['negative_outlook']:
        print(f"\n🔴 NEGATIVE OUTLOOK RESULTS:")
        print(f"📊 Processed {results['negative_outlook']['texts_extracted']} texts from {len(negative_outlook_stocks)} stocks")
        print(f"📈 Average confidence: {results['negative_outlook']['average_confidence']:.3f}")
        print("Top sentiment stocks (Min-Max):")
        print(results['negative_outlook']['minmax_rankings'][['stock_identifier', 'normalized_score']].head())
        print(f"Sum: {results['negative_outlook']['minmax_rankings']['normalized_score'].sum():.6f}")
    
    print(f"\n📁 Files created:")
    if results['positive_outlook']:
        print(f"- finbert_results_positive_{timestamp}.csv")
        print(f"- sentiment_rankings_positive_*_{timestamp}.csv")
    if results['negative_outlook']:
        print(f"- finbert_results_negative_{timestamp}.csv")
        print(f"- sentiment_rankings_negative_*_{timestamp}.csv")
    print(f"- dual_sentiment_summary_{timestamp}.json")
    
    return results

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
        # Check if it's dual-list mode (--dual flag)
        if '--dual' in sys.argv:
            print("[DUAL-MODE] Testing dual-list processing...")
            # Test with known stocks that exist in TextData
            positive_stocks = ['31846:01']  # Known to exist
            negative_stocks = ['62169:01']  # Known to exist
            
            print(f"[TEST] Positive outlook stocks: {positive_stocks}")
            print(f"[TEST] Negative outlook stocks: {negative_stocks}")
            
            results = process_dual_stock_lists(positive_stocks, negative_stocks)
            
        else:
            # Single-list mode: python complete_sentiment_pipeline.py STOCK1 STOCK2 STOCK3
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

def get_dual_sentiment_rankings(positive_outlook_stocks, negative_outlook_stocks, normalization_method='minmax'):
    """
    DUAL-LIST ALGORITHM INTEGRATION: Get separate sentiment rankings for positive and negative outlook stocks
    
    This is the function your algorithm should call when you have two separate lists of stocks.
    
    Args:
        positive_outlook_stocks: List of stocks with positive outlook ['31846:01', '62169:01']
        negative_outlook_stocks: List of stocks with negative outlook ['10349:01', '7906:01']
        normalization_method: 'minmax', 'softmax', or 'linear'
    
    Returns:
        dict: Separate sentiment rankings for both lists
            {
                'positive_rankings': DataFrame (normalized, sum=1.0),
                'negative_rankings': DataFrame (normalized, sum=1.0),
                'summary': dict with processing details
            }
    """
    print(f"[DUAL-ALGORITHM] PROCESSING POSITIVE ({len(positive_outlook_stocks)}) & NEGATIVE ({len(negative_outlook_stocks)}) OUTLOOK STOCKS...")
    
    # Validate inputs
    pos_stocks = validate_stock_input(positive_outlook_stocks) if positive_outlook_stocks else []
    neg_stocks = validate_stock_input(negative_outlook_stocks) if negative_outlook_stocks else []
    
    # Run dual-list pipeline
    results = process_dual_stock_lists(pos_stocks, neg_stocks)
    
    # Extract rankings based on normalization method
    output = {
        'positive_rankings': None,
        'negative_rankings': None,
        'summary': results['combined_summary']
    }
    
    if results['positive_outlook']:
        if normalization_method == 'minmax':
            output['positive_rankings'] = results['positive_outlook']['minmax_rankings']
        elif normalization_method == 'softmax':
            output['positive_rankings'] = results['positive_outlook']['softmax_rankings']
        elif normalization_method == 'linear':
            output['positive_rankings'] = results['positive_outlook']['linear_rankings']
        else:
            print(f"[WARN] Unknown normalization method '{normalization_method}', using minmax for positive")
            output['positive_rankings'] = results['positive_outlook']['minmax_rankings']
    
    if results['negative_outlook']:
        if normalization_method == 'minmax':
            output['negative_rankings'] = results['negative_outlook']['minmax_rankings']
        elif normalization_method == 'softmax':
            output['negative_rankings'] = results['negative_outlook']['softmax_rankings']
        elif normalization_method == 'linear':
            output['negative_rankings'] = results['negative_outlook']['linear_rankings']
        else:
            print(f"[WARN] Unknown normalization method '{normalization_method}', using minmax for negative")
            output['negative_rankings'] = results['negative_outlook']['minmax_rankings']
    
    return output

# EXAMPLE USAGE FOR ALGORITHMS:
"""
# Example 1: Single-list algorithm integration (existing functionality)

from complete_sentiment_pipeline import get_sentiment_rankings

# Your algorithm generates a list of stocks to analyze
algorithm_stocks = ['31846:01', '62169:01', '10349:01']

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

# Example 2: Dual-list algorithm integration (NEW FEATURE)

from complete_sentiment_pipeline import get_dual_sentiment_rankings

# Your algorithm generates TWO separate lists
positive_outlook_stocks = ['31846:01', '10349:01']  # Stocks with positive outlook
negative_outlook_stocks = ['62169:01', '7906:01']   # Stocks with negative outlook

# Get SEPARATE sentiment rankings for each list
dual_results = get_dual_sentiment_rankings(
    positive_outlook_stocks, 
    negative_outlook_stocks, 
    normalization_method='softmax'
)

# Use positive outlook rankings (sum=1.0)
print("POSITIVE OUTLOOK SENTIMENT RANKINGS:")
for _, row in dual_results['positive_rankings'].iterrows():
    stock = row['stock_identifier']
    weight = row['normalized_score']
    print(f"  {stock}: Weight = {weight:.4f}")

# Use negative outlook rankings (sum=1.0)  
print("NEGATIVE OUTLOOK SENTIMENT RANKINGS:")
for _, row in dual_results['negative_rankings'].iterrows():
    stock = row['stock_identifier']
    weight = row['normalized_score']
    print(f"  {stock}: Weight = {weight:.4f}")

# Combined usage in trading strategy
def your_trading_algorithm():
    # Get separate sentiment analysis for positive and negative outlook stocks
    results = get_dual_sentiment_rankings(
        positive_outlook_stocks=['31846:01', '10349:01'], 
        negative_outlook_stocks=['62169:01', '7906:01']
    )
    
    # Process positive outlook stocks
    for _, row in results['positive_rankings'].iterrows():
        stock = row['stock_identifier']
        sentiment_weight = row['normalized_score'] 
        # Apply positive bias to your position sizing
        position_size = calculate_positive_position(stock, sentiment_weight)
        execute_trade(stock, position_size, bias='positive')
    
    # Process negative outlook stocks
    for _, row in results['negative_rankings'].iterrows():
        stock = row['stock_identifier']
        sentiment_weight = row['normalized_score']
        # Apply negative bias to your position sizing  
        position_size = calculate_negative_position(stock, sentiment_weight)
        execute_trade(stock, position_size, bias='negative')
    
    return results
"""