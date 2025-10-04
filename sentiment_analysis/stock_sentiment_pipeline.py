"""
Input-Based Stock Sentiment Analysis Pipeline

This pipeline:
1. Receives pre-selected stocks (GVKEY + IID pairs) as input
2. Matches them to their corresponding TextData using GVKEY and IID for 2023-2025  
3. Extracts real SEC filing text from TextData
4. Processes through FinBERT for sentiment analysis
5. Ranks stocks by sentiment scores from their actual TextData

Usage:
    # For testing with real stocks found in data
    python stock_sentiment_pipeline.py --find-test-stocks
    
    # For processing provided stocks
    python stock_sentiment_pipeline.py --stocks "1004:01,1013:01,1019:01"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
import json
import argparse
import sys

class StockSentimentPipeline:
    def __init__(self, base_path: str = "C:/Users/positive/Documents/GitHub/quant-hackathon"):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "Data" 
        self.textdata_path = self.base_path / "TextData"
        self.output_path = self.base_path / "sentiment_analysis" / "lightning_ai"
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'text_years': [2023, 2024, 2025],  # Process these years
            'min_text_length': 1000,  # Minimum meaningful text length
            'max_text_length': 50000,  # Maximum for FinBERT processing  
            'date_range_months': 24,  # Look within 24 months for recent filings
            'text_columns': ['rf', 'mgmt', 'text'],  # Text columns to extract
            'max_texts_per_stock': 10  # Limit texts per stock for processing efficiency
        }
        
    def find_real_test_stocks(self, n_stocks: int = 10) -> List[Tuple[int, str]]:
        """Find real stocks that exist in both Data and TextData for testing"""
        self.logger.info(f"Finding {n_stocks} real test stocks...")
        
        # Get GVKEYs from recent TextData 
        textdata_gvkeys = set()
        
        for year in [2023, 2025]:  # Skip 2024 which has mock data
            year_path = self.textdata_path / str(year)
            if not year_path.exists():
                continue
                
            parquet_files = list(year_path.glob("*.parquet"))
            if not parquet_files:
                continue
                
            try:
                # Sample from first file to get GVKEYs
                df_sample = pd.read_parquet(parquet_files[0], engine='pyarrow')
                if 'gvkey' in df_sample.columns:
                    # Convert to numeric and remove NaN
                    gvkeys = pd.to_numeric(df_sample['gvkey'], errors='coerce').dropna()
                    textdata_gvkeys.update(gvkeys.astype(int).unique())
                    self.logger.info(f"Found {len(gvkeys.unique())} unique GVKEYs in {year}")
            except Exception as e:
                self.logger.warning(f"Error reading {year} data: {e}")
        
        self.logger.info(f"Total unique GVKEYs in TextData: {len(textdata_gvkeys)}")
        
        # Get GVKEYs + IIDs from Data folder (link table)
        link_file = self.data_path / "cik_gvkey_linktable_USA_only.csv"
        data_stocks = []
        
        if link_file.exists():
            try:
                # Read sample to get GVKEY-IID pairs
                df_link = pd.read_csv(link_file, nrows=10000)
                
                if 'gvkey' in df_link.columns and 'iid' in df_link.columns:
                    # Filter for stocks that exist in TextData
                    df_link = df_link[df_link['gvkey'].isin(textdata_gvkeys)]
                    
                    # Get unique GVKEY-IID pairs
                    for _, row in df_link.iterrows():
                        gvkey = int(row['gvkey'])
                        iid = str(row['iid'])
                        if gvkey in textdata_gvkeys:
                            data_stocks.append((gvkey, iid))
                            
                self.logger.info(f"Found {len(data_stocks)} GVKEY-IID pairs in Data")
                
            except Exception as e:
                self.logger.error(f"Error reading link table: {e}")
                
        # If we don't have enough from link table, use TextData GVKEYs with default IID
        if len(data_stocks) < n_stocks:
            additional_needed = n_stocks - len(data_stocks)
            textdata_list = list(textdata_gvkeys)[:additional_needed]
            
            for gvkey in textdata_list:
                if (gvkey, '01') not in data_stocks:  # Default IID
                    data_stocks.append((gvkey, '01'))
        
        # Select final test stocks
        test_stocks = data_stocks[:n_stocks]
        
        self.logger.info(f"Selected {len(test_stocks)} test stocks:")
        for i, (gvkey, iid) in enumerate(test_stocks, 1):
            self.logger.info(f"  {i}. GVKEY: {gvkey}, IID: {iid}")
            
        return test_stocks
    
    def parse_stock_input(self, stock_string: str) -> List[Tuple[int, str]]:
        """Parse stock input string like '1004:01,1013:01,1019:01' into GVKEY-IID pairs"""
        stocks = []
        
        try:
            for stock_pair in stock_string.split(','):
                stock_pair = stock_pair.strip()
                if ':' in stock_pair:
                    gvkey_str, iid = stock_pair.split(':', 1)
                    gvkey = int(gvkey_str)
                    stocks.append((gvkey, iid))
                else:
                    # Assume it's just GVKEY with default IID
                    gvkey = int(stock_pair)
                    stocks.append((gvkey, '01'))
                    
        except ValueError as e:
            raise ValueError(f"Invalid stock input format: {e}. Use format like '1004:01,1013:01'")
            
        self.logger.info(f"Parsed {len(stocks)} stocks from input:")
        for gvkey, iid in stocks:
            self.logger.info(f"  GVKEY: {gvkey}, IID: {iid}")
            
        return stocks
    
    def extract_textdata_for_stocks(self, stock_pairs: List[Tuple[int, str]]) -> pd.DataFrame:
        """Extract TextData for given GVKEY-IID pairs from recent years"""
        self.logger.info(f"Extracting TextData for {len(stock_pairs)} stocks...")
        
        all_texts = []
        
        # Convert stock pairs to sets for efficient lookup
        target_gvkeys = set(gvkey for gvkey, _ in stock_pairs)
        stock_dict = {gvkey: iid for gvkey, iid in stock_pairs}
        
        for year in self.config['text_years']:
            year_path = self.textdata_path / str(year)
            if not year_path.exists():
                self.logger.warning(f"TextData year {year} not found")
                continue
                
            parquet_files = list(year_path.glob("*.parquet"))
            if not parquet_files:
                self.logger.warning(f"No parquet files found for year {year}")
                continue
                
            self.logger.info(f"Processing year {year}: {len(parquet_files)} files")
            
            for parquet_file in parquet_files:
                try:
                    self.logger.info(f"  Reading {parquet_file.name}...")
                    
                    # Read the parquet file
                    df_text = pd.read_parquet(parquet_file, engine='pyarrow')
                    self.logger.info(f"    Loaded {len(df_text)} records")
                    
                    # Filter for our target stocks by GVKEY
                    if 'gvkey' in df_text.columns:
                        # Convert gvkey to numeric
                        df_text['gvkey'] = pd.to_numeric(df_text['gvkey'], errors='coerce')
                        df_text = df_text[df_text['gvkey'].isin(target_gvkeys)]
                        
                        self.logger.info(f"    Found {len(df_text)} records for target stocks")
                        
                        if len(df_text) == 0:
                            continue
                        
                        # Check for IID matching if IID column exists
                        if 'iid' in df_text.columns:
                            # Filter by IID as well
                            matched_records = []
                            for _, row in df_text.iterrows():
                                gvkey = row['gvkey']
                                row_iid = str(row.get('iid', '01'))  # Default to '01'
                                expected_iid = stock_dict.get(gvkey, '01')
                                
                                # Match if IIDs match or we're flexible about IID
                                if row_iid == expected_iid or expected_iid == '01':
                                    matched_records.append(row)
                            
                            if matched_records:
                                df_text = pd.DataFrame(matched_records)
                                self.logger.info(f"    After IID filtering: {len(df_text)} records")
                        
                        # Find available text columns
                        available_text_cols = [col for col in self.config['text_columns'] if col in df_text.columns]
                        
                        if not available_text_cols:
                            self.logger.warning(f"    No text columns found in {parquet_file.name}")
                            continue
                            
                        self.logger.info(f"    Available text columns: {available_text_cols}")
                        
                        # Extract text for each record
                        texts_extracted = 0
                        for _, row in df_text.iterrows():
                            gvkey = int(row['gvkey'])
                            iid = str(row.get('iid', '01'))
                            date = row.get('date', f"{year}-01-01")
                            
                            # Process each available text column
                            for text_col in available_text_cols:
                                text_content = row[text_col]
                                
                                # Skip if text is empty or too short
                                if pd.isna(text_content):
                                    continue
                                    
                                text_content = str(text_content).strip()
                                if len(text_content) < self.config['min_text_length']:
                                    continue
                                
                                # Truncate if too long for FinBERT
                                if len(text_content) > self.config['max_text_length']:
                                    text_content = text_content[:self.config['max_text_length']]
                                
                                # Create text entry
                                text_entry = {
                                    'text_id': f"REAL_{gvkey}_{iid}_{text_col}_{year}_{len(all_texts)+1:04d}",
                                    'gvkey': gvkey,
                                    'iid': iid,
                                    'stock_identifier': f"{gvkey}:{iid}",
                                    'text': text_content,
                                    'text_length': len(text_content),
                                    'text_type': text_col,
                                    'date': str(date),
                                    'year': year,
                                    'filing_type': row.get('file_type', row.get('filing_type', 'unknown')),
                                    'source': f"real_textdata_{year}"
                                }
                                
                                all_texts.append(text_entry)
                                texts_extracted += 1
                                
                                # Limit texts per stock for efficiency
                                stock_text_count = sum(1 for t in all_texts if t['gvkey'] == gvkey)
                                if stock_text_count >= self.config['max_texts_per_stock']:
                                    break
                        
                        self.logger.info(f"    Extracted {texts_extracted} texts from {parquet_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"  Error processing {parquet_file}: {e}")
                    continue
        
        if not all_texts:
            raise ValueError("No text data found for the provided stocks")
            
        df_texts = pd.DataFrame(all_texts)
        
        # Summary statistics
        self.logger.info(f"\n📊 TEXT EXTRACTION SUMMARY:")
        self.logger.info(f"  Total texts extracted: {len(df_texts)}")
        self.logger.info(f"  Unique stocks: {df_texts['gvkey'].nunique()}")
        self.logger.info(f"  Years covered: {sorted(df_texts['year'].unique())}")
        self.logger.info(f"  Text types: {df_texts['text_type'].value_counts().to_dict()}")
        self.logger.info(f"  Average text length: {df_texts['text_length'].mean():.0f} chars")
        self.logger.info(f"  Text length range: {df_texts['text_length'].min()}-{df_texts['text_length'].max()}")
        
        # Show texts per stock
        texts_per_stock = df_texts.groupby('stock_identifier').size().sort_values(ascending=False)
        self.logger.info(f"\n📋 TEXTS PER STOCK:")
        for stock_id, count in texts_per_stock.items():
            self.logger.info(f"  {stock_id}: {count} texts")
        
        return df_texts
    
    def format_for_finbert(self, df_texts: pd.DataFrame) -> pd.DataFrame:
        """Format extracted text data for FinBERT processing"""
        self.logger.info("Formatting data for FinBERT...")
        
        # Clean text content
        df_finbert = df_texts.copy()
        
        # Clean and normalize text
        df_finbert['text'] = df_finbert['text'].str.replace(r'\n+', ' ', regex=True)
        df_finbert['text'] = df_finbert['text'].str.replace(r'\s+', ' ', regex=True)
        df_finbert['text'] = df_finbert['text'].str.strip()
        
        # Recalculate text length after cleaning
        df_finbert['text_length'] = df_finbert['text'].str.len()
        
        # Select final columns for FinBERT
        output_columns = [
            'text_id', 'gvkey', 'iid', 'stock_identifier', 
            'text', 'text_length', 'text_type', 
            'date', 'year', 'filing_type', 'source'
        ]
        
        df_finbert = df_finbert[output_columns]
        
        self.logger.info(f"FinBERT data prepared: {len(df_finbert)} texts")
        self.logger.info(f"Average cleaned text length: {df_finbert['text_length'].mean():.0f} chars")
        
        return df_finbert
    
    def export_for_lightning_ai(self, df_finbert: pd.DataFrame, input_stocks: List[Tuple[int, str]]) -> Dict[str, str]:
        """Export formatted data for Lightning.ai processing"""
        self.logger.info("Exporting data for Lightning.ai...")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export main FinBERT input file
        input_file = self.output_path / f"finbert_real_stocks_{timestamp}.csv"
        df_finbert.to_csv(input_file, index=False)
        
        # Export text-only file
        texts_file = self.output_path / f"texts_only_real_{timestamp}.txt"
        with open(texts_file, 'w', encoding='utf-8') as f:
            for _, row in df_finbert.iterrows():
                f.write(f"=== {row['text_id']} ({row['stock_identifier']}) ===\n")
                f.write(f"Type: {row['text_type']} | Year: {row['year']} | Length: {row['text_length']}\n")
                f.write(f"{row['text']}\n\n")
        
        # Create metadata
        metadata = {
            'timestamp': timestamp,
            'input_stocks': [f"{gvkey}:{iid}" for gvkey, iid in input_stocks],
            'total_texts': len(df_finbert),
            'unique_stocks': df_finbert['gvkey'].nunique(),
            'texts_per_stock': df_finbert.groupby('stock_identifier').size().to_dict(),
            'text_types': df_finbert['text_type'].value_counts().to_dict(),
            'years_covered': sorted(df_finbert['year'].unique().tolist()),
            'avg_text_length': float(df_finbert['text_length'].mean()),
            'text_length_stats': {
                'min': int(df_finbert['text_length'].min()),
                'max': int(df_finbert['text_length'].max()), 
                'median': int(df_finbert['text_length'].median())
            },
            'processing_config': self.config,
            'data_source': 'real_textdata_input_stocks'
        }
        
        metadata_file = self.output_path / f"metadata_real_stocks_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create Lightning.ai instructions
        instructions = self.create_lightning_instructions(timestamp, len(df_finbert), input_stocks)
        instructions_file = self.output_path / f"lightning_instructions_real_{timestamp}.md"
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        # Create sentiment ranking processor
        self.create_sentiment_ranking_processor(timestamp)
        
        output_files = {
            'input_file': str(input_file),
            'texts_file': str(texts_file),
            'metadata_file': str(metadata_file),
            'instructions_file': str(instructions_file),
            'processor_file': str(self.output_path / "process_sentiment_rankings.py")
        }
        
        self.logger.info(f"\n📁 FILES EXPORTED:")
        for file_type, file_path in output_files.items():
            size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
            self.logger.info(f"  {file_type}: {Path(file_path).name} ({size:,} bytes)")
        
        return output_files
    
    def create_lightning_instructions(self, timestamp: str, num_texts: int, input_stocks: List[Tuple[int, str]]) -> str:
        """Create Lightning.ai processing instructions"""
        stock_list = ", ".join([f"{gvkey}:{iid}" for gvkey, iid in input_stocks])
        
        return f"""# Real Stock Sentiment Analysis - Lightning.ai Processing

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Input Stocks:** {stock_list}
**Total Texts:** {num_texts} real SEC filing texts

## Processing Steps

### 1. Upload Files to Lightning.ai Studio
- `finbert_real_stocks_{timestamp}.csv` (main input data)
- `process_sentiment_rankings.py` (sentiment ranking processor)

### 2. Install Dependencies
```bash
pip install transformers torch pandas numpy
```

### 3. Run FinBERT Processing
```python
import pandas as pd
from transformers import pipeline
import torch

# Load FinBERT model for financial sentiment
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    device=0 if torch.cuda.is_available() else -1,
    max_length=512,
    truncation=True
)

# Load real stock data
df = pd.read_csv('finbert_real_stocks_{timestamp}.csv')
print(f"Processing {{len(df)}} texts from real stocks: {stock_list}")

# Process each text through FinBERT
results = []
for idx, row in df.iterrows():
    try:
        result = sentiment_pipeline(row['text'])[0]
        
        results.append({{
            'text_id': row['text_id'],
            'gvkey': row['gvkey'],
            'iid': row['iid'], 
            'stock_identifier': row['stock_identifier'],
            'text_type': row['text_type'],
            'sentiment_label': result['label'],
            'sentiment_score': result['score'],
            'text_length': row['text_length'],
            'year': row['year']
        }})
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {{idx + 1}}/{{len(df)}} texts")
            
    except Exception as e:
        print(f"Error processing {{row['text_id']}}: {{e}}")
        results.append({{
            'text_id': row['text_id'],
            'gvkey': row['gvkey'],
            'iid': row['iid'],
            'stock_identifier': row['stock_identifier'], 
            'text_type': row['text_type'],
            'sentiment_label': 'neutral',
            'sentiment_score': 0.5,
            'text_length': row['text_length'],
            'year': row['year']
        }})

# Save FinBERT results
results_df = pd.DataFrame(results)
results_df.to_csv('finbert_results_real_stocks_{timestamp}.csv', index=False)
print(f"\\n✅ FinBERT processing complete! Results saved.")
print(f"📊 Processed {{len(results_df)}} texts from {{results_df['stock_identifier'].nunique()}} stocks")
```

### 4. Generate Sentiment Rankings
```bash
python process_sentiment_rankings.py
```

## Expected Output
- **Stock sentiment rankings** based on real SEC filing text
- **Detailed analysis** of sentiment by text type (Risk Factors vs Management Discussion)
- **Confidence scores** and text statistics for each stock

**Note:** This processes REAL TextData from years 2023-2025, not simulated data.
"""

    def create_sentiment_ranking_processor(self, timestamp: str):
        """Create sentiment ranking processor"""
        processor_code = f'''"""
Stock Sentiment Ranking Processor

Processes FinBERT sentiment results from real stock TextData
and generates final sentiment-based rankings.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def process_sentiment_rankings():
    """Process FinBERT results and create sentiment rankings"""
    print("🚀 PROCESSING REAL STOCK SENTIMENT RANKINGS")
    print("=" * 60)
    
    # Load FinBERT results
    results_file = "finbert_results_real_stocks_{timestamp}.csv"
    metadata_file = "metadata_real_stocks_{timestamp}.json"
    
    try:
        df_results = pd.read_csv(results_file)
        print(f"✅ Loaded {{len(df_results)}} sentiment results")
    except FileNotFoundError:
        print(f"❌ Results file not found: {{results_file}}")
        print("Make sure you've run the FinBERT processing first!")
        return
    
    # Load metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata for {{len(metadata.get('input_stocks', []))}} input stocks")
    except FileNotFoundError:
        print(f"⚠️  Metadata file not found: {{metadata_file}}")
        metadata = {{}}
    
    # Convert FinBERT labels to numerical scores
    print("\\n📊 Converting sentiment labels to scores...")
    
    df_results['positive_score'] = 0.0
    df_results['negative_score'] = 0.0
    df_results['neutral_score'] = 0.0
    
    for idx, row in df_results.iterrows():
        label = row['sentiment_label'].lower()
        score = row['sentiment_score']
        
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
    
    # Calculate net sentiment
    df_results['net_sentiment'] = df_results['positive_score'] - df_results['negative_score']
    
    print(f"Sentiment distribution:")
    print(f"  Positive: {{(df_results['sentiment_label'] == 'Positive').sum()}}")
    print(f"  Negative: {{(df_results['sentiment_label'] == 'Negative').sum()}}")
    print(f"  Neutral: {{(df_results['sentiment_label'] == 'Neutral').sum()}}")
    
    # Aggregate by stock
    print("\\n📈 Aggregating sentiment by stock...")
    
    stock_sentiment = df_results.groupby('stock_identifier').agg({{
        'gvkey': 'first',
        'iid': 'first',
        'positive_score': 'mean',
        'negative_score': 'mean', 
        'neutral_score': 'mean',
        'net_sentiment': 'mean',
        'sentiment_score': 'mean',
        'text_id': 'count',
        'text_type': lambda x: list(x.unique()),
        'year': lambda x: list(sorted(x.unique()))
    }}).round(4)
    
    stock_sentiment.columns = ['gvkey', 'iid', 'avg_positive', 'avg_negative', 'avg_neutral',
                              'net_sentiment', 'avg_confidence', 'text_count', 'text_types', 'years']
    stock_sentiment = stock_sentiment.reset_index()
    
    # Calculate weighted sentiment score
    # Weight by confidence and log of text count
    stock_sentiment['weighted_sentiment'] = (
        stock_sentiment['net_sentiment'] * 
        stock_sentiment['avg_confidence'] * 
        np.log1p(stock_sentiment['text_count'])
    )
    
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
    output_file = f"stock_sentiment_rankings_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.csv"
    stock_sentiment.to_csv(output_file, index=False)
    
    print(f"\\n🏆 FINAL SENTIMENT RANKINGS")
    print("=" * 60)
    display_cols = ['sentiment_rank', 'stock_identifier', 'net_sentiment', 
                   'sentiment_category', 'weighted_sentiment', 'text_count']
    print(stock_sentiment[display_cols].to_string(index=False))
    
    # Detailed analysis by text type
    print(f"\\n📋 SENTIMENT BY TEXT TYPE")
    print("=" * 60)
    
    text_type_analysis = df_results.groupby(['stock_identifier', 'text_type']).agg({{
        'net_sentiment': 'mean',
        'sentiment_score': 'mean',
        'text_id': 'count'
    }}).round(4)
    
    print("Stock-wise sentiment by text type:")
    for stock_identifier in stock_sentiment['stock_identifier']:
        print(f"\\n{{stock_identifier}}:")
        stock_texts = text_type_analysis.loc[stock_identifier] if stock_identifier in text_type_analysis.index else pd.DataFrame()
        if not stock_texts.empty:
            for text_type, row in stock_texts.iterrows():
                print(f"  {{text_type}}: {{row['net_sentiment']:.4f}} (confidence: {{row['sentiment_score']:.4f}}, texts: {{row['text_id']}})")
        else:
            print("  No text type breakdown available")
    
    # Summary statistics
    print(f"\\n📊 SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total stocks analyzed: {{len(stock_sentiment)}}")
    print(f"Average net sentiment: {{stock_sentiment['net_sentiment'].mean():.4f}}")
    print(f"Sentiment range: {{stock_sentiment['net_sentiment'].min():.4f}} to {{stock_sentiment['net_sentiment'].max():.4f}}")
    print(f"Most positive stock: {{stock_sentiment.iloc[0]['stock_identifier']}} ({{stock_sentiment.iloc[0]['net_sentiment']:.4f}})")
    print(f"Most negative stock: {{stock_sentiment.iloc[-1]['stock_identifier']}} ({{stock_sentiment.iloc[-1]['net_sentiment']:.4f}})")
    
    # Sentiment distribution
    sentiment_dist = stock_sentiment['sentiment_category'].value_counts()
    print(f"\\nSentiment distribution:")
    for category, count in sentiment_dist.items():
        print(f"  {{category}}: {{count}} stocks")
    
    print(f"\\n💾 Results saved to: {{output_file}}")
    
    # Create summary report
    summary = {{
        'processing_timestamp': datetime.now().isoformat(),
        'input_stocks': metadata.get('input_stocks', []),
        'total_stocks_analyzed': len(stock_sentiment),
        'total_texts_processed': len(df_results),
        'sentiment_rankings': stock_sentiment[['sentiment_rank', 'stock_identifier', 'net_sentiment', 'sentiment_category']].to_dict('records'),
        'summary_statistics': {{
            'avg_sentiment': float(stock_sentiment['net_sentiment'].mean()),
            'sentiment_range': [float(stock_sentiment['net_sentiment'].min()), float(stock_sentiment['net_sentiment'].max())],
            'most_positive': stock_sentiment.iloc[0]['stock_identifier'],
            'most_negative': stock_sentiment.iloc[-1]['stock_identifier']
        }},
        'sentiment_distribution': sentiment_dist.to_dict(),
        'output_file': output_file
    }}
    
    summary_file = f"sentiment_analysis_summary_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {{summary_file}}")
    print("\\n🎯 Stock sentiment analysis complete!")
    
    return output_file, summary_file

if __name__ == "__main__":
    process_sentiment_rankings()
'''
        
        processor_file = self.output_path / "process_sentiment_rankings.py"
        with open(processor_file, 'w', encoding='utf-8') as f:
            f.write(processor_code)
    
    def run_pipeline(self, stock_pairs: List[Tuple[int, str]]) -> Dict[str, str]:
        """Run the complete pipeline for given stocks"""
        self.logger.info("🚀 Starting Real Stock Sentiment Analysis Pipeline")
        self.logger.info(f"Input stocks: {[f'{gvkey}:{iid}' for gvkey, iid in stock_pairs]}")
        
        try:
            # Extract TextData for the stocks
            df_texts = self.extract_textdata_for_stocks(stock_pairs)
            
            # Format for FinBERT
            df_finbert = self.format_for_finbert(df_texts)
            
            # Export for Lightning.ai
            output_files = self.export_for_lightning_ai(df_finbert, stock_pairs)
            
            self.logger.info("✅ Pipeline completed successfully!")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Stock Sentiment Analysis Pipeline')
    parser.add_argument('--find-test-stocks', action='store_true', 
                       help='Find real test stocks from data')
    parser.add_argument('--stocks', type=str,
                       help='Comma-separated GVKEY:IID pairs (e.g., "1004:01,1013:01")')
    parser.add_argument('--n-test-stocks', type=int, default=10,
                       help='Number of test stocks to find (default: 10)')
    
    args = parser.parse_args()
    
    pipeline = StockSentimentPipeline()
    
    print("=" * 60)
    print("STOCK SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    print("Order of operations:")
    print("1. Receive pre-selected stocks (GVKEY:IID pairs)")
    print("2. Match to TextData using GVKEY and IID for 2023-2025")
    print("3. Extract real SEC filing text")
    print("4. Process through FinBERT for sentiment analysis")
    print("5. Rank stocks by sentiment scores")
    print("=" * 60)
    
    try:
        if args.find_test_stocks:
            # Find real test stocks
            print(f"\\nFinding {args.n_test_stocks} real test stocks...")
            stock_pairs = pipeline.find_real_test_stocks(args.n_test_stocks)
        elif args.stocks:
            # Parse provided stocks
            print(f"\\nProcessing provided stocks: {args.stocks}")
            stock_pairs = pipeline.parse_stock_input(args.stocks)
        else:
            print("\\n❌ Please provide either --find-test-stocks or --stocks argument")
            print("Examples:")
            print("  python stock_sentiment_pipeline.py --find-test-stocks")
            print("  python stock_sentiment_pipeline.py --stocks '1004:01,1013:01,1019:01'")
            return
        
        if not stock_pairs:
            print("❌ No valid stocks found or provided")
            return
            
        # Run the pipeline
        output_files = pipeline.run_pipeline(stock_pairs)
        
        print("\\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\\nFiles created for Lightning.ai:")
        for file_type, file_path in output_files.items():
            print(f"  - {file_type}: {Path(file_path).name}")
        
        print("\\nNext steps:")
        print("1. Upload files to Lightning.ai Studio")
        print("2. Follow instructions in the markdown file")
        print("3. Run FinBERT on real SEC filing data")
        print("4. Generate sentiment-based stock rankings!")
        
    except Exception as e:
        print(f"\\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()