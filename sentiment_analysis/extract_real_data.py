"""
Real Stock Data Extraction Pipeline

This script uses actual performance data and TextData to:
1. Load real stock performance from ret_sample.csv
2. Identify top 5 and bottom 5 performing stocks by GVKEY  
3. Extract corresponding SEC filing text from TextData (2023/2025)
4. Process real financial text through FinBERT for sentiment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
import json

class RealStockDataExtractor:
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
            'n_top_stocks': 5,
            'n_bottom_stocks': 5,
            'text_years': [2023, 2025],  # Use years with real data (not 2024 which has mock data)
            'performance_metric': 'ret_12_1',  # 12-month return
            'min_text_length': 1000,  # Minimum text length for meaningful analysis
            'max_text_length': 50000,  # Maximum text length for FinBERT processing
            'date_range_months': 24  # Look for text data within 24 months of performance date
        }
        
    def load_stock_performance(self) -> pd.DataFrame:
        """Load and process stock performance data"""
        self.logger.info("Loading stock performance data...")
        
        perf_file = self.data_path / "ret_sample.csv"
        if not perf_file.exists():
            raise FileNotFoundError(f"Performance file not found: {perf_file}")
        
        self.logger.info(f"Reading performance data from {perf_file} (this may take a moment...)")
        
        # Read in chunks to handle large file efficiently
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(perf_file, chunksize=chunk_size):
            # Filter for recent data and valid performance metrics
            chunk = chunk.dropna(subset=['gvkey', self.config['performance_metric']])
            
            # Convert date column to datetime
            if 'date' in chunk.columns:
                chunk['date'] = pd.to_datetime(chunk['date'])
                # Use recent data (last 2 years)
                recent_date = chunk['date'].max() - timedelta(days=730)
                chunk = chunk[chunk['date'] >= recent_date]
            
            if len(chunk) > 0:
                chunks.append(chunk)
        
        if not chunks:
            raise ValueError("No valid performance data found")
            
        # Combine chunks
        df_perf = pd.concat(chunks, ignore_index=True)
        
        self.logger.info(f"Loaded performance data: {len(df_perf)} records")
        self.logger.info(f"Unique stocks: {df_perf['gvkey'].nunique()}")
        self.logger.info(f"Date range: {df_perf['date'].min()} to {df_perf['date'].max()}")
        
        return df_perf
    
    def select_top_bottom_stocks(self, df_perf: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Select top 5 and bottom 5 performing stocks by GVKEY"""
        self.logger.info("Selecting top and bottom performing stocks...")
        
        # Aggregate performance by stock (GVKEY) - use mean of recent performance
        stock_performance = df_perf.groupby('gvkey').agg({
            self.config['performance_metric']: ['mean', 'count'],
            'date': 'max'
        }).reset_index()
        
        # Flatten column names
        stock_performance.columns = ['gvkey', 'avg_return', 'obs_count', 'latest_date']
        
        # Filter stocks with sufficient observations
        min_obs = 5
        stock_performance = stock_performance[stock_performance['obs_count'] >= min_obs]
        
        # Sort by performance
        stock_performance = stock_performance.sort_values('avg_return', ascending=False)
        
        self.logger.info(f"Performance summary for {len(stock_performance)} stocks:")
        self.logger.info(f"  Top return: {stock_performance['avg_return'].max():.4f}")
        self.logger.info(f"  Bottom return: {stock_performance['avg_return'].min():.4f}")
        self.logger.info(f"  Median return: {stock_performance['avg_return'].median():.4f}")
        
        # Select top and bottom performers
        top_stocks = stock_performance.head(self.config['n_top_stocks'])['gvkey'].tolist()
        bottom_stocks = stock_performance.tail(self.config['n_bottom_stocks'])['gvkey'].tolist()
        
        self.logger.info(f"Selected stocks:")
        self.logger.info(f"  Top 5: {top_stocks}")
        self.logger.info(f"  Bottom 5: {bottom_stocks}")
        
        # Log performance details
        for i, gvkey in enumerate(top_stocks, 1):
            perf = stock_performance[stock_performance['gvkey'] == gvkey]['avg_return'].iloc[0]
            self.logger.info(f"    Top #{i}: GVKEY {gvkey} = {perf:.4f}")
            
        for i, gvkey in enumerate(bottom_stocks, 1):
            perf = stock_performance[stock_performance['gvkey'] == gvkey]['avg_return'].iloc[0]
            self.logger.info(f"    Bottom #{i}: GVKEY {gvkey} = {perf:.4f}")
        
        return top_stocks, bottom_stocks
    
    def extract_text_data_for_stocks(self, top_stocks: List[int], bottom_stocks: List[int]) -> pd.DataFrame:
        """Extract text data from TextData files for selected stocks"""
        self.logger.info("Extracting text data for selected stocks...")
        
        all_stocks = top_stocks + bottom_stocks
        all_texts = []
        
        for year in self.config['text_years']:
            year_path = self.textdata_path / str(year)
            if not year_path.exists():
                self.logger.warning(f"TextData year {year} not found: {year_path}")
                continue
                
            # Find parquet files for this year
            parquet_files = list(year_path.glob("*.parquet"))
            if not parquet_files:
                self.logger.warning(f"No parquet files found for year {year}")
                continue
                
            self.logger.info(f"Processing {len(parquet_files)} files for year {year}")
            
            for parquet_file in parquet_files:
                try:
                    self.logger.info(f"  Reading {parquet_file.name}...")
                    
                    # Read file in chunks to handle memory efficiently
                    df_text = pd.read_parquet(parquet_file, engine='pyarrow')
                    self.logger.info(f"    Loaded {len(df_text)} records")
                    
                    # Filter for our target stocks
                    if 'gvkey' in df_text.columns:
                        # Convert gvkey to int if it's not already
                        df_text['gvkey'] = pd.to_numeric(df_text['gvkey'], errors='coerce')
                        df_text = df_text[df_text['gvkey'].isin(all_stocks)]
                        
                        self.logger.info(f"    Found {len(df_text)} records for target stocks")
                        
                        if len(df_text) == 0:
                            continue
                            
                        # Process text columns (rf = Risk Factors, mgmt = Management Discussion)
                        text_columns = ['rf', 'mgmt']
                        existing_text_cols = [col for col in text_columns if col in df_text.columns]
                        
                        if not existing_text_cols:
                            self.logger.warning(f"    No text columns found in {parquet_file.name}")
                            continue
                            
                        # Extract text for each stock and text type
                        for _, row in df_text.iterrows():
                            gvkey = row['gvkey']
                            date = row.get('date', f"{year}-01-01")
                            
                            # Determine stock category
                            if gvkey in top_stocks:
                                stock_category = 'top_performer'
                                stock_rank = top_stocks.index(gvkey) + 1
                            else:
                                stock_category = 'bottom_performer'
                                stock_rank = bottom_stocks.index(gvkey) + 1
                            
                            # Process each text column
                            for text_col in existing_text_cols:
                                text_content = row[text_col]
                                
                                if pd.isna(text_content) or len(str(text_content).strip()) < self.config['min_text_length']:
                                    continue
                                    
                                text_content = str(text_content).strip()
                                
                                # Truncate if too long for FinBERT
                                if len(text_content) > self.config['max_text_length']:
                                    text_content = text_content[:self.config['max_text_length']]
                                
                                # Create text entry
                                text_entry = {
                                    'text_id': f"GVKEY_{gvkey}_{text_col}_{year}_{len(all_texts)+1:03d}",
                                    'stock_id': f"GVKEY_{gvkey}",
                                    'gvkey': gvkey,
                                    'stock_category': stock_category,
                                    'stock_rank': stock_rank,
                                    'text': text_content,
                                    'text_length': len(text_content),
                                    'text_type': text_col,
                                    'date': date,
                                    'year': year,
                                    'filing_type': row.get('file_type', 'unknown'),
                                    'source': f"{year}_real_data"
                                }
                                
                                all_texts.append(text_entry)
                                
                                self.logger.info(f"      Added {text_col} text for GVKEY {gvkey}: {len(text_content)} chars")
                        
                except Exception as e:
                    self.logger.error(f"  Error processing {parquet_file}: {e}")
                    continue
        
        if not all_texts:
            raise ValueError("No text data found for selected stocks")
            
        df_texts = pd.DataFrame(all_texts)
        
        # Summary statistics
        self.logger.info(f"\nText extraction summary:")
        self.logger.info(f"  Total texts extracted: {len(df_texts)}")
        self.logger.info(f"  Unique stocks: {df_texts['gvkey'].nunique()}")
        self.logger.info(f"  Text types: {df_texts['text_type'].value_counts().to_dict()}")
        self.logger.info(f"  Categories: {df_texts['stock_category'].value_counts().to_dict()}")
        self.logger.info(f"  Average text length: {df_texts['text_length'].mean():.0f} chars")
        self.logger.info(f"  Text length range: {df_texts['text_length'].min()}-{df_texts['text_length'].max()}")
        
        return df_texts
    
    def format_for_finbert(self, df_texts: pd.DataFrame) -> pd.DataFrame:
        """Format extracted text data for FinBERT processing"""
        self.logger.info("Formatting data for FinBERT...")
        
        # Create FinBERT input format
        finbert_data = df_texts.copy()
        
        # Ensure text is clean and properly formatted
        finbert_data['text'] = finbert_data['text'].str.replace(r'\n+', ' ', regex=True)
        finbert_data['text'] = finbert_data['text'].str.replace(r'\s+', ' ', regex=True)
        finbert_data['text'] = finbert_data['text'].str.strip()
        
        # Select columns for FinBERT
        output_columns = [
            'text_id', 'stock_id', 'gvkey', 'stock_category', 'stock_rank',
            'text', 'text_length', 'text_type', 'date', 'year', 'filing_type', 'source'
        ]
        
        finbert_data = finbert_data[output_columns]
        
        self.logger.info(f"FinBERT data prepared: {len(finbert_data)} texts")
        
        return finbert_data
    
    def export_for_lightning_ai(self, df_finbert: pd.DataFrame) -> Dict[str, str]:
        """Export formatted data for Lightning.ai processing"""
        self.logger.info("Exporting data for Lightning.ai...")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export main FinBERT input file
        input_file = self.output_path / f"finbert_real_data_{timestamp}.csv"
        df_finbert.to_csv(input_file, index=False)
        
        # Export text-only file for simple processing
        texts_file = self.output_path / f"texts_only_real_{timestamp}.txt"
        with open(texts_file, 'w', encoding='utf-8') as f:
            for _, row in df_finbert.iterrows():
                f.write(f"=== {row['text_id']} ===\n")
                f.write(f"{row['text']}\n\n")
        
        # Create metadata file
        metadata = {
            'timestamp': timestamp,
            'total_texts': len(df_finbert),
            'unique_stocks': df_finbert['gvkey'].nunique(),
            'stock_categories': df_finbert['stock_category'].value_counts().to_dict(),
            'text_types': df_finbert['text_type'].value_counts().to_dict(),
            'avg_text_length': float(df_finbert['text_length'].mean()),
            'text_length_stats': {
                'min': int(df_finbert['text_length'].min()),
                'max': int(df_finbert['text_length'].max()),
                'median': int(df_finbert['text_length'].median())
            },
            'years_covered': sorted(df_finbert['year'].unique().tolist()),
            'performance_metric_used': self.config['performance_metric'],
            'data_source': 'real_textdata_and_performance'
        }
        
        metadata_file = self.output_path / f"metadata_real_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create Lightning.ai instructions
        instructions = self.create_lightning_instructions(timestamp, len(df_finbert))
        instructions_file = self.output_path / f"lightning_instructions_real_{timestamp}.md"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        # Update results processor for real data
        self.create_real_results_processor(timestamp)
        
        output_files = {
            'input_file': str(input_file),
            'texts_file': str(texts_file),
            'metadata_file': str(metadata_file),
            'instructions_file': str(instructions_file),
            'processor_file': str(self.output_path / "process_real_results.py")
        }
        
        self.logger.info(f"Files exported:")
        for file_type, file_path in output_files.items():
            size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
            self.logger.info(f"  {file_type}: {Path(file_path).name} ({size:,} bytes)")
        
        return output_files
    
    def create_lightning_instructions(self, timestamp: str, num_texts: int) -> str:
        """Create Lightning.ai processing instructions"""
        return f"""# FinBERT Sentiment Analysis - Real Data Processing

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data:** {num_texts} real SEC filing texts from actual stock performance data

## Lightning.ai Processing Steps

### 1. Upload Files to Lightning.ai Studio
- `finbert_real_data_{timestamp}.csv` (main input data)
- `process_real_results.py` (results processor)

### 2. Install Dependencies
```bash
pip install transformers torch pandas numpy
```

### 3. Run FinBERT Processing
```python
import pandas as pd
from transformers import pipeline
import torch

# Load FinBERT model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    device=0 if torch.cuda.is_available() else -1,
    max_length=512,
    truncation=True
)

# Load real data
df = pd.read_csv('finbert_real_data_{timestamp}.csv')
print(f"Processing {{len(df)}} real SEC filing texts...")

# Process each text
results = []
for idx, row in df.iterrows():
    result = sentiment_pipeline(row['text'])[0]
    results.append({{
        'text_id': row['text_id'],
        'gvkey': row['gvkey'],
        'stock_category': row['stock_category'],
        'sentiment_label': result['label'],
        'sentiment_score': result['score']
    }})
    
    if (idx + 1) % 5 == 0:
        print(f"Processed {{idx + 1}}/{{len(df)}} texts")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('finbert_real_results_{timestamp}.csv', index=False)
print("✅ Real data processing complete!")
```

### 4. Process Final Rankings
```bash
python process_real_results.py
```

## Expected Output
- Real sentiment scores from actual SEC filings
- Stock rankings based on genuine financial text sentiment
- Performance correlation analysis between returns and sentiment

**Note:** This uses REAL TextData from your dataset, not simulated data.
"""

    def create_real_results_processor(self, timestamp: str):
        """Create results processor for real data"""
        processor_code = f'''"""
Real Data Results Processor

Processes FinBERT sentiment results from real SEC filing data
and generates final stock rankings with performance correlation analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def process_real_results():
    """Process FinBERT results from real data"""
    print("🚀 Processing Real FinBERT Results")
    print("=" * 50)
    
    # Load results
    results_file = "finbert_real_results_{timestamp}.csv"
    metadata_file = "metadata_real_{timestamp}.json"
    
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
        print(f"✅ Loaded metadata")
    except FileNotFoundError:
        print(f"⚠️  Metadata file not found: {{metadata_file}}")
        metadata = {{}}
    
    # Convert FinBERT labels to numerical scores
    df_results['positive_score'] = 0.0
    df_results['negative_score'] = 0.0
    df_results['neutral_score'] = 0.0
    
    for idx, row in df_results.iterrows():
        label = row['sentiment_label'].lower()
        score = row['sentiment_score']
        
        if label == 'positive':
            df_results.at[idx, 'positive_score'] = score
            df_results.at[idx, 'neutral_score'] = (1 - score) * 0.7
            df_results.at[idx, 'negative_score'] = (1 - score) * 0.3
        elif label == 'negative':
            df_results.at[idx, 'negative_score'] = score
            df_results.at[idx, 'neutral_score'] = (1 - score) * 0.7
            df_results.at[idx, 'positive_score'] = (1 - score) * 0.3
        else:  # neutral
            df_results.at[idx, 'neutral_score'] = score
            df_results.at[idx, 'positive_score'] = (1 - score) * 0.5
            df_results.at[idx, 'negative_score'] = (1 - score) * 0.5
    
    # Calculate net sentiment
    df_results['net_sentiment'] = df_results['positive_score'] - df_results['negative_score']
    
    # Aggregate by stock (GVKEY)
    stock_sentiment = df_results.groupby(['gvkey', 'stock_category']).agg({{
        'positive_score': 'mean',
        'negative_score': 'mean',
        'neutral_score': 'mean',
        'net_sentiment': 'mean',
        'sentiment_score': 'mean',
        'text_id': 'count'
    }}).round(4)
    
    stock_sentiment.columns = ['avg_positive', 'avg_negative', 'avg_neutral', 
                              'net_sentiment', 'avg_confidence', 'text_count']
    stock_sentiment = stock_sentiment.reset_index()
    
    # Calculate weighted sentiment (considering confidence and text count)
    stock_sentiment['weighted_sentiment'] = (
        stock_sentiment['net_sentiment'] * 
        stock_sentiment['avg_confidence'] * 
        np.log1p(stock_sentiment['text_count'])  # Log transform text count
    )
    
    # Rank stocks by sentiment
    stock_sentiment = stock_sentiment.sort_values('weighted_sentiment', ascending=False)
    stock_sentiment['sentiment_rank'] = range(1, len(stock_sentiment) + 1)
    
    # Save detailed results
    output_file = f"real_stock_sentiment_rankings_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.csv"
    stock_sentiment.to_csv(output_file, index=False)
    
    print(f"\\n📊 FINAL REAL DATA SENTIMENT RANKINGS")
    print("=" * 50)
    print(stock_sentiment[['gvkey', 'stock_category', 'net_sentiment', 
                          'weighted_sentiment', 'text_count', 'sentiment_rank']].to_string(index=False))
    
    # Performance vs Sentiment Analysis
    print(f"\\n🔍 PERFORMANCE VS SENTIMENT ANALYSIS")
    print("=" * 50)
    
    top_performers = stock_sentiment[stock_sentiment['stock_category'] == 'top_performer']
    bottom_performers = stock_sentiment[stock_sentiment['stock_category'] == 'bottom_performer']
    
    print(f"Top Performers Sentiment - Mean: {{top_performers['net_sentiment'].mean():.4f}}")
    print(f"Bottom Performers Sentiment - Mean: {{bottom_performers['net_sentiment'].mean():.4f}}")
    
    sentiment_diff = top_performers['net_sentiment'].mean() - bottom_performers['net_sentiment'].mean()
    print(f"Sentiment Difference (Top - Bottom): {{sentiment_diff:.4f}}")
    
    if sentiment_diff > 0:
        print("✅ Expected result: Top performers have more positive sentiment")
    else:
        print("⚠️  Unexpected: Bottom performers have more positive sentiment")
    
    print(f"\\n💾 Results saved to: {{output_file}}")
    
    # Create summary report
    summary = {{
        'processing_timestamp': datetime.now().isoformat(),
        'total_stocks_analyzed': len(stock_sentiment),
        'total_texts_processed': df_results['text_id'].nunique(),
        'sentiment_statistics': {{
            'top_performer_avg_sentiment': float(top_performers['net_sentiment'].mean()),
            'bottom_performer_avg_sentiment': float(bottom_performers['net_sentiment'].mean()),
            'sentiment_difference': float(sentiment_diff),
            'correlation_direction': 'positive' if sentiment_diff > 0 else 'negative'
        }},
        'metadata': metadata,
        'output_file': output_file
    }}
    
    summary_file = f"real_sentiment_analysis_summary_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {{summary_file}}")
    print("\\n🎯 Real sentiment analysis complete!")
    
    return output_file, summary_file

if __name__ == "__main__":
    process_real_results()
'''
        
        processor_file = self.output_path / "process_real_results.py"
        with open(processor_file, 'w') as f:
            f.write(processor_code)
            
    def run_complete_pipeline(self) -> Dict[str, str]:
        """Run the complete real data extraction pipeline"""
        self.logger.info("🚀 Starting Real Stock Data Extraction Pipeline")
        
        try:
            # Step 1: Load stock performance data
            df_perf = self.load_stock_performance()
            
            # Step 2: Select top and bottom performers
            top_stocks, bottom_stocks = self.select_top_bottom_stocks(df_perf)
            
            # Step 3: Extract real text data
            df_texts = self.extract_text_data_for_stocks(top_stocks, bottom_stocks)
            
            # Step 4: Format for FinBERT
            df_finbert = self.format_for_finbert(df_texts)
            
            # Step 5: Export for Lightning.ai
            output_files = self.export_for_lightning_ai(df_finbert)
            
            self.logger.info("✅ Real data extraction pipeline completed successfully!")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {{e}}")
            raise

def main():
    """Main execution function"""
    extractor = RealStockDataExtractor()
    
    print("=" * 60)
    print("REAL STOCK DATA EXTRACTION PIPELINE")
    print("=" * 60)
    print("This pipeline uses:")
    print("- Real stock performance data from ret_sample.csv")
    print("- Actual SEC filing text from TextData/2023 and TextData/2025")
    print("- GVKEY-based mapping between performance and text data")
    print("=" * 60)
    
    try:
        output_files = extractor.run_complete_pipeline()
        
        print("\\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\\nFiles created for Lightning.ai:")
        for file_type, file_path in output_files.items():
            print(f"  - {{file_type}}: {{Path(file_path).name}}")
        
        print("\\nNext steps:")
        print("1. Upload files to Lightning.ai")
        print("2. Follow the markdown instructions")
        print("3. Run FinBERT on real SEC filing data")
        print("4. Get genuine sentiment analysis results!")
        
    except Exception as e:
        print(f"\\n❌ Pipeline failed: {{e}}")
        print("Check the logs above for details.")

if __name__ == "__main__":
    main()