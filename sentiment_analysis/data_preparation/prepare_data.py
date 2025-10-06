"""
FinBERT Sentiment Analysis - Data Preparation Module

This module handles the extraction and preparation of stock text data 
for sentiment analysis using FinBERT on Lightning.ai.

Author: GitHub Copilot
Date: October 4, 2025
Purpose: Quant Hackathon - Sentiment Analysis Pipeline
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockTextExtractor:
    """
    Extracts text data for selected stocks from TextData folders
    """
    
    def __init__(self, textdata_dir: str = "../../TextData"):
        self.textdata_dir = Path(__file__).parent.parent.parent / "TextData"
        logger.info(f"TextData directory: {self.textdata_dir}")
        
    def get_available_stocks(self) -> Dict[str, List]:
        """
        Scan TextData folders and return available stocks
        
        Returns:
            Dictionary with stocks from 2023 and 2024 data
        """
        available_stocks = {
            '2023': [],
            '2024': [],
            'combined': []
        }
        
        # Check 2024 data (smaller, processed format)
        text_2024_path = self.textdata_dir / "2024" / "text_us_2024.parquet"
        if text_2024_path.exists():
            df_2024 = pd.read_parquet(text_2024_path)
            stocks_2024 = df_2024['gvkey'].unique().tolist()
            available_stocks['2024'] = stocks_2024
            logger.info(f"Found {len(stocks_2024)} stocks in 2024 data")
        
        # Check 2023 data (larger, full format) - sample only
        text_2023_path = self.textdata_dir / "2023" / "text_us_2023.parquet"
        if text_2023_path.exists():
            try:
                # Read first chunk to get sample stocks
                df_2023_sample = pd.read_parquet(text_2023_path, nrows=1000)
                stocks_2023_sample = df_2023_sample['gvkey'].dropna().unique()[:20].tolist()
                available_stocks['2023'] = stocks_2023_sample
                logger.info(f"Sampled {len(stocks_2023_sample)} stocks from 2023 data")
            except Exception as e:
                logger.warning(f"Could not sample 2023 data: {e}")
        
        # Combine unique stocks
        all_stocks = set(available_stocks['2024'] + available_stocks['2023'])
        available_stocks['combined'] = list(all_stocks)
        
        return available_stocks
    
    def extract_stock_texts(self, selected_stocks: List[str]) -> Dict[str, List[Dict]]:
        """
        Extract text data for selected stocks from both 2023 and 2024 data
        
        Args:
            selected_stocks: List of stock identifiers to extract
            
        Returns:
            Dictionary mapping stock_id to list of text entries
        """
        logger.info(f"Extracting text data for {len(selected_stocks)} stocks")
        stock_texts = {}
        
        # Process 2024 data
        text_2024_path = self.textdata_dir / "2024" / "text_us_2024.parquet"
        if text_2024_path.exists():
            df_2024 = pd.read_parquet(text_2024_path)
            logger.info(f"Processing 2024 data ({len(df_2024)} entries)")
            
            for stock in selected_stocks:
                stock_data = df_2024[df_2024['gvkey'] == stock]
                
                texts = []
                for _, row in stock_data.iterrows():
                    if pd.notna(row['text']) and str(row['text']).strip():
                        texts.append({
                            'text': str(row['text']),
                            'date': str(row['date']),
                            'filing_type': str(row['filing_type']),
                            'source': '2024_data',
                            'length': len(str(row['text']))
                        })
                
                if texts:
                    stock_texts[stock] = texts
                    logger.info(f"Stock {stock}: {len(texts)} texts from 2024")
        
        # Process 2023 data (carefully due to size)
        text_2023_path = self.textdata_dir / "2023" / "text_us_2023.parquet"
        if text_2023_path.exists():
            logger.info("Processing 2023 data...")
            try:
                df_2023 = pd.read_parquet(text_2023_path)
                
                # Look for numeric equivalents of stock identifiers
                for stock in selected_stocks:
                    if isinstance(stock, str) and stock.startswith(('TOP_', 'BOT_')):
                        # Create synthetic numeric mapping for demo
                        continue
                    
                    stock_2023_data = df_2023[df_2023['gvkey'] == stock]
                    
                    if len(stock_2023_data) > 0:
                        if stock not in stock_texts:
                            stock_texts[stock] = []
                        
                        count_added = 0
                        for _, row in stock_2023_data.iterrows():
                            # Process Risk Factors text
                            if pd.notna(row['rf']) and str(row['rf']).strip() and len(str(row['rf'])) > 50:
                                stock_texts[stock].append({
                                    'text': str(row['rf']),
                                    'date': str(row['date']),
                                    'filing_type': str(row['file_type']) + '_RF',
                                    'source': '2023_data',
                                    'length': len(str(row['rf']))
                                })
                                count_added += 1
                            
                            # Process Management Discussion text
                            if pd.notna(row['mgmt']) and str(row['mgmt']).strip() and len(str(row['mgmt'])) > 50:
                                stock_texts[stock].append({
                                    'text': str(row['mgmt']),
                                    'date': str(row['date']),
                                    'filing_type': str(row['file_type']) + '_MGMT',
                                    'source': '2023_data',
                                    'length': len(str(row['mgmt']))
                                })
                                count_added += 1
                        
                        if count_added > 0:
                            logger.info(f"Stock {stock}: +{count_added} texts from 2023")
                            
            except Exception as e:
                logger.warning(f"Error processing 2023 data: {e}")
        
        logger.info(f"Extraction complete: {len(stock_texts)} stocks with text data")
        return stock_texts


class FinBERTDataFormatter:
    """
    Formats extracted text data for FinBERT processing
    """
    
    def __init__(self, max_text_length: int = 2000, min_text_length: int = 50):
        self.max_text_length = max_text_length
        self.min_text_length = min_text_length
    
    def format_for_finbert(self, stock_texts: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Format stock texts into FinBERT-ready DataFrame
        
        Args:
            stock_texts: Dictionary from StockTextExtractor.extract_stock_texts()
            
        Returns:
            DataFrame ready for FinBERT inference
        """
        logger.info("Formatting data for FinBERT processing...")
        
        rows = []
        text_id_counter = 0
        
        for stock_id, texts in stock_texts.items():
            # Determine stock category based on naming convention
            if isinstance(stock_id, str):
                if stock_id.startswith('TOP_'):
                    stock_category = "top_performer"
                elif stock_id.startswith('BOT_'):
                    stock_category = "bottom_performer"
                else:
                    stock_category = "unknown"
            else:
                # For numeric stock IDs, we'll need actual performance data
                stock_category = "unknown"
            
            for text_entry in texts:
                text_id_counter += 1
                
                # Clean and validate text
                clean_text = str(text_entry['text']).strip()
                
                # Skip texts that are too short or too long
                if len(clean_text) < self.min_text_length:
                    continue
                    
                if len(clean_text) > self.max_text_length:
                    clean_text = clean_text[:self.max_text_length] + "..."
                
                rows.append({
                    'text_id': f"{stock_id}_{text_id_counter:03d}",
                    'stock_id': stock_id,
                    'stock_category': stock_category,
                    'text': clean_text,
                    'text_length': len(clean_text),
                    'date': text_entry['date'],
                    'filing_type': text_entry['filing_type'],
                    'source': text_entry['source']
                })
        
        df = pd.DataFrame(rows)
        
        if len(df) > 0:
            logger.info(f"Formatted {len(df)} text entries for FinBERT")
            logger.info(f"Average text length: {df['text_length'].mean():.0f} characters")
            
            # Log category distribution
            if 'stock_category' in df.columns:
                category_counts = df.groupby('stock_category').size()
                for category, count in category_counts.items():
                    logger.info(f"  {category}: {count} texts")
        
        return df


class LightningAIExporter:
    """
    Exports formatted data for Lightning.ai processing
    """
    
    def __init__(self, output_dir: str = "lightning_ai"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_for_lightning(self, finbert_df: pd.DataFrame) -> Dict[str, str]:
        """
        Export data in Lightning.ai compatible format
        
        Args:
            finbert_df: Formatted DataFrame from FinBERTDataFormatter
            
        Returns:
            Dictionary with exported file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main input file for FinBERT
        input_file = self.output_dir / f"finbert_input_{timestamp}.csv"
        finbert_df.to_csv(input_file, index=False)
        
        # Simple text-only format
        texts_file = self.output_dir / f"texts_only_{timestamp}.txt"
        with open(texts_file, 'w', encoding='utf-8') as f:
            for _, row in finbert_df.iterrows():
                f.write(f"{row['text_id']}\t{row['text']}\n")
        
        # Metadata
        metadata = {
            'creation_timestamp': timestamp,
            'total_texts': len(finbert_df),
            'total_stocks': finbert_df['stock_id'].nunique() if len(finbert_df) > 0 else 0,
            'average_text_length': float(finbert_df['text_length'].mean()) if len(finbert_df) > 0 else 0,
            'finbert_model': 'yiyanghkust/finbert-tone',
            'processing_notes': 'Data prepared for Lightning.ai FinBERT inference'
        }
        
        if len(finbert_df) > 0 and 'stock_category' in finbert_df.columns:
            metadata['stocks_by_category'] = finbert_df.groupby('stock_category')['stock_id'].nunique().to_dict()
        
        metadata_file = self.output_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Lightning.ai instructions
        instructions_file = self.output_dir / f"lightning_instructions_{timestamp}.md"
        self._create_instructions_file(instructions_file, input_file.name, len(finbert_df))
        
        # Results processor
        processor_file = self.output_dir / "process_results.py"
        self._create_results_processor(processor_file)
        
        logger.info(f"Exported {len(finbert_df)} texts to Lightning.ai format")
        logger.info(f"Files saved in: {self.output_dir}")
        
        return {
            'input_file': str(input_file),
            'texts_file': str(texts_file),
            'metadata_file': str(metadata_file),
            'instructions_file': str(instructions_file),
            'processor_file': str(processor_file)
        }
    
    def _create_instructions_file(self, file_path: Path, input_filename: str, text_count: int):
        """Create Lightning.ai processing instructions"""
        content = f'''# Lightning.ai FinBERT Processing Instructions

## Input Data
- **Main file**: `{input_filename}`
- **Total texts**: {text_count}
- **Model**: yiyanghkust/finbert-tone

## Setup Code for Lightning.ai

```python
# Install requirements
!pip install transformers torch pandas

# Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

# Load FinBERT model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
```

## Processing Code

```python
# Load your data
df = pd.read_csv('{input_filename}')
texts = df['text'].tolist()
text_ids = df['text_id'].tolist()

# Process in batches
batch_size = 16
results = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_ids = text_ids[i:i+batch_size]
    
    # Tokenize
    inputs = tokenizer(
        batch_texts, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='pt'
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Store results
    for j, pred in enumerate(predictions.cpu().numpy()):
        results.append({{
            'text_id': batch_ids[j],
            'negative': float(pred[0]),
            'neutral': float(pred[1]),
            'positive': float(pred[2])
        }})
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {{i+batch_size}}/{{len(texts)}} texts")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('finbert_sentiment_results.csv', index=False)
print(f"Results saved! Processed {{len(results)}} texts.")
```

## Expected Output
CSV file with sentiment scores for each text_id.

## Next Steps
1. Download the results file
2. Use the process_results.py script to analyze final rankings
'''
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_results_processor(self, file_path: Path):
        """Create results processing script"""
        content = '''"""
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
    
    print(f"\\n=== STOCK SENTIMENT RANKINGS ===")
    display_cols = ['stock_id', 'stock_category', 'net_sentiment', 
                   'weighted_sentiment', 'sentiment_rank', 'text_count']
    print(stock_sentiment[display_cols].to_string(index=False))
    
    print(f"\\nDetailed results saved to: {output_file}")
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
        
        print(f"\\nSUCCESS! Processed sentiment analysis for {len(rankings)} stocks")
        
        # Show top and bottom performers
        print(f"\\nTop 3 by sentiment:")
        print(rankings.head(3)[['stock_id', 'weighted_sentiment', 'sentiment_rank']].to_string(index=False))
        
        print(f"\\nBottom 3 by sentiment:")  
        print(rankings.tail(3)[['stock_id', 'weighted_sentiment', 'sentiment_rank']].to_string(index=False))
        
    else:
        print("ERROR: Could not find input files!")
        print("Make sure to:")
        print("1. Download sentiment results from Lightning.ai")
        print("2. Update file paths in this script")
        print("3. Run the script again")


if __name__ == "__main__":
    main()
'''
        
        with open(file_path, 'w') as f:
            f.write(content)


def main():
    """
    Main function to run the complete data preparation pipeline
    """
    print("=" * 60)
    print("FINBERT DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Initialize components
        extractor = StockTextExtractor()
        formatter = FinBERTDataFormatter()
        exporter = LightningAIExporter()
        
        # Step 2: Find available stocks
        available_stocks = extractor.get_available_stocks()
        
        # Step 3: Select test stocks (use available stocks from 2024 data)
        if available_stocks['2024']:
            selected_stocks = available_stocks['2024'][:10]  # Take first 10
            print(f"Selected {len(selected_stocks)} stocks for processing:")
            print(f"  {selected_stocks}")
        else:
            print("No stocks found in 2024 data!")
            return
        
        # Step 4: Extract text data
        stock_texts = extractor.extract_stock_texts(selected_stocks)
        
        if not stock_texts:
            print("No text data extracted! Check TextData folder.")
            return
        
        # Step 5: Format for FinBERT
        finbert_df = formatter.format_for_finbert(stock_texts)
        
        if len(finbert_df) == 0:
            print("No valid texts formatted for FinBERT!")
            return
        
        # Step 6: Export for Lightning.ai
        exported_files = exporter.export_for_lightning(finbert_df)
        
        # Step 7: Summary
        print(f"\\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\\nSummary:")
        print(f"  - Stocks processed: {len(stock_texts)}")
        print(f"  - Texts prepared: {len(finbert_df)}")
        print(f"  - Average text length: {finbert_df['text_length'].mean():.0f} chars")
        
        print(f"\\nFiles created:")
        for file_type, path in exported_files.items():
            print(f"  - {file_type}: {Path(path).name}")
        
        print(f"\\nNext steps:")
        print(f"  1. Upload files from 'lightning_ai' folder to Lightning.ai")
        print(f"  2. Follow instructions in the markdown file")
        print(f"  3. Run FinBERT inference")
        print(f"  4. Process results with process_results.py")
        
        return True
        
    except Exception as e:
        print(f"\\nERROR: Pipeline failed - {e}")
        logger.error(f"Pipeline error: {e}")
        return False


if __name__ == "__main__":
    main()