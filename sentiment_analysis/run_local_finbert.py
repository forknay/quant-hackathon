"""
Local FinBERT Processing Script

Run FinBERT sentiment analysis locally on your machine.
Note: This requires a GPU for reasonable performance on large texts.
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import os
from pathlib import Path
import json
from datetime import datetime

def run_local_finbert():
    """Run FinBERT processing locally"""
    print("🚀 LOCAL FINBERT SENTIMENT ANALYSIS")
    print("=" * 50)
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected - processing will be slow")
        proceed = input("Continue with CPU processing? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            return
    
    # Find the latest input file
    lightning_ai_path = Path("lightning_ai")
    if not lightning_ai_path.exists():
        print(f"❌ Lightning AI folder not found: {lightning_ai_path}")
        return
    
    input_files = list(lightning_ai_path.glob("finbert_real_stocks_*.csv"))
    if not input_files:
        print("❌ No FinBERT input files found")
        return
    
    latest_file = sorted(input_files)[-1]
    print(f"📁 Using input file: {latest_file.name}")
    
    # Load data
    try:
        df = pd.read_csv(latest_file)
        print(f"📊 Loaded {len(df)} texts from {df['stock_identifier'].nunique()} stocks")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Load FinBERT model
    print("\n🤖 Loading FinBERT model...")
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="yiyanghkust/finbert-tone",
            device=device,
            max_length=512,
            truncation=True,
            padding=True
        )
        print("✅ FinBERT model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading FinBERT: {e}")
        return
    
    # Process texts
    print(f"\n📝 Processing {len(df)} texts through FinBERT...")
    results = []
    
    for idx, row in df.iterrows():
        try:
            # Process through FinBERT
            text_input = str(row['text'])[:512]  # Truncate for FinBERT
            result = sentiment_pipeline(text_input)[0]
            
            # Store result
            results.append({
                'text_id': row['text_id'],
                'gvkey': row['gvkey'],
                'iid': row['iid'],
                'stock_identifier': row['stock_identifier'],
                'text_type': row['text_type'],
                'sentiment_label': result['label'],
                'sentiment_score': float(result['score']),
                'text_length': row['text_length'],
                'year': row['year'],
                'processing_date': datetime.now().isoformat()
            })
            
            # Progress update
            if (idx + 1) % 5 == 0 or (idx + 1) == len(df):
                print(f"  Progress: {idx + 1}/{len(df)} texts ({((idx + 1)/len(df)*100):.1f}%)")
                
        except Exception as e:
            print(f"  ⚠️  Error processing text {row['text_id']}: {e}")
            # Add neutral result for failed texts
            results.append({
                'text_id': row['text_id'],
                'gvkey': row['gvkey'],
                'iid': row['iid'],
                'stock_identifier': row['stock_identifier'],
                'text_type': row['text_type'],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'text_length': row['text_length'],
                'year': row['year'],
                'processing_date': datetime.now().isoformat()
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = lightning_ai_path / f"finbert_results_local_{timestamp}.csv"
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✅ FINBERT PROCESSING COMPLETE!")
    print(f"📊 Results Summary:")
    print(f"  - Total texts processed: {len(results_df)}")
    print(f"  - Unique stocks: {results_df['stock_identifier'].nunique()}")
    print(f"  - Sentiment distribution:")
    
    sentiment_counts = results_df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"    {sentiment}: {count}")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Show sample results
    print(f"\n📋 Sample Results:")
    sample_cols = ['stock_identifier', 'text_type', 'sentiment_label', 'sentiment_score']
    print(results_df[sample_cols].head(10).to_string(index=False))
    
    print(f"\n🎯 Next: Use process_sentiment_rankings.py to generate final rankings")
    
    return str(output_file)

if __name__ == "__main__":
    run_local_finbert()