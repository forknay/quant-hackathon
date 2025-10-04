# Real Stock Sentiment Analysis - Lightning.ai Processing

**Generated:** 2025-10-04 18:35:43
**Input Stocks:** 1004:01, 1045:01, 1050:01, 1075:01, 1076:01
**Total Texts:** 34 real SEC filing texts

## Processing Steps

### 1. Upload Files to Lightning.ai Studio
- `finbert_real_stocks_20251004_183543.csv` (main input data)
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
df = pd.read_csv('finbert_real_stocks_20251004_183543.csv')
print(f"Processing {len(df)} texts from real stocks: 1004:01, 1045:01, 1050:01, 1075:01, 1076:01")

# Process each text through FinBERT
results = []
for idx, row in df.iterrows():
    try:
        result = sentiment_pipeline(row['text'])[0]
        
        results.append({
            'text_id': row['text_id'],
            'gvkey': row['gvkey'],
            'iid': row['iid'], 
            'stock_identifier': row['stock_identifier'],
            'text_type': row['text_type'],
            'sentiment_label': result['label'],
            'sentiment_score': result['score'],
            'text_length': row['text_length'],
            'year': row['year']
        })
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(df)} texts")
            
    except Exception as e:
        print(f"Error processing {row['text_id']}: {e}")
        results.append({
            'text_id': row['text_id'],
            'gvkey': row['gvkey'],
            'iid': row['iid'],
            'stock_identifier': row['stock_identifier'], 
            'text_type': row['text_type'],
            'sentiment_label': 'neutral',
            'sentiment_score': 0.5,
            'text_length': row['text_length'],
            'year': row['year']
        })

# Save FinBERT results
results_df = pd.DataFrame(results)
results_df.to_csv('finbert_results_real_stocks_20251004_183543.csv', index=False)
print(f"\n✅ FinBERT processing complete! Results saved.")
print(f"📊 Processed {len(results_df)} texts from {results_df['stock_identifier'].nunique()} stocks")
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
