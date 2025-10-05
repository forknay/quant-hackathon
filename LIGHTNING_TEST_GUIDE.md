# Lightning.ai FinBERT Testing Guide

## Quick Start: Test with 3 Real Stocks

### Step 1: Generate FinBERT Input File

```python
# Run this in Lightning.ai Studio or your local environment
from sentiment_analysis.lightning_ai.process_sentiment_rankings import process_sentiment_rankings

# Generate FinBERT input for 3 test stocks with real SEC filing data
process_sentiment_rankings(use_real_data=True)
```

This will:
- Extract real SEC filings for stocks 1004:01, 1045:01, 1050:01
- Generate `finbert_input_real_stocks_[timestamp].csv`
- Ready for FinBERT processing

### Step 2: Upload and Process on Lightning.ai

1. **Upload the generated CSV** to Lightning.ai Studio
2. **Run FinBERT processing**:

```python
# On Lightning.ai Studio
import pandas as pd
from transformers import pipeline

# Load the input file
df = pd.read_csv('finbert_input_real_stocks_[timestamp].csv')

# Initialize FinBERT
finbert = pipeline("text-classification", 
                  model="ProsusAI/finbert", 
                  device=0)  # Use GPU

# Process texts
results = []
for idx, row in df.iterrows():
    result = finbert(row['text'])
    results.append({
        'text_id': row['text_id'],
        'stock_identifier': row['stock_identifier'], 
        'gvkey': row['gvkey'],
        'iid': row['iid'],
        'confidence_score': result[0]['score'],
        'label': result[0]['label']
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('lightning_finbert_results.csv', index=False)
```

### Step 3: Apply Enhanced Normalization

```python
# Back in your local environment or on Lightning.ai
from sentiment_analysis.lightning_ai.process_sentiment_rankings import normalize_sentiment_scores

# Load FinBERT results
finbert_results = pd.read_csv('lightning_finbert_results.csv')

# Apply normalization (choose method)
normalized_minmax = normalize_sentiment_scores(finbert_results, method='minmax')
normalized_softmax = normalize_sentiment_scores(finbert_results, method='softmax')
normalized_linear = normalize_sentiment_scores(finbert_results, method='linear')

# Verify normalization (should all equal 1.000000)
print(f"Min-Max sum: {normalized_minmax['normalized_score'].sum():.6f}")
print(f"Softmax sum: {normalized_softmax['normalized_score'].sum():.6f}")  
print(f"Linear sum: {normalized_linear['normalized_score'].sum():.6f}")

# Get final sentiment rankings
print("\nTop sentiment stocks:")
print(normalized_minmax.nlargest(3, 'normalized_score')[['stock_identifier', 'normalized_score']])
```

## Custom Stock Testing

```python
# Test with your own stocks
my_stocks = ['1004:01', '1013:01', '1019:01', '1045:01']  # Your algorithm output

from sentiment_analysis.lightning_ai.process_sentiment_rankings import extract_real_textdata_for_stocks

# Extract SEC filings for your stocks
real_texts = extract_real_textdata_for_stocks(stock_identifiers=my_stocks)
print(f"Found {len(real_texts)} SEC filing texts for your stocks")

# Generate FinBERT input
process_sentiment_rankings(use_real_data=True)  # Uses default stocks, modify function to accept custom stocks
```

## Expected Results

- **Input**: 3 stocks → ~36 SEC filing text segments
- **FinBERT**: Sentiment scores for each text segment  
- **Normalization**: Scores sum exactly to 1.000000
- **Output**: Stock sentiment rankings ready for trading decisions

## Troubleshooting

1. **No SEC filings found**: Check if your stocks have data in TextData/2023-2025/
2. **Lightning.ai GPU memory**: Process in smaller batches if texts are very long
3. **Normalization not summing to 1.0**: Check for NaN values in FinBERT results

## Files Ready for Testing

✅ `sentiment_analysis/lightning_ai/process_sentiment_rankings.py` - Main processor  
✅ `sentiment_analysis/data_preparation/prepare_data.py` - Data extraction  
✅ `TextData/` - 226K+ SEC filings database  
✅ Complete workflow documentation in README.md

**Ready to test on Lightning.ai with real SEC filing data!** 🚀