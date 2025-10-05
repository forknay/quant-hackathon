"""
LIGHTNING.AI UPLOAD GUIDE - REAL TEXTDATA EXTRACTION
=====================================================

📋 EXACTLY WHAT TO UPLOAD TO LIGHTNING.AI FOR ANY STOCK PROCESSING

🎯 GOAL: Make the sentiment pipeline work with ANY GVKEY:IID by uploading real TextData

📁 REQUIRED FILES TO UPLOAD:
============================

1. **complete_sentiment_pipeline.py** ← Main script (REQUIRED)
   - The fixed all-in-one pipeline
   - Always extracts from real TextData
   - No pre-generated datasets

2. **TextData/ directory** ← Real SEC filing data (REQUIRED)
   Upload the entire TextData directory structure:
   
   TextData/
   ├── 2023/
   │   ├── batch_1.parquet
   │   ├── batch_2.parquet
   │   └── ... (all parquet files)
   ├── 2024/
   │   ├── batch_1.parquet
   │   └── ... (all parquet files)
   └── 2025/
       └── ... (all parquet files)

3. **lightning_quick_test.py** ← Test script (OPTIONAL)
   - Quick validation before running full pipeline

🚫 DO NOT UPLOAD:
================
- finbert_input_*.csv files (pre-generated, limited stocks)
- lightning_complete_finbert.py (old version with fallbacks)
- Any other CSV files with predetermined data

⚡ LIGHTNING.AI SETUP STEPS:
============================

STEP 1: Upload Files
-------------------
1. Create new Lightning.ai Studio
2. Upload complete_sentiment_pipeline.py
3. Upload entire TextData/ directory (this may take time due to size)
4. Optional: Upload lightning_quick_test.py

STEP 2: Install Dependencies
---------------------------
In Lightning.ai terminal:
```bash
pip install transformers torch pandas numpy pyarrow
```

STEP 3: Test with Any Stocks
---------------------------
```bash
# Test with your specific stocks
python complete_sentiment_pipeline.py 1234:01 5678:01 9999:01

# Test with any GVKEY you want
python complete_sentiment_pipeline.py 2045:01 7890:01

# Quick validation first
python lightning_quick_test.py
```

STEP 4: Algorithm Integration
----------------------------
```python
# In your Lightning.ai notebook/script
from complete_sentiment_pipeline import get_sentiment_rankings

# ANY stocks your algorithm provides
my_algorithm_stocks = ['1234:01', '5678:01', '9999:01']

# Get sentiment rankings for ANY stocks
rankings = get_sentiment_rankings(my_algorithm_stocks, normalization_method='softmax')

# Use the rankings (guaranteed sum=1.0)
for _, row in rankings.iterrows():
    stock = row['stock_identifier']
    weight = row['normalized_score']
    print(f"Stock {stock}: Weight = {weight:.4f}")
```

📊 TEXTDATA REQUIREMENTS:
=========================

File Structure Expected:
- TextData/YYYY/*.parquet files
- Each parquet must have columns:
  * gvkey (or GVKEY) - the stock identifier
  * text (or content, filing_text) - the SEC filing text

Supported Column Names:
- GVKEY: gvkey, GVKEY, gvkey_id, stock_id
- TEXT: text, content, filing_text, TEXT, CONTENT

💾 TEXTDATA SIZE CONSIDERATIONS:
===============================

Full TextData is ~226GB with 226K+ filings
- 2023: ~137K filings 
- 2024: ~87K filings
- 2025: ~2.6K filings

Options for Lightning.ai:
1. **Full Upload**: Upload entire TextData (best coverage, any stock)
2. **Year Subset**: Upload only recent years (2024-2025)
3. **Stock Subset**: Pre-filter TextData for specific stocks locally, then upload

🎯 VERIFICATION ON LIGHTNING.AI:
===============================

After upload, verify the setup:
```bash
# Check TextData structure
ls -la TextData/
ls -la TextData/2023/

# Quick test with known stocks
python complete_sentiment_pipeline.py 1004:01 1045:01

# Test with your algorithm's stocks
python complete_sentiment_pipeline.py YOUR_STOCK1:01 YOUR_STOCK2:01
```

Expected Output:
```
[PIPELINE] COMPLETE SENTIMENT ANALYSIS PIPELINE
[EXTRACT] Extracting SEC filing texts for 2 stocks...
[PATH] Using TextData path: TextData
[SCAN] Scanning 12 parquet files in 2023...
[FOUND] Found 15 texts in batch_1.parquet
[RESULTS] Found data for 2 out of 2 requested stocks
[EXTRACTED] Successfully extracted 30 texts from real TextData
[FINBERT] Running FinBERT analysis on 30 texts...
[GPU] Initializing FinBERT model (GPU)...
[COMPLETE] PIPELINE COMPLETE!
```

🚀 READY FOR ANY STOCK:
=======================

Once TextData is uploaded to Lightning.ai, you can process ANY stock that exists in your SEC filings database. No more predetermined datasets or limitations!

The pipeline will:
✅ Search through ALL parquet files
✅ Find any GVKEY you specify
✅ Extract real SEC filing texts
✅ Generate sentiment rankings
✅ Work with your algorithm seamlessly
"""