"""
LIGHTNING.AI TESTING GUIDE
==========================

Complete step-by-step instructions for testing the sentiment analysis pipeline on Lightning.ai Studio.

PREREQUISITES:
- Lightning.ai Studio account
- Upload complete_sentiment_pipeline.py to Lightning.ai
- Upload TextData directory (or use existing finbert_input_*.csv files)

TESTING STEPS:
"""

# =============================================================================
# STEP 1: UPLOAD FILES TO LIGHTNING.AI
# =============================================================================

"""
1. Go to Lightning.ai Studio
2. Create new Studio or open existing one
3. Upload these files:
   - complete_sentiment_pipeline.py (the main script)
   - TextData/ directory (if you have it)
   OR
   - Any finbert_input_*.csv files you want to test

File structure should look like:
/
├── complete_sentiment_pipeline.py
├── TextData/
│   ├── 2023/
│   ├── 2024/
│   └── 2025/
OR
├── finbert_input_real_stocks_20251004_230807.csv
"""

# =============================================================================
# STEP 2: INSTALL DEPENDENCIES
# =============================================================================

"""
In Lightning.ai Studio terminal, run:
"""

# !pip install transformers torch pandas numpy

"""
This installs:
- transformers (for FinBERT)
- torch (PyTorch backend) 
- pandas (data processing)
- numpy (numerical operations)
"""

# =============================================================================
# STEP 3: TEST METHODS
# =============================================================================

print("="*60)
print("LIGHTNING.AI TESTING METHODS")
print("="*60)

# -----------------------------------------------------------------------------
# METHOD 1: Command Line Testing
# -----------------------------------------------------------------------------

print("\n🚀 METHOD 1: COMMAND LINE TESTING")
print("-" * 40)

print("""
In Lightning.ai terminal, run:

# Test with specific stocks
python complete_sentiment_pipeline.py 1004:01 1045:01 1050:01

# Test with auto-detection (uses default stocks or finds CSV files)  
python complete_sentiment_pipeline.py

This will:
1. Extract SEC filing texts for the specified stocks
2. Run FinBERT sentiment analysis
3. Generate 3 normalized ranking files
4. Show results in terminal
""")

# -----------------------------------------------------------------------------
# METHOD 2: Jupyter Notebook Testing
# -----------------------------------------------------------------------------

print("\n📓 METHOD 2: JUPYTER NOTEBOOK TESTING")
print("-" * 40)

print("""
Create a new Jupyter notebook in Lightning.ai and run:
""")

notebook_code = '''
# Cell 1: Import and test
import sys
from complete_sentiment_pipeline import get_sentiment_rankings, process_stocks_sentiment

# Cell 2: Test algorithm integration
algorithm_stocks = ['1004:01', '1045:01', '1050:01'] 
rankings = get_sentiment_rankings(algorithm_stocks, normalization_method='softmax')
print("Top sentiment stocks:")
print(rankings.head())

# Cell 3: Full pipeline test
results = process_stocks_sentiment(stock_identifiers=['1004:01', '1045:01', '1050:01'])
print(f"Processed {len(results['raw_results'])} texts")
print("All normalization methods sum to 1.0:")
print(f"MinMax sum: {results['minmax_rankings']['normalized_score'].sum():.6f}")
print(f"Softmax sum: {results['softmax_rankings']['normalized_score'].sum():.6f}")
print(f"Linear sum: {results['linear_rankings']['normalized_score'].sum():.6f}")
'''

print(notebook_code)

# -----------------------------------------------------------------------------
# METHOD 3: Direct Function Testing
# -----------------------------------------------------------------------------

print("\n⚡ METHOD 3: DIRECT FUNCTION TESTING")
print("-" * 40)

print("""
In Lightning.ai Python console or notebook:
""")

direct_code = '''
# Quick function test
from complete_sentiment_pipeline import validate_stock_input

# Test stock validation
test_stocks = ['1004', '1045:01', '1050']
validated = validate_stock_input(test_stocks)
print(f"Validated: {validated}")

# Test with your algorithm's stock list
my_algorithm_stocks = ['1004:01', '2034:01', '3567:01']
from complete_sentiment_pipeline import get_sentiment_rankings
rankings = get_sentiment_rankings(my_algorithm_stocks)
print("Rankings ready for algorithm:")
print(rankings[['stock_identifier', 'normalized_score']])
'''

print(direct_code)

# =============================================================================
# STEP 4: EXPECTED OUTPUT
# =============================================================================

print("\n📊 EXPECTED OUTPUT ON LIGHTNING.AI")
print("=" * 40)

expected_output = '''
[PIPELINE] COMPLETE SENTIMENT ANALYSIS PIPELINE
==================================================
[START] Started at: 2025-10-04 15:30:45
[STOCKS] Processing stocks: ['1004:01', '1045:01', '1050:01']
[EXTRACT] Extracting SEC filing texts for 3 stocks...
[PATH] Using TextData path: TextData
[FOUND] Found 12 texts in 2023/batch_1.parquet
[SUCCESS] Extracted 36 total texts
[FINBERT] Running FinBERT analysis on 36 texts...
[GPU] Initializing FinBERT model (GPU)...
[OK] GPU initialization successful
Processing 1/36 (2.8%)...
Processing 11/36 (30.6%)...
Processing 21/36 (58.3%)...
Processing 31/36 (86.1%)...
[SUCCESS] Successfully processed 36 texts
[STATS] Average confidence: 0.847
[RANKINGS] Generating sentiment rankings with 3 normalization methods...
[NORMALIZE] Applying minmax normalization...
[OK] Normalized scores sum: 1.000000
[NORMALIZE] Applying softmax normalization...
[OK] Normalized scores sum: 1.000000
[NORMALIZE] Applying linear normalization...
[OK] Normalized scores sum: 1.000000

[COMPLETE] PIPELINE COMPLETE!
[PROCESSED] Processed 36 texts from 3 stocks
[SAVED] Results saved with timestamp: 20251004_153147

[RESULTS] TOP SENTIMENT STOCKS (All Methods):

Min-Max Normalization:
  stock_identifier  confidence_score  normalized_score
0         1045:01          0.892341          0.421053
1         1004:01          0.834219          0.368421  
2         1050:01          0.801234          0.210526
Sum: 1.000000

Files created:
- finbert_results_20251004_153147.csv
- sentiment_rankings_minmax_20251004_153147.csv
- sentiment_rankings_softmax_20251004_153147.csv
- sentiment_rankings_linear_20251004_153147.csv
- sentiment_summary_20251004_153147.json
'''

print(expected_output)

# =============================================================================
# STEP 5: TROUBLESHOOTING
# =============================================================================

print("\n🔧 TROUBLESHOOTING ON LIGHTNING.AI")
print("=" * 40)

troubleshooting = '''
COMMON ISSUES & SOLUTIONS:

1. "TextData directory not found"
   Solution: Either upload TextData/ directory OR use existing CSV files
   
2. "No module named 'transformers'"
   Solution: Run: !pip install transformers torch pandas numpy
   
3. GPU out of memory
   Solution: Script automatically falls back to CPU
   
4. "No texts found for stocks"
   Solution: Check if your stock IDs exist in TextData or use test CSV
   
5. Unicode encoding errors
   Solution: Fixed in latest version - should work on Lightning.ai
   
6. Import errors
   Solution: Make sure complete_sentiment_pipeline.py is in root directory

VERIFICATION COMMANDS:
- Check files: !ls -la
- Check Python: !python --version  
- Check GPU: !nvidia-smi
- Test import: python -c "from complete_sentiment_pipeline import validate_stock_input; print('OK')"
'''

print(troubleshooting)

# =============================================================================
# STEP 6: INTEGRATION WITH YOUR ALGORITHM
# =============================================================================

print("\n🤖 ALGORITHM INTEGRATION ON LIGHTNING.AI")
print("=" * 40)

integration_example = '''
# Example: Complete trading algorithm integration on Lightning.ai

def my_trading_algorithm():
    """Your main trading algorithm"""
    
    # Step 1: Your algorithm generates stock candidates
    candidate_stocks = generate_stock_candidates()  # Your function
    print(f"Algorithm found {len(candidate_stocks)} candidate stocks")
    
    # Step 2: Get sentiment rankings
    from complete_sentiment_pipeline import get_sentiment_rankings
    sentiment_rankings = get_sentiment_rankings(
        candidate_stocks, 
        normalization_method='softmax'  # or 'minmax' or 'linear'
    )
    
    # Step 3: Use sentiment weights in your algorithm
    for _, row in sentiment_rankings.iterrows():
        stock = row['stock_identifier']
        sentiment_weight = row['normalized_score']
        confidence = row['confidence_score']
        
        # Integrate with your trading logic
        final_weight = calculate_position_size(stock, sentiment_weight, confidence)
        execute_trade(stock, final_weight)
        
        print(f"Stock {stock}: Sentiment={sentiment_weight:.4f}, Position={final_weight:.4f}")
    
    return sentiment_rankings

# Run on Lightning.ai
results = my_trading_algorithm()
'''

print(integration_example)

print("\n" + "="*60)
print("🎯 READY TO TEST ON LIGHTNING.AI!")
print("Upload complete_sentiment_pipeline.py and run any of the methods above.")
print("="*60)