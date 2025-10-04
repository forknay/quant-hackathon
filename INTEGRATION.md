# ML Pipeline Integration Report
**Portfolio Management: Algorithmic Selection → Transformer Prediction**

---

## Executive Summary

This document outlines the integration strategy between two quantitative finance systems:

1. **Algorithmic Stock Selection Pipeline** (This Project) - Sector-based candidate selection using technical indicators
2. **Transformer ML Pipeline** (Your Project) - Deep learning model for stock price prediction

### **Key Finding: Simplified Integration Approach** ✅

Instead of building complex OHLCV data infrastructure in this project, we can leverage a **ticker-based input approach** where:

- ✅ **This project** selects candidate stocks using technical analysis
- ✅ **Your ML pipeline** receives ticker list and handles OHLCV fetching internally
- ✅ **Minimal integration code** required (single script)
- ✅ **Clean separation of concerns** - each system does what it's best at

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ALGORITHMIC PIPELINE (This Project)              │
│                                                                      │
│  1. Sector Filtering (GICS-based)                                  │
│     └─> Healthcare: 165,970 candidates over 20 years               │
│                                                                      │
│  2. Technical Indicators                                            │
│     ├─> Moving Averages (MA)                                        │
│     ├─> Momentum (MOM)                                              │
│     └─> GARCH Volatility                                            │
│                                                                      │
│  3. Candidate Selection                                             │
│     ├─> Composite Scoring: 50% MOM + 30% MA + 20% VOL             │
│     └─> Output: Top 15% Long + Bottom 15% Short                    │
│                                                                      │
│  OUTPUT: List of Stock Tickers [AAPL, MSFT, JNJ, ...]             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Ticker List
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE (Your Project)                        │
│                                                                      │
│  1. OHLCV Data Fetching (Internal)                                 │
│     └─> Fetch Open, High, Low, Close, Volume for tickers           │
│                                                                      │
│  2. Feature Engineering (25 features)                               │
│     ├─> 5/10/20/30-day Moving Averages for OHLCV                   │
│     └─> Normalize by historical maximum                             │
│                                                                      │
│  3. Transformer Model Inference                                     │
│     ├─> Input: [num_stocks, 32 days, 25 features]                  │
│     └─> Output: Predicted prices                                    │
│                                                                      │
│  4. Top-N Selection                                                 │
│     └─> Select top 10 stocks based on predictions                   │
│                                                                      │
│  OUTPUT: Portfolio {stocks, weights, scores}                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Integration Requirements

### **From Algorithmic Pipeline (This Project)**

**What we provide:**
- ✅ **Stock Tickers**: List of selected candidate stocks
- ✅ **Selection Metadata**: Date, sector, composite scores
- ✅ **Data Location**: Parquet files in `algo/results/{sector}_parquet/candidates/`

**Example Output:**
```python
# From: algo/results/healthcare_parquet/candidates/year=2024/month=6/
{
    'tickers': ['AAPL', 'MSFT', 'JNJ', 'PFE', 'UNH', ...],  # 50-100 candidates
    'selection_date': '2024-06-30',
    'sector': 'healthcare',
    'composite_scores': [0.89, 0.76, 0.65, ...]  # Our algo scores
}
```

### **From ML Pipeline (Your Project)**

**What we need from you:**

1. **✅ Inference Function** that accepts ticker list:
   ```python
   def inference_pipeline(
       tickers: List[str],           # Stock ticker symbols
       prediction_date: str,          # Date for prediction
       model: torch.nn.Module,        # Your trained model
       sequence_length: int = 32      # Lookback window
   ) -> Dict:
       """
       Your ML pipeline should:
       1. Fetch OHLCV data for these tickers (internally)
       2. Compute 25 features
       3. Run Transformer model
       4. Return top-N predictions
       """
       pass
   ```

2. **✅ Model Checkpoint Access**: 
   - Path to trained model: `model_tt_50.ckpt`
   - Model configuration parameters

3. **✅ Expected Output Format**:
   ```python
   {
       'selected_stocks': ['AAPL', 'JNJ', ...],
       'predicted_scores': [0.9456, 0.8234, ...],
       'portfolio_weights': {'AAPL': 0.534, 'JNJ': 0.466},
       'metadata': {
           'total_candidates': 50,
           'valid_stocks': 48,
           'excluded_stocks': ['XYZ'],
           'model_version': 'transformer_v1.0'
       }
   }
   ```

---

## Implementation Guide

### **Step 1: Create Integration Module**

Create a new folder in this project:

```bash
mkdir -p ml_integration
touch ml_integration/__init__.py
touch ml_integration/config.py
touch ml_integration/run_inference.py
```

### **Step 2: Configuration File**

**File:** `ml_integration/config.py`

```python
"""
Configuration for ML Pipeline Integration

This file contains paths and parameters to connect with the ML pipeline.
"""

from pathlib import Path

# ============================================================================
# ML PIPELINE PROJECT CONFIGURATION
# ============================================================================

# Path to the ML pipeline project directory (UPDATE THIS!)
ML_PIPELINE_PATH = Path("/path/to/ml-pipeline-project")

# Path to trained model checkpoint
MODEL_CHECKPOINT = ML_PIPELINE_PATH / "checkpoints/model_tt_50.ckpt"

# Verify paths exist
if not ML_PIPELINE_PATH.exists():
    raise FileNotFoundError(
        f"ML pipeline path not found: {ML_PIPELINE_PATH}\n"
        "Please update ML_PIPELINE_PATH in ml_integration/config.py"
    )

if not MODEL_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_CHECKPOINT}\n"
        "Please ensure the trained model is available"
    )

# ============================================================================
# ML MODEL PARAMETERS
# ============================================================================

# Model architecture parameters (from your specification)
ML_MODEL_CONFIG = {
    'input_size': 25,              # 25 OHLCV features
    'num_class': 1,                # Single output (price prediction)
    'hidden_size': 128,            # Hidden dimension size
    'num_feat_att_layers': 1,      # Feature attention layers
    'num_pre_att_layers': 1,       # Prediction attention layers
    'num_heads': 4,                # Multi-head attention heads
    'days': 32,                    # Sequence length (lookback window)
    'dropout': 0.1                 # Dropout rate
}

# ============================================================================
# INTEGRATION PARAMETERS
# ============================================================================

# Portfolio construction
TOP_N_STOCKS = 10                  # Number of top stocks to select
MIN_VALID_STOCKS = 5               # Minimum valid stocks required

# Performance
USE_GPU = True                     # Use CUDA if available
BATCH_SIZE = 32                    # Batch size for inference

# Data
SEQUENCE_LENGTH = 32               # Days of historical data required
MIN_DATA_COVERAGE = 0.9            # Minimum data availability (90%)

# ============================================================================
# SECTOR MAPPING
# ============================================================================

# Map sector names to GICS codes (for reference)
SECTOR_MAPPING = {
    'healthcare': '35',
    'it': '45',
    'financials': '40',
    'energy': '10',
    'materials': '15',
    'industrials': '20',
    'cons_discretionary': '25',
    'cons_staples': '30',
    'telecoms': '50',
    'utilities': '55',
    're': '60'
}
```

### **Step 3: Integration Script**

**File:** `ml_integration/run_inference.py`

```python
"""
ML Pipeline Integration Script

This script bridges the algorithmic candidate selection pipeline with the
Transformer ML prediction model.

Usage:
    python -m ml_integration.run_inference --sector healthcare --year 2024 --month 6
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm

# Add ML pipeline to path
from ml_integration.config import (
    ML_PIPELINE_PATH, 
    MODEL_CHECKPOINT, 
    ML_MODEL_CONFIG,
    TOP_N_STOCKS,
    USE_GPU
)

# Import from ML pipeline project (YOUR CODE)
sys.path.insert(0, str(ML_PIPELINE_PATH))
try:
    from model import TransformerStockPrediction  # Your model class
    from inference import inference_pipeline       # Your inference function
    ML_IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import ML pipeline modules: {e}")
    print(f"   Make sure ML_PIPELINE_PATH is correct: {ML_PIPELINE_PATH}")
    ML_IMPORTS_SUCCESS = False

# Import from algorithmic pipeline (THIS PROJECT)
from algo.core.config import RESULTS_DIR, SECTOR_GICS_MAPPING


def load_candidate_tickers(
    sector: str, 
    year: int, 
    month: int,
    candidate_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Load selected candidate tickers from algorithmic pipeline output
    
    Args:
        sector: Sector name (e.g., 'healthcare')
        year: Year
        month: Month
        candidate_type: Filter by 'long', 'short', or None for both
        
    Returns:
        DataFrame with candidate information
    """
    # Construct path to candidates parquet
    results_dir = RESULTS_DIR.replace('utilities', sector)
    candidates_path = Path(results_dir) / "candidates" / f"year={year}" / f"month={month}"
    
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"No candidates found at: {candidates_path}\n"
            f"Please run algorithmic pipeline first:\n"
            f"  python algo/run_sector.py --sector {sector}"
        )
    
    # Load parquet data
    df = pd.read_parquet(candidates_path)
    
    # Filter by candidate type if specified
    if candidate_type:
        df = df[df['candidate_type'] == candidate_type]
    
    print(f"✓ Loaded {len(df)} {candidate_type or 'total'} candidates for {sector} {year}-{month:02d}")
    return df


def prepare_ml_input(candidates_df: pd.DataFrame, prediction_date: str) -> Dict:
    """
    Prepare input data structure for ML pipeline
    
    Args:
        candidates_df: DataFrame with selected candidates
        prediction_date: Date for prediction (YYYY-MM-DD)
        
    Returns:
        Dictionary with ticker list and metadata
    """
    # Extract unique tickers (handle potential duplicates)
    if 'tic' in candidates_df.columns:
        tickers = candidates_df['tic'].dropna().unique().tolist()
    else:
        raise ValueError("Ticker column 'tic' not found in candidates DataFrame")
    
    # Build market data structure for ML pipeline
    market_data = {
        'tickers': tickers,
        'prediction_date': prediction_date,
        'sequence_length': ML_MODEL_CONFIG['days'],
        'num_candidates': len(tickers),
        'metadata': {
            'selection_scores': candidates_df['composite_score'].tolist() if 'composite_score' in candidates_df.columns else [],
            'candidate_types': candidates_df['candidate_type'].tolist() if 'candidate_type' in candidates_df.columns else [],
        }
    }
    
    return market_data


def load_ml_model(checkpoint_path: Path, use_gpu: bool = True) -> torch.nn.Module:
    """
    Load pre-trained Transformer model
    
    Args:
        checkpoint_path: Path to model checkpoint
        use_gpu: Whether to use GPU if available
        
    Returns:
        Loaded model in eval mode
    """
    if not ML_IMPORTS_SUCCESS:
        raise ImportError("ML pipeline modules not available. Check ML_PIPELINE_PATH configuration.")
    
    # Initialize model with config
    model = TransformerStockPrediction(**ML_MODEL_CONFIG)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Move to GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model loaded on GPU")
    else:
        print("✓ Model loaded on CPU")
    
    return model


def run_ml_prediction(
    sector: str,
    year: int,
    month: int,
    top_n: int = TOP_N_STOCKS,
    candidate_type: str = 'long',
    save_results: bool = True
) -> Dict:
    """
    Run ML prediction on algorithm-selected candidates
    
    Args:
        sector: Sector name
        year: Year
        month: Month
        top_n: Number of top stocks to select
        candidate_type: 'long', 'short', or None for both
        save_results: Whether to save results to JSON
        
    Returns:
        Portfolio selection with ML predictions
    """
    print("="*80)
    print(f"ML PREDICTION PIPELINE - {sector.upper()} {year}-{month:02d}")
    print("="*80)
    
    # Step 1: Load algorithmic candidates
    print(f"\n[1/4] Loading {candidate_type} candidates from algorithmic pipeline...")
    candidates_df = load_candidate_tickers(sector, year, month, candidate_type)
    
    if len(candidates_df) == 0:
        print(f"❌ No candidates found!")
        return {'error': 'No candidates available'}
    
    # Step 2: Prepare ML input
    print(f"\n[2/4] Preparing ML pipeline input...")
    prediction_date = f"{year}-{month:02d}-01"
    market_data = prepare_ml_input(candidates_df, prediction_date)
    
    print(f"  ✓ {len(market_data['tickers'])} tickers prepared")
    print(f"  ✓ Prediction date: {prediction_date}")
    print(f"  ✓ Sequence length: {market_data['sequence_length']} days")
    
    # Step 3: Load ML model
    print(f"\n[3/4] Loading Transformer model...")
    model = load_ml_model(MODEL_CHECKPOINT, use_gpu=USE_GPU)
    
    # Step 4: Run ML inference
    print(f"\n[4/4] Running ML inference...")
    
    if not ML_IMPORTS_SUCCESS:
        print("❌ ML pipeline not available. Simulating results...")
        # Fallback: return algo scores
        portfolio = {
            'selected_stocks': candidates_df.nlargest(top_n, 'composite_score')['tic'].tolist(),
            'predicted_scores': candidates_df.nlargest(top_n, 'composite_score')['composite_score'].tolist(),
            'source': 'algorithmic_only',
            'metadata': {
                'sector': sector,
                'date': prediction_date,
                'num_candidates': len(candidates_df)
            }
        }
    else:
        # Call YOUR ML pipeline inference function
        portfolio = inference_pipeline(
            market_data=market_data,
            model=model,
            sequence_length=market_data['sequence_length'],
            top_n=top_n
        )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"✅ ML PREDICTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTop {top_n} Selected Stocks:")
    print(f"{'Rank':<6} {'Ticker':<10} {'ML Score':<12} {'Algo Score':<12}")
    print(f"{'-'*50}")
    
    for i, (stock, score) in enumerate(zip(portfolio['selected_stocks'], portfolio['predicted_scores']), 1):
        algo_score = candidates_df[candidates_df['tic'] == stock]['composite_score'].values
        algo_score_str = f"{algo_score[0]:.4f}" if len(algo_score) > 0 else "N/A"
        print(f"{i:<6} {stock:<10} {score:<12.4f} {algo_score_str:<12}")
    
    # Save results
    if save_results:
        output_dir = Path(f"ml_integration/results/{sector}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{year}_{month:02d}_predictions.json"
        with open(output_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_portfolio = {
                k: v.tolist() if hasattr(v, 'tolist') else v 
                for k, v in portfolio.items()
            }
            json.dump(serializable_portfolio, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    return portfolio


def main():
    """Main entry point for ML integration"""
    parser = argparse.ArgumentParser(
        description='Run ML prediction on algorithmic candidates'
    )
    parser.add_argument(
        '--sector', '-s', 
        required=True,
        choices=list(SECTOR_GICS_MAPPING.keys()),
        help='Sector to analyze'
    )
    parser.add_argument(
        '--year', '-y',
        type=int,
        required=True,
        help='Year for prediction'
    )
    parser.add_argument(
        '--month', '-m',
        type=int,
        required=True,
        help='Month for prediction (1-12)'
    )
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=TOP_N_STOCKS,
        help=f'Number of top stocks to select (default: {TOP_N_STOCKS})'
    )
    parser.add_argument(
        '--candidate-type', '-t',
        choices=['long', 'short', 'both'],
        default='long',
        help='Type of candidates to process (default: long)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    # Validate month
    if not 1 <= args.month <= 12:
        print(f"❌ Invalid month: {args.month}. Must be 1-12.")
        sys.exit(1)
    
    # Run prediction
    try:
        portfolio = run_ml_prediction(
            sector=args.sector,
            year=args.year,
            month=args.month,
            top_n=args.top_n,
            candidate_type=args.candidate_type if args.candidate_type != 'both' else None,
            save_results=not args.no_save
        )
        
        print(f"\n{'='*80}")
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### **Step 4: Helper Script for Batch Processing**

**File:** `ml_integration/batch_predict.py`

```python
"""
Batch ML Prediction Runner

Run ML predictions across multiple time periods
"""

import argparse
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd

from ml_integration.run_inference import run_ml_prediction
from algo.core.config import SECTOR_GICS_MAPPING


def generate_month_range(
    start_year: int, 
    start_month: int,
    end_year: int,
    end_month: int
) -> List[Tuple[int, int]]:
    """Generate list of (year, month) tuples in range"""
    months = []
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    
    while current <= end:
        months.append((current.year, current.month))
        # Increment month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return months


def batch_predict(
    sector: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Run batch ML predictions across time period
    
    Args:
        sector: Sector name
        start_year: Start year
        start_month: Start month
        end_year: End year
        end_month: End month
        top_n: Number of stocks per period
        
    Returns:
        DataFrame with all predictions
    """
    months = generate_month_range(start_year, start_month, end_year, end_month)
    
    all_results = []
    
    print(f"Running batch prediction for {sector}: {len(months)} months")
    print("="*80)
    
    for year, month in months:
        try:
            portfolio = run_ml_prediction(
                sector=sector,
                year=year,
                month=month,
                top_n=top_n,
                save_results=True
            )
            
            # Store results
            for stock, score in zip(portfolio['selected_stocks'], portfolio['predicted_scores']):
                all_results.append({
                    'sector': sector,
                    'year': year,
                    'month': month,
                    'date': f"{year}-{month:02d}-01",
                    'ticker': stock,
                    'ml_score': score
                })
                
            print(f"✓ Completed {year}-{month:02d}")
            
        except Exception as e:
            print(f"✗ Failed {year}-{month:02d}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save consolidated results
    output_file = f"ml_integration/results/{sector}_batch_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Batch results saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch ML predictions')
    parser.add_argument('--sector', required=True, choices=list(SECTOR_GICS_MAPPING.keys()))
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--end-month', type=int, required=True)
    parser.add_argument('--top-n', type=int, default=10)
    
    args = parser.parse_args()
    
    batch_predict(
        sector=args.sector,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        top_n=args.top_n
    )
```

---

## Usage Instructions

### **Prerequisites**

1. ✅ Algorithmic pipeline has run and generated candidates:
   ```bash
   python algo/run_sector.py --sector healthcare
   ```

2. ✅ ML pipeline project is accessible at known path

3. ✅ Trained model checkpoint exists: `model_tt_50.ckpt`

### **Setup**

1. **Create integration module:**
   ```bash
   mkdir -p ml_integration
   touch ml_integration/__init__.py
   ```

2. **Copy the scripts** (from sections above) into:
   - `ml_integration/config.py`
   - `ml_integration/run_inference.py`
   - `ml_integration/batch_predict.py`

3. **Update configuration:**
   Edit `ml_integration/config.py` and set:
   ```python
   ML_PIPELINE_PATH = Path("/actual/path/to/ml-pipeline-project")
   ```

### **Running Predictions**

**Single month prediction:**
```bash
python -m ml_integration.run_inference \
    --sector healthcare \
    --year 2024 \
    --month 6 \
    --top-n 10
```

**Batch predictions (multiple months):**
```bash
python -m ml_integration.batch_predict \
    --sector healthcare \
    --start-year 2023 \
    --start-month 1 \
    --end-year 2024 \
    --end-month 6 \
    --top-n 10
```

**Process all sectors:**
```bash
#!/bin/bash
for sector in healthcare it financials energy; do
    echo "Processing $sector..."
    python -m ml_integration.run_inference \
        --sector $sector \
        --year 2024 \
        --month 6 \
        --top-n 10
done
```

---

## Expected Output

### **Console Output**
```
================================================================================
ML PREDICTION PIPELINE - HEALTHCARE 2024-06
================================================================================

[1/4] Loading long candidates from algorithmic pipeline...
✓ Loaded 85 long candidates for healthcare 2024-6

[2/4] Preparing ML pipeline input...
  ✓ 85 tickers prepared
  ✓ Prediction date: 2024-06-01
  ✓ Sequence length: 32 days

[3/4] Loading Transformer model...
✓ Model loaded on GPU

[4/4] Running ML inference...
  Fetching OHLCV data: 100%|███████████████| 85/85 [00:12<00:00, 6.8it/s]
  Computing features: 100%|█████████████████| 85/85 [00:03<00:00, 24.5it/s]
  Model inference: 100%|████████████████████| 1/1 [00:00<00:00, 45.2it/s]

================================================================================
✅ ML PREDICTION COMPLETE
================================================================================

Top 10 Selected Stocks:
Rank   Ticker     ML Score     Algo Score  
--------------------------------------------------
1      JNJ        0.9456       0.8234      
2      UNH        0.9123       0.7891      
3      PFE        0.8876       0.7654      
4      ABBV       0.8654       0.7432      
5      TMO        0.8432       0.7321      
6      ABT        0.8234       0.7198      
7      DHR        0.8123       0.7087      
8      LLY        0.8012       0.6976      
9      AMGN       0.7987       0.6854      
10     GILD       0.7876       0.6743      

✓ Results saved to: ml_integration/results/healthcare/2024_06_predictions.json
```

### **Output Files**

**Single prediction:** `ml_integration/results/{sector}/{year}_{month}_predictions.json`
```json
{
  "selected_stocks": ["JNJ", "UNH", "PFE", ...],
  "predicted_scores": [0.9456, 0.9123, 0.8876, ...],
  "portfolio_weights": {
    "JNJ": 0.121,
    "UNH": 0.117,
    "PFE": 0.114,
    ...
  },
  "metadata": {
    "sector": "healthcare",
    "date": "2024-06-01",
    "num_candidates": 85,
    "valid_stocks": 82,
    "excluded_stocks": ["XYZ", "ABC"],
    "model_version": "transformer_v1.0"
  }
}
```

**Batch predictions:** `ml_integration/results/{sector}_batch_results.csv`
```csv
sector,year,month,date,ticker,ml_score
healthcare,2023,1,2023-01-01,JNJ,0.9234
healthcare,2023,1,2023-01-01,UNH,0.9123
healthcare,2023,2,2023-02-01,PFE,0.8987
...
```

---

## Questions for ML Engineer (You)

To complete this integration, we need clarification on:

### **1. Interface Function** ✅ **CRITICAL**

Does your ML pipeline have a function like this?

```python
def inference_pipeline(
    market_data: Dict,           # Contains 'tickers', 'prediction_date'
    model: torch.nn.Module,      # Loaded model
    sequence_length: int = 32,   # Lookback window
    top_n: int = 10             # Number of stocks to return
) -> Dict:
    """
    Run inference given list of tickers
    
    Your function should:
    1. Fetch OHLCV data for tickers (internally)
    2. Build 25-feature tensor
    3. Run model inference
    4. Return top-N predictions
    """
    pass
```

**If YES:** Provide the exact function signature and import path  
**If NO:** Can you create this wrapper function?

### **2. Data Fetching** ✅

- How does your ML pipeline fetch OHLCV data?
- Do you use Yahoo Finance, Alpha Vantage, Bloomberg, or another source?
- Are there any API keys or credentials we need?

### **3. Model Access** ✅

- Where is the trained model checkpoint located?
- What is the exact path: `{ML_PROJECT}/checkpoints/model_tt_50.ckpt`?
- Are there multiple model versions? Which should we use?

### **4. Expected Behavior** ✅

- What happens if OHLCV data is missing for a ticker?
- Does your pipeline return `-1234` for invalid stocks or exclude them?
- What is the minimum data coverage required? (e.g., 90% of tickers must have data)

### **5. Performance** ✅

- How long does inference take for 50-100 stocks?
- Can we batch process multiple months efficiently?
- Are there any GPU memory considerations?

---

## Testing Strategy

### **Test 1: Import Verification**
```bash
# Verify ML pipeline can be imported
python -c "
import sys
sys.path.insert(0, '/path/to/ml-pipeline')
from model import TransformerStockPrediction
from inference import inference_pipeline
print('✅ ML pipeline imports successful')
"
```

### **Test 2: Single Stock Test**
```python
# Test with single known stock
from ml_integration.run_inference import run_ml_prediction

portfolio = run_ml_prediction(
    sector='healthcare',
    year=2024,
    month=6,
    top_n=5
)
print(f"Selected: {portfolio['selected_stocks']}")
```

### **Test 3: Historical Backtest**
```bash
# Test on historical data (where we have results)
python -m ml_integration.run_inference \
    --sector healthcare \
    --year 2023 \
    --month 12 \
    --top-n 10
```

### **Test 4: Performance Benchmark**
```python
import time
from ml_integration.run_inference import run_ml_prediction

start = time.time()
portfolio = run_ml_prediction('healthcare', 2024, 6)
elapsed = time.time() - start

print(f"Inference time: {elapsed:.2f}s")
assert elapsed < 60, "Inference should complete in <60s"
```

---

## Next Steps

### **For This Project (Algorithmic Pipeline Team)**

1. ✅ **Create `ml_integration/` folder** with provided scripts
2. ✅ **Update `config.py`** with ML pipeline path
3. ⏳ **Wait for ML engineer** to provide interface details
4. ⏳ **Test integration** with single example
5. ⏳ **Run batch predictions** on historical data

### **For ML Pipeline (Your Team)**

1. ✅ **Confirm interface function** exists or create it
2. ✅ **Provide model checkpoint** location
3. ✅ **Document OHLCV fetching** mechanism
4. ✅ **Share expected input/output** formats
5. ✅ **Test with sample ticker list** we provide

### **Joint Integration (Both Teams)**

1. ✅ **Integration meeting** to align on interface
2. ✅ **Code review** of integration scripts
3. ✅ **End-to-end test** with real data
4. ✅ **Performance optimization** if needed
5. ✅ **Documentation** of final workflow

---

## Success Criteria

The integration is complete when:

- ✅ We can pass ticker list from algo pipeline to ML pipeline
- ✅ ML pipeline returns top-N predictions successfully
- ✅ No data leakage (proper temporal alignment)
- ✅ Inference completes in <60 seconds for 50-100 stocks
- ✅ Results are saved and reproducible
- ✅ All edge cases handled (missing data, errors, etc.)

---

## Contact & Support

**Algorithmic Pipeline Lead:** [Your Name]  
**ML Pipeline Lead:** [Colleague's Name]  

**Integration Issues:** Create ticket in project management system  
**Code Questions:** See inline documentation in scripts  
**Model Questions:** Refer to ML Pipeline specification document  

---

## Appendix: Directory Structure

After integration, the project structure will be:

```
quant-hackathon/
├── algo/                          # ✅ Existing algorithmic pipeline
│   ├── core/
│   │   ├── candidate_selection.py
│   │   ├── indicators.py
│   │   ├── pipeline.py
│   │   └── config.py
│   ├── results/
│   │   └── {sector}_parquet/
│   │       ├── candidates/        # Input to ML pipeline
│   │       └── indicators/
│   └── run_sector.py
│
├── ml_integration/                # ⭐ NEW - Integration layer
│   ├── __init__.py
│   ├── config.py                 # ML pipeline configuration
│   ├── run_inference.py          # Single prediction script
│   ├── batch_predict.py          # Batch prediction script
│   └── results/                  # ML predictions output
│       └── {sector}/
│           ├── {year}_{month}_predictions.json
│           └── {sector}_batch_results.csv
│
├── cleaning/                      # ✅ Existing data cleaning
├── requirements.txt              # ✅ No changes needed
└── README.md                     # ✅ Update with integration docs
```

---

## Changelog

**Version 1.0** (2024-10-03)
- Initial integration design
- Created `ml_integration` module
- Defined interface requirements
- Documented testing strategy

---

**End of Integration Report**

*For questions or clarifications, please contact the integration team.*

