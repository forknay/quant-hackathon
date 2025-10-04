# Main Quantitative Trading Pipeline

## Overview

The main pipeline orchestrates the complete quantitative trading workflow:

1. **ALGORITHM** → Run sector analysis to identify candidate stocks
2. **CANDIDATE SELECTION** → Extract top N (long) and bottom M (short) candidates with confidence scores
3. **DATA PREPARATION** → Generate filtered OHLCV data for each batch (preventing look-ahead bias)
4. **ML INFERENCE** → Run pre-trained model on each batch to get ML confidence scores  
5. **PORTFOLIO CONSTRUCTION** → Combine algorithm + ML scores for final portfolio weights
6. **RESULTS** → Save complete portfolio with weights and confidence scores

## Key Features

- **Look-ahead Bias Prevention**: Automatically filters OHLCV data to exclude future information
- **Dual Scoring System**: Combines algorithm confidence with ML model predictions
- **Batch Processing**: Processes long and short candidates separately for memory efficiency
- **Flexible Weighting**: Configurable weights for algorithm vs ML scores
- **Complete Traceability**: Saves all intermediate scores and metadata

## Usage

### Basic Usage

```bash
# Run pipeline for healthcare sector, June 2024
python main_pipeline.py --sector healthcare --year 2024 --month 6
```

### Advanced Usage

```bash
# Custom parameters with specific model and output
python main_pipeline.py \
    --sector healthcare \
    --year 2024 \
    --month 6 \
    --top-n 20 \
    --bottom-m 15 \
    --model ml-model/models/pre_train_models/market-NASDAQ_days-32_feature-describe-all_ongoing-task-stock_mask_rate-0.3_lr-0.001_pretrain-coefs-1-0-0/model_tv_100.ckpt \
    --output results/portfolio_2024_06_healthcare.json \
    --algo-weight 0.7 \
    --ml-weight 0.3
```

### Skip Algorithm Step

If you've already run the algorithm and want to use existing results:

```bash
python main_pipeline.py \
    --sector healthcare \
    --year 2024 \
    --month 6 \
    --skip-algo
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sector` | Sector to analyze (required) | - |
| `--year` | Target year for predictions (required) | - |
| `--month` | Target month for predictions (required) | - |
| `--top-n` | Number of long positions | 20 |
| `--bottom-m` | Number of short positions | 15 |
| `--model` | Path to ML model checkpoint | Default pre-trained model |
| `--output` | Output path for results | Auto-generated |
| `--algo-weight` | Weight for algorithm scores | 0.6 |
| `--ml-weight` | Weight for ML scores | 0.4 |
| `--skip-algo` | Skip algorithm execution | False |
| `--list-sectors` | List available sectors | - |

## Available Sectors

- `energy` (GICS: 10)
- `materials` (GICS: 15)
- `industrials` (GICS: 20)
- `cons_discretionary` (GICS: 25)
- `cons_staples` (GICS: 30)
- `healthcare` (GICS: 35)
- `financials` (GICS: 40)
- `it` (GICS: 45)
- `telecoms` (GICS: 50)
- `utilities` (GICS: 55)
- `re` (GICS: 60)

## Output Format

The pipeline generates a JSON file with the following structure:

```json
{
  "long_positions": [
    {
      "company_id": "comp_001004_01",
      "gvkey": 1004,
      "iid": "01",
      "algo_score_raw": 0.8534,
      "algo_score_normalized": 0.9123,
      "ml_score_raw": 0.7234,
      "ml_score_normalized": 0.8456,
      "combined_score": 0.8832,
      "portfolio_weight": 0.05,
      "candidate_type": "momentum_growth"
    }
  ],
  "short_positions": [...],
  "summary": {
    "total_long_positions": 20,
    "total_short_positions": 15,
    "total_positions": 35
  },
  "parameters": {
    "algo_weight": 0.6,
    "ml_weight": 0.4
  },
  "metadata": {
    "sector": "healthcare",
    "target_year": 2024,
    "target_month": 6,
    "creation_timestamp": "2025-10-04T..."
  }
}
```

## Pipeline Steps in Detail

### Step 1: Algorithm Execution
- Sets environment variables for the target sector
- Runs the sector-specific algorithm to identify candidates
- Generates candidate files with confidence scores

### Step 2: Candidate Extraction
- Loads algorithm results for the specified year/month
- Sorts candidates by composite score
- Selects top N for long positions, bottom M for short positions

### Step 3: Data Preparation (Batch Processing)
- **Look-ahead Bias Prevention**: Filters OHLCV data to exclude dates after the prediction month
- Creates temporary filtered CSV files for each batch
- Processes filtered data using `data.py` to generate feature matrices
- Cleans up temporary files after processing

### Step 4: ML Inference (Dual Runs)
- Loads the pre-trained ML model
- Runs inference on the long candidate batch
- Runs inference on the short candidate batch
- Collects ML confidence scores for each company

### Step 5: Portfolio Construction
- Normalizes algorithm scores within each batch
- Normalizes ML scores within each batch
- Combines scores using weighted average: `combined = algo_weight * algo_score + ml_weight * ml_score`
- Calculates portfolio weights (equal weight within position type)
- Ranks positions by combined score

### Step 6: Results Saving
- Combines all data into comprehensive portfolio structure
- Saves as JSON with complete traceability
- Includes metadata for reproducibility

## Look-ahead Bias Prevention

The pipeline automatically prevents look-ahead bias by:

1. **Date Cutoff Calculation**: For predictions in month M of year Y, data is cut off at the last day of month M-1
2. **OHLCV Filtering**: All OHLCV CSV files are filtered to remove future dates before processing
3. **Data Processing**: The filtered data is then processed normally with `data.py`
4. **Validation**: The system logs the cutoff date and confirms no future data is included

Example: For predictions in June 2024, all data after May 31, 2024 is excluded.

## Dependencies

- Python 3.8+
- pandas, numpy
- PyTorch (for ML model)
- All quantitative algorithm dependencies
- Pre-trained ML model checkpoint

## Error Handling

The pipeline includes comprehensive error handling:

- **Missing Data**: Graceful handling of missing OHLCV files
- **Insufficient Data**: Skips companies with insufficient historical data
- **Model Errors**: Detailed error reporting for ML inference issues
- **File I/O**: Robust file operations with cleanup on failure

## Testing

Use the test script to verify the pipeline works:

```bash
python test_main_pipeline.py
```

Or test specific components:

```bash
# Test end-to-end with small dataset
python test/end_to_end_pipeline_test.py --max-companies 5

# Test with real model
python test/end_to_end_pipeline_test.py --use-real-model --model path/to/model.ckpt
```

## Performance Notes

- **Memory Usage**: Batched processing keeps memory usage manageable
- **Processing Time**: Typically 5-15 minutes depending on dataset size
- **Disk Space**: Temporary files are cleaned up automatically
- **Parallelization**: Algorithm step uses parallel processing where possible

## Troubleshooting

### Common Issues

1. **"No candidates found"**: Ensure the algorithm has been run for the target sector/period
2. **"Model not found"**: Check the model path is correct and file exists
3. **"Insufficient data"**: Some companies may not have enough historical data
4. **Memory errors**: Reduce `--top-n` and `--bottom-m` for testing

### Debug Steps

1. Run with `--skip-algo` to isolate data/ML issues
2. Check algorithm results in `algo/results/{sector}_parquet/`
3. Verify OHLCV data exists in `inference/company_ohlcv_data/`
4. Test ML model with smaller datasets first