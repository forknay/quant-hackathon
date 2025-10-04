# FinBERT Sentiment Analysis System

A complete pipeline for analyzing stock sentiment using FinBERT on Lightning.ai.

## Overview

This system extracts text data from financial filings and processes it through FinBERT (Financial BERT) to generate sentiment scores for stocks. It's designed to work with Lightning.ai for cloud-based GPU inference.

## What It Does

```
Input: Stock identifiers → Extract text data → Format for FinBERT → Cloud inference → Sentiment rankings
```

1. **Extracts** text from 10-Q and 10-K filings for selected stocks
2. **Formats** the text data for FinBERT processing  
3. **Exports** data in Lightning.ai compatible format
4. **Provides** complete instructions for cloud processing
5. **Processes** results into final sentiment rankings

## Quick Start

### 1. Install Requirements
```bash
pip install pandas numpy pyarrow
```

### 2. Run the Pipeline
```bash
cd sentiment_analysis
python run_sentiment_pipeline.py
```

### 3. Upload to Lightning.ai
- Upload files from `lightning_ai/` folder to Lightning.ai studio
- Follow the instructions in the generated markdown file
- Run FinBERT inference on Lightning.ai GPUs

### 4. Process Results
- Download sentiment results from Lightning.ai
- Use `process_results.py` to generate final rankings

## File Structure

```
sentiment_analysis/
├── run_sentiment_pipeline.py       # Main entry point
├── data_preparation/
│   └── prepare_data.py             # Data extraction & formatting
├── lightning_ai/                   # Generated files for Lightning.ai
│   ├── finbert_input_*.csv         # Main input data
│   ├── lightning_instructions_*.md # Processing guide  
│   ├── metadata_*.json             # Data metadata
│   └── process_results.py          # Results processor
├── results/                        # Final output folder
├── legacy/                         # Original development files
└── README.md                       # This file
```

## Input Data Requirements

The system expects TextData in this structure:
```
TextData/
├── 2023/
│   └── text_us_2023.parquet       # Full 2023 text data
└── 2024/
    └── text_us_2024.parquet       # Processed 2024 data
```

**Data Columns Expected:**
- **2024 format**: `gvkey`, `date`, `text`, `filing_type`
- **2023 format**: `gvkey`, `date`, `rf`, `mgmt`, `file_type`, `cusip`, `year`

## Output

### Lightning.ai Files
- **`finbert_input_*.csv`**: Formatted text data ready for FinBERT
- **`lightning_instructions_*.md`**: Complete processing guide
- **`process_results.py`**: Script to handle inference results

### Final Results
- **Stock sentiment rankings** (most positive to most negative)
- **Confidence scores** for each sentiment prediction
- **Text statistics** (count, length, sources per stock)

## Example Output

```
=== STOCK SENTIMENT RANKINGS ===
stock_id | category        | net_sentiment | rank | text_count
TOP_02   | top_performer   | +0.742       | 1    | 3
TOP_01   | top_performer   | +0.653       | 2    | 2  
...
BOT_05   | bottom_performer| -0.521       | 10   | 4
```

## Lightning.ai Processing

The system generates complete instructions for Lightning.ai, including:

```python
# Setup
!pip install transformers torch pandas
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Process your data (see generated instructions for complete code)
```

## Configuration

### Customizable Parameters
- **Text length limits**: Min 50, Max 2000 characters
- **Batch size**: Default 16 (adjust for GPU memory)
- **Stock selection**: Modify in `prepare_data.py`
- **Output directory**: Change in `LightningAIExporter`

### Performance Optimization
- **Batch processing**: Handles large datasets efficiently
- **Memory management**: Chunks large files automatically  
- **GPU optimization**: Optimized for Lightning.ai infrastructure

## Troubleshooting

### Common Issues

**1. No TextData found**
- Ensure TextData folder exists in parent directory
- Check that parquet files are present

**2. No stocks extracted**  
- Verify parquet file format matches expected columns
- Check that stock identifiers exist in the data

**3. Import errors**
- Install required packages: `pip install pandas numpy pyarrow`
- Ensure Python 3.7+ is being used

**4. Lightning.ai processing fails**
- Check GPU memory limits (reduce batch size)
- Verify FinBERT model loading
- Review token limits (max 512 tokens per text)

### Debug Mode
Add logging to see detailed processing:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Stock Selection
Modify the stock selection in `prepare_data.py`:
```python
# Instead of auto-detection, specify stocks:
selected_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
```

### Different FinBERT Models
Change the model in exported instructions:
```python
model_name = "ProsusAI/finbert"  # Alternative FinBERT model
```

### Local Processing (if you have GPU)
The formatted CSV can be used with any FinBERT implementation:
```python
import pandas as pd
df = pd.read_csv('finbert_input_*.csv')
# Use df['text'] with your local FinBERT setup
```

## Technical Details

### FinBERT Model
- **Model**: yiyanghkust/finbert-tone
- **Task**: Financial sentiment classification  
- **Output**: Positive/Negative/Neutral probabilities
- **Max tokens**: 512 (BERT limit)

### Data Processing
- **Text cleaning**: Removes empty/short texts
- **Length normalization**: Truncates long texts
- **Batch optimization**: Groups texts for efficient processing
- **Metadata preservation**: Tracks sources and dates

### Performance
- **Typical processing**: 40-50 texts in ~2-3 minutes on Lightning.ai
- **Scalability**: Handles hundreds of stocks with chunked processing
- **Memory efficiency**: Processes large datasets without memory issues

## Contributing

To extend this system:

1. **Add new data sources**: Modify `StockTextExtractor`
2. **Change output format**: Update `LightningAIExporter`  
3. **Add preprocessing**: Extend `FinBERTDataFormatter`
4. **Custom models**: Update Lightning.ai instructions

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generated log files
3. Ensure all requirements are met
4. Test with a small subset of data first

---

**Ready to analyze stock sentiment with FinBERT!** 🚀📈