#!/usr/bin/env python3
"""
FinBERT Sentiment Analysis Pipeline - Main Runner

This is the main entry point for the FinBERT sentiment analysis system.
It orchestrates the complete pipeline from data preparation to Lightning.ai export.

Usage:
    python run_sentiment_pipeline.py

Files Structure:
    sentiment_analysis/
    ├── run_sentiment_pipeline.py       # This file - main runner
    ├── data_preparation/
    │   └── prepare_data.py             # Data extraction and formatting
    ├── lightning_ai/                   # Output folder for Lightning.ai files
    ├── results/                        # Final results folder
    └── README.md                       # Documentation

Author: GitHub Copilot
Date: October 4, 2025
"""

import sys
from pathlib import Path
import logging

# Add data_preparation to path
sys.path.append(str(Path(__file__).parent / "data_preparation"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if required packages and data are available"""
    
    print("Checking system requirements...")
    
    # Check Python packages
    required_packages = ['pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nERROR: Missing packages: {missing_packages}")
        print("Install with: pip install pandas numpy")
        return False
    
    # Check data availability
    textdata_path = Path(__file__).parent.parent / "TextData"
    
    if not textdata_path.exists():
        print(f"✗ TextData folder not found at: {textdata_path}")
        return False
    else:
        print(f"✓ TextData folder found")
    
    # Check specific data files
    data_2024 = textdata_path / "2024" / "text_us_2024.parquet"
    data_2023 = textdata_path / "2023" / "text_us_2023.parquet"
    
    if data_2024.exists():
        print(f"✓ 2024 text data available")
    else:
        print(f"⚠ 2024 text data not found")
    
    if data_2023.exists():
        print(f"✓ 2023 text data available")
    else:
        print(f"⚠ 2023 text data not found")
    
    if not (data_2024.exists() or data_2023.exists()):
        print("✗ No text data files found!")
        return False
    
    print("\n✓ All requirements satisfied")
    return True


def run_pipeline():
    """Run the complete sentiment analysis pipeline"""
    
    print("="*60)
    print("FINBERT SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    try:
        # Import the data preparation module
        from prepare_data import main as run_data_preparation
        
        # Run data preparation
        logger.info("Starting data preparation pipeline")
        success = run_data_preparation()
        
        if success:
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            lightning_dir = Path(__file__).parent / "lightning_ai"
            
            print(f"\nNext Steps:")
            print(f"1. Check the files in: {lightning_dir}")
            print(f"2. Upload to Lightning.ai studio")
            print(f"3. Follow the markdown instructions")
            print(f"4. Run FinBERT inference on Lightning.ai")
            print(f"5. Download results and process locally")
            
            # List created files
            if lightning_dir.exists():
                files = list(lightning_dir.glob("*"))
                if files:
                    print(f"\nFiles ready for Lightning.ai:")
                    for file in files:
                        size = file.stat().st_size if file.is_file() else 0
                        print(f"  - {file.name} ({size} bytes)")
            
            return True
        else:
            print("\n✗ Pipeline failed during data preparation")
            return False
            
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Make sure all required files are in the correct locations")
        return False
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        logger.error(f"Pipeline failed: {e}")
        return False


def show_help():
    """Show help information"""
    
    help_text = """
FinBERT Sentiment Analysis Pipeline
==================================

This pipeline extracts text data for stocks and prepares it for 
sentiment analysis using FinBERT on Lightning.ai.

USAGE:
    python run_sentiment_pipeline.py

WHAT IT DOES:
1. Scans TextData folder for available stocks
2. Extracts text from financial filings (10-Q, 10-K)
3. Formats text data for FinBERT processing
4. Creates Lightning.ai compatible files
5. Provides instructions for cloud inference

OUTPUT:
- CSV file with formatted text data
- Instructions for Lightning.ai processing
- Results processing script

REQUIREMENTS:
- pandas, numpy (install with: pip install pandas numpy)
- TextData folder with parquet files
- Lightning.ai account (for inference)

FILES CREATED:
- lightning_ai/finbert_input_*.csv        # Main input data
- lightning_ai/lightning_instructions_*.md # Processing guide
- lightning_ai/process_results.py          # Results processor

For more information, see README.md
"""
    
    print(help_text)


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        return
    
    # Check requirements
    if not check_requirements():
        print("\nPlease fix the issues above and try again.")
        return
    
    # Run pipeline
    success = run_pipeline()
    
    if success:
        print("\n🚀 Ready for Lightning.ai processing!")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")


if __name__ == "__main__":
    main()