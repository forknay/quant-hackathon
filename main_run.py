#!/usr/bin/env python3
"""
MAIN PIPELINE RUNNER
====================

Complete end-to-end quantitative trading system execution:
1. Data Cleaning & Preprocessing
2. Multi-Sector Algorithm Processing 
3. Comprehensive Backtesting & Analysis

Usage:
    # Full pipeline with default settings
    python main_run.py

    # Custom backtesting period
    python main_run.py --start-year 2018 --start-month 1 --end-year 2023 --end-month 12

    # Custom portfolio parameters
    python main_run.py --top-n 3 --bottom-m 3 --max-workers 6

    # Skip data cleaning (if already done)
    python main_run.py --skip-cleaning

    # Run specific sectors only
    python main_run.py --sectors energy healthcare it

Author: Hierarchical Multi-Modal Signal Processing Team
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Optional
import json

class MainPipelineRunner:
    """
    Orchestrates the complete quantitative trading pipeline
    """
    
    def __init__(self, skip_cleaning=False, sectors=None, **backtest_args):
        """
        Initialize the main pipeline runner
        
        Args:
            skip_cleaning (bool): Skip data cleaning step
            sectors (List[str]): Specific sectors to process (default: all 11)
            **backtest_args: Arguments for backtesting suite
        """
        self.project_root = Path(__file__).parent
        self.skip_cleaning = skip_cleaning
        self.backtest_args = backtest_args
        
        # All 11 sectors (matching GICS classification)
        self.ALL_SECTORS = [
            'energy', 'materials', 'industrials', 'cons_discretionary', 
            'cons_staples', 'healthcare', 'financials', 'it', 
            'telecoms', 'utilities', 're'
        ]
        
        # Use specified sectors or all sectors
        self.sectors_to_process = sectors if sectors else self.ALL_SECTORS
        
        # Required input files
        self.required_files = [
            'Data/ret_sample.csv',  # Main data file
            'sectorinfo.csv'   # Sector mapping file (renamed from sector file)
        ]
        
        print(f"MAIN PIPELINE RUNNER INITIALIZED")
        print(f"   Project Root: {self.project_root}")
        print(f"   Skip Cleaning: {skip_cleaning}")
        print(f"   Sectors to Process: {len(self.sectors_to_process)}")
        print(f"   Backtesting Period: {backtest_args.get('start_month', 1)}/{backtest_args.get('start_year', 2015)} - {backtest_args.get('end_month', 5)}/{backtest_args.get('end_year', 2025)}")
        print(f"   Portfolio Config: Top {backtest_args.get('top_n', 5)} / Bottom {backtest_args.get('bottom_m', 5)}")

    def check_prerequisites(self) -> bool:
        """
        Check that all required input files exist
        
        Returns:
            bool: True if all files exist, False otherwise
        """
        print(f"\nSTEP 0: CHECKING PREREQUISITES")
        print(f"   Checking for required input files...")
        
        all_exist = True
        
        for file_path in self.required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                file_size = full_path.stat().st_size / (1024**2)  # MB
                print(f"   [OK] {file_path} ({file_size:.1f} MB)")
            else:
                print(f"   [MISSING] {file_path} - NOT FOUND")
                all_exist = False
        
        if not all_exist:
            print(f"\nPREREQUISITE CHECK FAILED")
            print(f"   Required files missing. Please ensure the following files are in your directory:")
            for file_path in self.required_files:
                print(f"   - {file_path}")
            print(f"\n   Note: sectorinfo.csv should be your sector mapping file (renamed)")
            return False
        
        print(f"   All required files found!")
        return True

    def run_cleaning(self) -> bool:
        """
        Execute data cleaning pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.skip_cleaning:
            print(f"\nSTEP 1: DATA CLEANING - SKIPPED")
            
            # Check if cleaned data exists
            output_file = self.project_root / "cleaning" / "cleaned_all.parquet"
            if output_file.exists():
                file_size = output_file.stat().st_size / (1024**2)  # MB
                print(f"   Using existing cleaned data: {output_file.name} ({file_size:.1f} MB)")
                return True
            else:
                print(f"   Warning: Cleaned data not found at {output_file}")
                print(f"   You may need to run cleaning first!")
                return False
        
        print(f"\nSTEP 1: DATA CLEANING")
        print(f"   Running cleaning/clean_all.py...")
        
        try:
            # Change to project root and run cleaning
            cmd = [sys.executable, "cleaning/clean_all.py"]
            
            print(f"   Command: {' '.join(cmd)}")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(f"   {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print(f"   Data cleaning completed successfully!")
                return True
            else:
                print(f"   Data cleaning failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"   Error during data cleaning: {e}")
            return False

    def run_sector_processing(self) -> bool:
        """
        Execute algorithm processing for all specified sectors
        
        Returns:
            bool: True if all sectors processed successfully, False otherwise
        """
        print(f"\nSTEP 2: MULTI-SECTOR ALGORITHM PROCESSING")
        print(f"   Processing {len(self.sectors_to_process)} sectors...")
        
        # List all sectors first
        print(f"   Sectors to process:")
        for i, sector in enumerate(self.sectors_to_process, 1):
            print(f"     {i:2d}. {sector}")
        
        failed_sectors = []
        successful_sectors = []
        
        for i, sector in enumerate(self.sectors_to_process, 1):
            print(f"\n   [{i}/{len(self.sectors_to_process)}] Processing {sector.upper()} sector...")
            
            try:
                # Run algo/run_sector.py for this sector
                cmd = [sys.executable, "algo/run_sector.py", "--sector", sector]
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per sector
                )
                
                if result.returncode == 0:
                    print(f"   [OK] {sector} completed successfully")
                    successful_sectors.append(sector)
                else:
                    print(f"   [FAIL] {sector} failed (return code: {result.returncode})")
                    if result.stderr:
                        print(f"      Error: {result.stderr[:200]}...")
                    failed_sectors.append(sector)
                    
            except subprocess.TimeoutExpired:
                print(f"   [TIMEOUT] {sector} timed out (>5 minutes)")
                failed_sectors.append(sector)
            except Exception as e:
                print(f"   [ERROR] {sector} error: {e}")
                failed_sectors.append(sector)
        
        # Summary
        print(f"\n   SECTOR PROCESSING SUMMARY:")
        print(f"   Successful: {len(successful_sectors)}/{len(self.sectors_to_process)}")
        print(f"   Failed: {len(failed_sectors)}")
        
        if failed_sectors:
            print(f"   Failed sectors: {', '.join(failed_sectors)}")
        
        # Consider successful if at least 50% of sectors completed
        success_rate = len(successful_sectors) / len(self.sectors_to_process)
        return success_rate >= 0.5

    def run_backtesting(self) -> bool:
        """
        Execute comprehensive backtesting suite
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\nSTEP 3: COMPREHENSIVE BACKTESTING")
        print(f"   Running comprehensive_backtesting_suite.py...")
        
        try:
            # Build command with backtesting arguments
            cmd = [sys.executable, "comprehensive_backtesting_suite.py"]
            
            # Add command line arguments
            for arg_name, arg_value in self.backtest_args.items():
                if arg_value is not None:
                    # Convert underscores to dashes for command line
                    arg_flag = f"--{arg_name.replace('_', '-')}"
                    cmd.extend([arg_flag, str(arg_value)])
            
            print(f"   Command: {' '.join(cmd)}")
            print(f"   This may take several minutes...")
            
            # Run backtesting with real-time output
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output with filtering for key updates
            for line in process.stdout:
                line_stripped = line.rstrip()
                # Show important progress lines
                if any(keyword in line_stripped.lower() for keyword in 
                       ['progress:', 'completed', 'summary', 'r²', 'sharpe', 'alpha', 'error', 'failed']):
                    print(f"   {line_stripped}")
                elif line_stripped.startswith('   ') and len(line_stripped) < 100:
                    print(f"   {line_stripped}")
            
            process.wait()
            
            if process.returncode == 0:
                print(f"   Backtesting completed successfully!")
                return True
            else:
                print(f"   Backtesting failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"   Error during backtesting: {e}")
            return False

    def generate_final_report(self):
        """
        Generate a final summary report of the entire pipeline run
        """
        print(f"\nPIPELINE EXECUTION SUMMARY")
        print(f"=" * 80)
        
        # Check for key output files
        key_outputs = {
            'Cleaned Data': 'cleaning/cleaned_all.parquet',
            'Sector Results': 'results/',
            'Backtesting Results': 'results/backtesting/backtest_summary_report.txt',
            'Performance Metrics': 'results/backtesting/performance_metrics.json'
        }
        
        print(f"OUTPUT FILES:")
        for description, path in key_outputs.items():
            full_path = self.project_root / path
            if full_path.exists():
                if full_path.is_file():
                    size = full_path.stat().st_size / (1024**2)  # MB
                    print(f"   [OK] {description}: {path} ({size:.1f} MB)")
                else:
                    # Directory - count files
                    try:
                        file_count = len(list(full_path.glob('*.*')))
                        print(f"   [OK] {description}: {path} ({file_count} files)")
                    except:
                        print(f"   [OK] {description}: {path}")
            else:
                print(f"   [MISSING] {description}: {path} - NOT FOUND")
        
        # Try to show key performance metrics if available
        try:
            metrics_file = self.project_root / 'results/backtesting/performance_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                print(f"\nKEY PERFORMANCE METRICS:")
                print(f"   Annual Return:        {metrics.get('avg_annual_return', 0):.2%}")
                print(f"   Annual Volatility:    {metrics.get('annual_volatility', 0):.2%}")
                print(f"   Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"   Annual Alpha:         {metrics.get('annual_alpha', 0):.2%}")
                print(f"   OOS R² vs S&P 500:    {metrics.get('oos_r2_vs_sp500', 0):.4f}")
                print(f"   Max Drawdown:         {metrics.get('max_drawdown', 0):.2%}")
                print(f"   Months Tracked:       {metrics.get('months_tracked', 0)}")
        except Exception as e:
            print(f"   (Could not load performance metrics: {e})")
        
        print(f"=" * 80)
        print(f"HIERARCHICAL MULTI-MODAL PIPELINE EXECUTION COMPLETE!")

    def run_full_pipeline(self) -> bool:
        """
        Execute the complete pipeline
        
        Returns:
            bool: True if entire pipeline successful, False otherwise
        """
        start_time = time.time()
        
        print(f"\nSTARTING COMPLETE QUANTITATIVE TRADING PIPELINE")
        print(f"   Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 0: Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Step 1: Data cleaning
        if not self.run_cleaning():
            print(f"\nPIPELINE FAILED at Step 1: Data Cleaning")
            return False
        
        # Step 2: Sector processing
        if not self.run_sector_processing():
            print(f"\nPIPELINE FAILED at Step 2: Sector Processing")
            return False
        
        # Step 3: Backtesting
        if not self.run_backtesting():
            print(f"\nPIPELINE FAILED at Step 3: Backtesting")
            return False
        
        # Final report
        elapsed = time.time() - start_time
        print(f"\nCOMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
        print(f"   Total Runtime: {elapsed/60:.1f} minutes")
        
        self.generate_final_report()
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Complete End-to-End Quantitative Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_run.py                                    # Full pipeline, default settings
  python main_run.py --skip-cleaning                   # Skip data cleaning step
  python main_run.py --start-year 2018 --end-year 2023 # Custom backtest period
  python main_run.py --sectors energy healthcare it    # Process specific sectors only
  python main_run.py --top-n 3 --bottom-m 3            # Custom portfolio parameters
        """
    )
    
    # Pipeline control arguments
    parser.add_argument("--skip-cleaning", action="store_true",
                       help="Skip data cleaning step (use existing cleaned data)")
    parser.add_argument("--sectors", nargs="+", 
                       choices=['energy', 'materials', 'industrials', 'cons_discretionary', 
                               'cons_staples', 'healthcare', 'financials', 'it', 
                               'telecoms', 'utilities', 're'],
                       help="Specific sectors to process (default: all 11)")
    
    # Backtesting arguments (passed through to comprehensive_backtesting_suite.py)
    parser.add_argument("--start-year", type=int, default=2015,
                       help="Backtesting start year (default: 2015)")
    parser.add_argument("--start-month", type=int, default=1,
                       help="Backtesting start month (default: 1)")
    parser.add_argument("--end-year", type=int, default=2025,
                       help="Backtesting end year (default: 2025)")
    parser.add_argument("--end-month", type=int, default=5,
                       help="Backtesting end month (default: 5)")
    parser.add_argument("--top-n", type=int, default=5,
                       help="Top N stocks per sector for long positions (default: 5)")
    parser.add_argument("--bottom-m", type=int, default=5,
                       help="Bottom M stocks per sector for short positions (default: 5)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum parallel workers for backtesting (default: 4)")
    
    args = parser.parse_args()
    
    # Extract backtesting arguments
    backtest_args = {
        'start_year': args.start_year,
        'start_month': args.start_month,
        'end_year': args.end_year,
        'end_month': args.end_month,
        'top_n': args.top_n,
        'bottom_m': args.bottom_m,
        'max_workers': args.max_workers
    }
    
    # Create and run pipeline
    runner = MainPipelineRunner(
        skip_cleaning=args.skip_cleaning,
        sectors=args.sectors,
        **backtest_args
    )
    
    success = runner.run_full_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()