#!/usr/bin/env python3
"""
BACKTESTING QUICK DEMO
======================

A quick demo of the comprehensive backtesting suite using a shorter time period
to test functionality before running the full 10+ year backtest.

This demo runs:
- 6 months of backtesting (Jan 2023 - Jun 2023)
- 2 stocks per sector (top 2 long, bottom 2 short)
- All 11 sectors
- Complete analysis pipeline

Usage:
    python backtest_quick_demo.py

Author: GitHub Copilot
"""

import sys
import subprocess
from datetime import datetime

def run_quick_demo():
    """Run a quick backtesting demo"""
    
    print("QUANTITATIVE BACKTESTING - QUICK DEMO")
    print("=" * 60)
    print("Testing the complete backtesting pipeline with:")
    print("   Period: Jan 2023 - Jun 2023 (6 months)")
    print("   Strategy: Top 2 long + Bottom 2 short per sector")
    print("   Sectors: All 11 GICS sectors")
    print("   Expected runtime: 5-10 minutes")
    print()
    
    # Demo parameters
    demo_params = [
        "--start-year", "2023",
        "--start-month", "1", 
        "--end-year", "2023",
        "--end-month", "6",
        "--top-n", "2",
        "--bottom-m", "2",
        "--max-workers", "2"  # Conservative for demo
    ]
    
    try:
        print("Launching complete backtesting workflow...")
        print(f"Command: python run_complete_backtest.py {' '.join(demo_params)}")
        print()
        
        # Run the complete workflow
        result = subprocess.run([
            sys.executable, "run_complete_backtest.py"
        ] + demo_params, check=True, text=True)
        
        print("\nQUICK DEMO COMPLETED SUCCESSFULLY!")
        print()
        print("Generated files:")
        print("   - results/backtesting/portfolio_returns.csv")
        print("   - results/backtesting/sp500_benchmark.csv") 
        print("   - results/backtesting/performance_metrics.json")
        print("   - results/backtesting/enhanced_performance_report.txt")
        print("   - results/backtesting/EXECUTIVE_SUMMARY.md")
        print()
        print("Next steps:")
        print("   1. Review the EXECUTIVE_SUMMARY.md for key findings")
        print("   2. If results look good, run full backtest with:")
        print("      python run_complete_backtest.py")
        print("   3. For custom periods, use parameters like:")
        print("      python run_complete_backtest.py --start-year 2018 --end-year 2024")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nQuick demo failed with return code {e.returncode}")
        print("\nTroubleshooting tips:")
        print("   1. Make sure all sector results are generated in algo/results/")
        print("   2. Check that main_pipeline.py is working correctly")
        print("   3. Verify Python environment has all required packages")
        print("   4. Try running individual components separately")
        return False
        
    except Exception as e:
        print(f"\nQuick demo failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_quick_demo()
    exit(0 if success else 1)