#!/usr/bin/env python3
"""
UNIFIED BACKTESTING ORCHESTRATOR
================================

Master script that orchestrates the complete backtesting workflow:
1. Runs comprehensive backtesting suite across all sectors and time periods
2. Performs enhanced portfolio analysis with real market data
3. Generates comprehensive reports and visualizations
4. Provides summary statistics and performance attribution

This script combines:
- comprehensive_backtesting_suite.py (pipeline execution and portfolio construction)
- enhanced_portfolio_analysis.py (detailed performance analysis)

Usage:
    # Default backtesting (Jan 2015 - May 2025)
    python run_complete_backtest.py

    # Custom parameters
    python run_complete_backtest.py --start-year 2018 --start-month 1 \
        --end-year 2023 --end-month 12 --top-n 3 --bottom-m 3

    # Quick test (shorter period)
    python run_complete_backtest.py --start-year 2023 --start-month 1 \
        --end-year 2023 --end-month 6 --max-workers 2

Author: GitHub Copilot
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

def run_comprehensive_backtest(start_year=2015, start_month=1, end_year=2025, end_month=5,
                             top_n=5, bottom_n=5, max_workers=4):
    """
    Run the comprehensive backtesting suite
    
    Args:
        start_year (int): Start year for backtesting
        start_month (int): Start month for backtesting  
        end_year (int): End year for backtesting
        end_month (int): End month for backtesting
        top_n (int): Number of top stocks per sector (long)
        bottom_n (int): Number of bottom stocks per sector (short)
        max_workers (int): Maximum parallel workers
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("STEP 1: Running Comprehensive Backtesting Suite")
    print("=" * 80)
    
    try:
        cmd = [
            sys.executable, "comprehensive_backtesting_suite.py",
            "--start-year", str(start_year),
            "--start-month", str(start_month),
            "--end-year", str(end_year),
            "--end-month", str(end_month),
            "--top-n", str(top_n),
            "--bottom-m", str(bottom_n),
            "--max-workers", str(max_workers)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, check=True, text=True)
        
        print("\nComprehensive backtesting completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nComprehensive backtesting failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nComprehensive backtesting failed: {str(e)}")
        return False

def run_enhanced_analysis(results_dir="results/backtesting"):
    """
    Run the enhanced portfolio analysis
    
    Args:
        results_dir (str): Directory containing backtesting results
    
    Returns:
        dict: Performance metrics, or None if failed
    """
    print("\nSTEP 2: Running Enhanced Portfolio Analysis")
    print("=" * 80)
    
    try:
        cmd = [
            sys.executable, "enhanced_portfolio_analysis.py",
            "--results-dir", results_dir
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        print(result.stdout)  # Print the analysis output
        
        # Try to load the generated metrics
        metrics_file = Path(results_dir) / "enhanced_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print("\nEnhanced portfolio analysis completed successfully!")
            return metrics
        else:
            print("\nEnhanced analysis completed but metrics file not found")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"\nEnhanced analysis failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return None
    except Exception as e:
        print(f"\nEnhanced analysis failed: {str(e)}")
        return None

def generate_executive_summary(metrics, backtest_params):
    """
    Generate an executive summary report
    
    Args:
        metrics (dict): Performance metrics from enhanced analysis
        backtest_params (dict): Backtesting parameters
    """
    print("\nSTEP 3: Generating Executive Summary")
    print("=" * 80)
    
    if not metrics:
        print("Cannot generate summary - no metrics available")
        return
    
    results_dir = Path("results/backtesting")
    summary_file = results_dir / "EXECUTIVE_SUMMARY.md"
    
    try:
        with open(summary_file, 'w') as f:
            f.write("# QUANTITATIVE STRATEGY BACKTESTING - EXECUTIVE SUMMARY\n\n")
            
            # Strategy overview
            f.write("## Strategy Overview\n\n")
            f.write(f"**Strategy:** Long/Short Equity across 11 GICS sectors\n")
            f.write(f"**Period:** {backtest_params['start_month']:02d}/{backtest_params['start_year']} - {backtest_params['end_month']:02d}/{backtest_params['end_year']}\n")
            f.write(f"**Portfolio Construction:** Top {backtest_params['top_n']} long + Bottom {backtest_params['bottom_n']} short per sector\n")
            f.write(f"**Rebalancing:** Monthly\n")
            f.write(f"**Total Months Analyzed:** {metrics.get('months_tracked', 'N/A')}\n\n")
            
            # Key performance metrics
            f.write("## Key Performance Metrics\n\n")
            f.write("| Metric | Portfolio | S&P 500 | Difference |\n")
            f.write("|--------|-----------|---------|------------|\n")
            
            port_return = metrics.get('annualized_return', 0)
            sp500_return = metrics.get('sp500_annualized_return', 0)
            excess_return = port_return - sp500_return
            
            port_vol = metrics.get('annual_volatility', 0)
            sp500_vol = metrics.get('sp500_annual_volatility', 0)
            
            f.write(f"| Annualized Return | {port_return:.2%} | {sp500_return:.2%} | {excess_return:+.2%} |\n")
            f.write(f"| Annual Volatility | {port_vol:.2%} | {sp500_vol:.2%} | {port_vol - sp500_vol:+.2%} |\n")
            f.write(f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.3f} | N/A | N/A |\n")
            f.write(f"| Maximum Drawdown | {metrics.get('max_drawdown', 0):.2%} | N/A | N/A |\n\n")
            
            # Risk-adjusted performance
            f.write("## Risk-Adjusted Performance\n\n")
            f.write(f"**Alpha (Annual):** {metrics.get('alpha_annual', 0):.2%}\n")
            f.write(f"**Beta:** {metrics.get('beta', 0):.3f}\n")
            f.write(f"**Information Ratio:** {metrics.get('information_ratio', 0):.3f}\n")
            f.write(f"**Sortino Ratio:** {metrics.get('sortino_ratio', 0):.3f}\n\n")
            
            # Statistical significance
            alpha_pvalue = metrics.get('alpha_pvalue', 1)
            alpha_significant = alpha_pvalue < 0.05
            
            f.write("## Statistical Significance\n\n")
            f.write(f"**Alpha t-statistic:** {metrics.get('alpha_tstat', 0):.3f}\n")
            f.write(f"**Alpha p-value:** {alpha_pvalue:.4f}\n")
            
            if alpha_significant:
                f.write("**Alpha is statistically significant at 95% confidence level**\n\n")
            else:
                f.write("**Alpha is not statistically significant at 95% confidence level**\n\n")
            
            # Performance highlights
            f.write("## Performance Highlights\n\n")
            f.write(f"- **Win Rate:** {metrics.get('win_rate', 0):.1%} of months were positive\n")
            f.write(f"- **Win Rate vs S&P 500:** {metrics.get('win_rate_vs_sp500', 0):.1%} of months outperformed benchmark\n")
            f.write(f"- **Best Month:** {metrics.get('best_month', 0):.2%} ({metrics.get('best_month_date', 'N/A')})\n")
            f.write(f"- **Worst Month:** {metrics.get('worst_month', 0):.2%} ({metrics.get('worst_month_date', 'N/A')})\n")
            f.write(f"- **VaR (95%):** {metrics.get('var_95', 0):.2%} monthly\n\n")
            
            # Strategy performance assessment
            f.write("## Strategy Assessment\n\n")
            
            # Overall performance grade
            if excess_return > 0.02 and alpha_significant:
                grade = "A"
                assessment = "Excellent"
            elif excess_return > 0 and metrics.get('sharpe_ratio', 0) > 1:
                grade = "B"
                assessment = "Good"
            elif excess_return > 0:
                grade = "C"
                assessment = "Satisfactory"
            else:
                grade = "D"
                assessment = "Needs Improvement"
            
            f.write(f"**Overall Grade: {grade} ({assessment})**\n\n")
            
            # Detailed assessment
            strengths = []
            weaknesses = []
            
            if excess_return > 0:
                strengths.append(f"Positive excess return of {excess_return:.2%} vs S&P 500")
            else:
                weaknesses.append(f"Negative excess return of {excess_return:.2%} vs S&P 500")
            
            if metrics.get('sharpe_ratio', 0) > 1:
                strengths.append(f"Strong Sharpe ratio of {metrics.get('sharpe_ratio', 0):.3f}")
            elif metrics.get('sharpe_ratio', 0) > 0.5:
                strengths.append(f"Reasonable Sharpe ratio of {metrics.get('sharpe_ratio', 0):.3f}")
            else:
                weaknesses.append(f"Low Sharpe ratio of {metrics.get('sharpe_ratio', 0):.3f}")
            
            if alpha_significant:
                strengths.append("Statistically significant alpha generation")
            else:
                weaknesses.append("Alpha is not statistically significant")
            
            if metrics.get('max_drawdown', 0) > -0.20:
                strengths.append(f"Controlled maximum drawdown of {metrics.get('max_drawdown', 0):.2%}")
            else:
                weaknesses.append(f"High maximum drawdown of {metrics.get('max_drawdown', 0):.2%}")
            
            if strengths:
                f.write("### Strengths\n")
                for strength in strengths:
                    f.write(f"- {strength}\n")
                f.write("\n")
            
            if weaknesses:
                f.write("### Areas for Improvement\n")
                for weakness in weaknesses:
                    f.write(f"- {weakness}\n")
                f.write("\n")
            
            # Conclusions and recommendations
            f.write("## Conclusions & Recommendations\n\n")
            
            if grade in ['A', 'B']:
                f.write("The quantitative strategy demonstrates strong performance characteristics ")
                f.write("with positive risk-adjusted returns. Consider:\n\n")
                f.write("- **Implementation:** Strategy shows promise for live trading\n")
                f.write("- **Optimization:** Fine-tune position sizing and rebalancing frequency\n")
                f.write("- **Risk Management:** Monitor drawdown levels and implement stop-losses\n")
            else:
                f.write("The strategy requires further development before implementation. Consider:\n\n")
                f.write("- **Model Enhancement:** Improve signal generation and stock selection\n")
                f.write("- **Risk Management:** Implement stronger risk controls\n")
                f.write("- **Parameter Optimization:** Test different portfolio construction methods\n")
            
            f.write("\n---\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            f.write("*Analysis based on simulated returns - implement with actual market data for live trading*\n")
        
        print(f"Executive summary generated: {summary_file}")
        
        # Print summary to console
        print(f"\nPERFORMANCE SUMMARY")
        print(f"   Strategy Grade: {grade} ({assessment})")
        print(f"   Annualized Return: {port_return:.2%}")
        print(f"   Excess Return vs S&P 500: {excess_return:+.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   Alpha (Annual): {metrics.get('alpha_annual', 0):.2%}")
        print(f"   Alpha Significant: {'Yes' if alpha_significant else 'No'}")
        
    except Exception as e:
        print(f"Failed to generate executive summary: {str(e)}")

def main():
    """Main orchestrator function"""
    parser = argparse.ArgumentParser(description="Complete Backtesting Orchestrator")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year (default: 2015)")
    parser.add_argument("--start-month", type=int, default=1, help="Start month (default: 1)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--end-month", type=int, default=5, help="End month (default: 5)")
    parser.add_argument("--top-n", type=int, default=5, help="Top N stocks per sector (default: 5)")
    parser.add_argument("--bottom-m", type=int, default=5, help="Bottom M stocks per sector (default: 5)")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers (default: 4)")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtesting, run analysis only")
    
    args = parser.parse_args()
    
    print("QUANTITATIVE STRATEGY BACKTESTING ORCHESTRATOR")
    print("=" * 80)
    print(f"Period: {args.start_month:02d}/{args.start_year} - {args.end_month:02d}/{args.end_year}")
    print(f"Strategy: Long top {args.top_n} + Short bottom {args.bottom_m} per sector")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    backtest_params = {
        'start_year': args.start_year,
        'start_month': args.start_month,
        'end_year': args.end_year,
        'end_month': args.end_month,
        'top_n': args.top_n,
        'bottom_n': args.bottom_m,
        'max_workers': args.max_workers
    }
    
    success = True
    
    # Step 1: Run comprehensive backtesting (unless skipped)
    if not args.skip_backtest:
        success = run_comprehensive_backtest(**backtest_params)
        if not success:
            print("\nBacktesting failed. Exiting.")
            return 1
    else:
        print("Skipping backtesting step (using existing results)")
    
    # Step 2: Run enhanced analysis
    metrics = run_enhanced_analysis()
    if not metrics:
        print("\nEnhanced analysis failed. Exiting.")
        return 1
    
    # Step 3: Generate executive summary
    generate_executive_summary(metrics, backtest_params)
    
    # Final summary
    print(f"\nCOMPLETE BACKTESTING WORKFLOW FINISHED!")
    print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Results directory: results/backtesting/")
    print(f"   Executive summary: results/backtesting/EXECUTIVE_SUMMARY.md")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())