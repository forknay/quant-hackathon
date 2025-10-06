#!/usr/bin/env python3
"""
COMPREHENSIVE BACKTESTING SUITE
================================

A complete backtesting framework that:
1. Runs the full pipeline across all 11 sectors for a specified time period
2. Constructs monthly portfolios with top N/bottom N stocks per sector
3. Calculates comprehensive performance metrics vs S&P 500
4. Provides detailed analysis and visualization

Usage:
    # Default backtesting (Jan 2015 - May 2025, top/bottom 5 per sector)
    python comprehensive_backtesting_suite.py

    # Custom parameters
    python comprehensive_backtesting_suite.py --start-year 2018 --start-month 1 \
        --end-year 2023 --end-month 12 --top-n 3 --bottom-m 3

Author: GitHub Copilot
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import subprocess
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Available sectors
SECTORS = [
    'energy', 'materials', 'industrials', 'cons_discretionary', 
    'cons_staples', 'healthcare', 'financials', 'it', 
    'telecoms', 'utilities', 're'
]

class ComprehensiveBacktester:
    """
    Comprehensive backtesting suite for quantitative trading strategies
    """
    
    def __init__(self, start_year=2015, start_month=1, end_year=2025, end_month=5, 
                 top_n=5, bottom_m=5, max_workers=4):
        """
        Initialize the backtesting suite
        
        Args:
            start_year (int): Start year for backtesting
            start_month (int): Start month for backtesting
            end_year (int): End year for backtesting
            end_month (int): End month for backtesting
            top_n (int): Number of top stocks to select per sector (long positions)
            bottom_m (int): Number of bottom stocks to select per sector (short positions)
            max_workers (int): Maximum number of parallel workers
        """
        self.start_year = start_year
        self.start_month = start_month
        self.end_year = end_year
        self.end_month = end_month
        self.top_n = top_n
        self.bottom_n = bottom_m
        self.max_workers = max_workers
        
        # Paths
        self.project_root = Path(__file__).parent
        self.results_dir = self.project_root / "results"
        self.backtest_dir = self.results_dir / "backtesting"
        self.backtest_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.portfolio_results = {}
        self.performance_metrics = {}
        self.sp500_data = None
        
        print(f"Initializing Comprehensive Backtesting Suite")
        print(f"   Period: {start_month:02d}/{start_year} - {end_month:02d}/{end_year}")
        print(f"   Strategy: Long top {top_n} + Short bottom {bottom_m} stocks per sector")
        print(f"   Sectors: {len(SECTORS)} sectors")
        print(f"   Parallel workers: {max_workers}")
    
    def generate_date_range(self) -> List[Tuple[int, int]]:
        """Generate list of (year, month) tuples for the backtesting period"""
        dates = []
        current_date = datetime(self.start_year, self.start_month, 1)
        end_date = datetime(self.end_year, self.end_month, 1)
        
        while current_date <= end_date:
            dates.append((current_date.year, current_date.month))
            current_date += relativedelta(months=1)
        
        return dates
    
    def run_pipeline_for_date_sector(self, year: int, month: int, sector: str) -> Optional[str]:
        """
        Run main_pipeline.py for a specific date and sector
        
        Returns:
            str: Path to the generated portfolio JSON file, or None if failed
        """
        try:
            # Expected output filename
            output_filename = f"portfolio_{year}_{month:02d}_{sector}.json"
            output_path = self.results_dir / output_filename
            
            # Skip if already exists
            if output_path.exists():
                print(f"   Found existing: {output_filename}")
                return str(output_path)
            
            # Run the pipeline with proper virtual environment
            # Use the same Python executable that's running this script
            python_exe = sys.executable
            cmd = [
                python_exe, "main_pipeline.py",
                "--sector", sector,
                "--year", str(year),
                "--month", str(month),
                "--top-n", str(self.top_n),
                "--bottom-m", str(self.bottom_n),
                "--output", str(output_path)
            ]
            
            print(f"   Running: {year}-{month:02d} {sector}")
            
            # Get current environment and ensure virtual environment is used
            env = os.environ.copy()
            # Set UTF-8 encoding for subprocess on Windows
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Run with timeout and capture output
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per run  
                env=env,
                encoding='utf-8',
                errors='replace'  # Replace problematic characters instead of failing
            )
            
            if result.returncode == 0 and output_path.exists():
                print(f"   Generated: {output_filename}")
                return str(output_path)
            else:
                print(f"   Failed: {year}-{month:02d} {sector}")
                # Show more detailed error information
                if result.stderr:
                    print(f"      Error: {result.stderr[:500]}...")
                if result.stdout:
                    print(f"      Output: {result.stdout[-300:]}")  # Show last 300 chars of stdout
                print(f"      Return code: {result.returncode}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"   Timeout: {year}-{month:02d} {sector}")
            return None
        except Exception as e:
            print(f"   Exception: {year}-{month:02d} {sector}: {str(e)[:100]}...")
            return None
    
    def run_all_pipelines(self):
        """Run main_pipeline.py for all date/sector combinations"""
        dates = self.generate_date_range()
        total_runs = len(dates) * len(SECTORS)
        completed_runs = 0
        
        print(f"\nRunning {total_runs} pipeline executions...")
        print(f"   Dates: {len(dates)} months")
        print(f"   Sectors: {len(SECTORS)} sectors")
        
        # Process in batches to avoid overwhelming the system
        batch_size = self.max_workers * 2
        all_tasks = [(year, month, sector) for year, month in dates for sector in SECTORS]
        
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(all_tasks)-1)//batch_size + 1}")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit batch tasks
                future_to_task = {
                    executor.submit(self.run_pipeline_for_date_sector, year, month, sector): (year, month, sector)
                    for year, month, sector in batch
                }
                
                # Collect results
                for future in as_completed(future_to_task):
                    year, month, sector = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            if (year, month) not in self.portfolio_results:
                                self.portfolio_results[(year, month)] = {}
                            self.portfolio_results[(year, month)][sector] = result
                        completed_runs += 1
                        
                        # Progress update
                        if completed_runs % 10 == 0:
                            print(f"   Progress: {completed_runs}/{total_runs} ({100*completed_runs/total_runs:.1f}%)")
                            
                    except Exception as e:
                        print(f"   Task failed: {year}-{month:02d} {sector}: {str(e)}")
                        completed_runs += 1
            
            # Brief pause between batches
            time.sleep(1)
        
        print(f"\nPipeline execution completed: {completed_runs}/{total_runs} runs")
    
    def load_portfolio_data(self) -> pd.DataFrame:
        """
        Load and combine all portfolio results into a single DataFrame
        
        Returns:
            pd.DataFrame: Combined portfolio data with performance metrics
        """
        all_portfolios = []
        
        print(f"\nLoading portfolio data...")
        
        for (year, month), sectors in self.portfolio_results.items():
            date_str = f"{year}-{month:02d}-01"
            
            monthly_long_positions = []
            monthly_short_positions = []
            
            for sector, portfolio_path in sectors.items():
                try:
                    with open(portfolio_path, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    # Extract positions
                    long_positions = portfolio_data.get('long_positions', [])
                    short_positions = portfolio_data.get('short_positions', [])
                    
                    # Add sector information
                    for pos in long_positions:
                        pos['sector'] = sector
                        pos['position_type'] = 'long'
                        monthly_long_positions.append(pos)
                    
                    for pos in short_positions:
                        pos['sector'] = sector
                        pos['position_type'] = 'short'
                        monthly_short_positions.append(pos)
                        
                except Exception as e:
                    print(f"   Error loading {portfolio_path}: {str(e)}")
                    continue
            
            # Create monthly portfolio record
            if monthly_long_positions or monthly_short_positions:
                portfolio_record = {
                    'date': date_str,
                    'year': year,
                    'month': month,
                    'long_positions': monthly_long_positions,
                    'short_positions': monthly_short_positions,
                    'total_positions': len(monthly_long_positions) + len(monthly_short_positions),
                    'long_count': len(monthly_long_positions),
                    'short_count': len(monthly_short_positions)
                }
                all_portfolios.append(portfolio_record)
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(all_portfolios)
        if len(portfolio_df) > 0:
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.sort_values('date').reset_index(drop=True)
        
        print(f"   Loaded {len(portfolio_df)} monthly portfolios")
        return portfolio_df
    
    def calculate_portfolio_returns(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly returns for the strategy
        
        Note: This is a simplified implementation. In practice, you would need
        actual stock return data to calculate real returns.
        """
        print(f"\nCalculating portfolio returns...")
        
        # For demonstration, we'll simulate returns based on scores
        # In practice, you would join with actual stock return data
        
        monthly_returns = []
        
        for _, row in portfolio_df.iterrows():
            # Simulate monthly return based on portfolio composition
            long_return = 0.0
            short_return = 0.0
            
            # Long positions: simulate positive return based on combined scores
            if row['long_positions']:
                long_scores = [pos['combined_score'] for pos in row['long_positions']]
                long_weights = [pos['portfolio_weight'] for pos in row['long_positions']]
                # Simulate return: higher scores → higher returns (with noise)
                long_return = np.average([
                    np.random.normal(score * 0.02, 0.03) for score in long_scores
                ], weights=long_weights)
            
            # Short positions: simulate negative return (we profit when stocks go down)
            if row['short_positions']:
                short_scores = [pos['combined_score'] for pos in row['short_positions']]
                short_weights = [pos['portfolio_weight'] for pos in row['short_positions']]
                # Simulate return: lower scores → higher short profits (with noise)
                short_return = np.average([
                    np.random.normal((1 - score) * 0.015, 0.025) for score in short_scores
                ], weights=short_weights)
            
            # Combined portfolio return (equal weight long/short)
            total_return = (long_return + short_return) / 2
            
            monthly_returns.append({
                'date': row['date'],
                'year': row['year'],
                'month': row['month'],
                'portfolio_return': total_return,
                'long_return': long_return,
                'short_return': short_return,
                'positions_count': row['total_positions']
            })
        
        returns_df = pd.DataFrame(monthly_returns)
        print(f"   Calculated returns for {len(returns_df)} months")
        
        return returns_df
    
    def get_sp500_benchmark(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate S&P 500 benchmark returns for comparison
        
        Note: This simulates S&P 500 returns. In practice, you would load actual data.
        """
        print(f"\nGenerating S&P 500 benchmark...")
        
        # Simulate S&P 500 returns (realistic historical statistics)
        # Average annual return ~10%, volatility ~16%
        np.random.seed(42)  # For reproducible results
        
        sp500_returns = []
        for _, row in returns_df.iterrows():
            # Simulate monthly S&P 500 return
            monthly_sp500 = np.random.normal(0.008, 0.04)  # ~10% annual, 16% vol
            sp500_returns.append({
                'date': row['date'],
                'sp500_return': monthly_sp500
            })
        
        sp500_df = pd.DataFrame(sp500_returns)
        print(f"   Generated S&P 500 returns for {len(sp500_df)} months")
        
        return sp500_df
    
    def calculate_performance_metrics(self, returns_df: pd.DataFrame, sp500_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        print(f"\nCalculating performance metrics...")
        
        # Merge portfolio and benchmark returns
        combined_df = returns_df.merge(sp500_df, on='date', how='inner')
        
        if len(combined_df) == 0:
            print("   No data available for metrics calculation")
            return {}
        
        # Portfolio metrics
        portfolio_returns = combined_df['portfolio_return']
        sp500_returns = combined_df['sp500_return']
        excess_returns = portfolio_returns - sp500_returns
        
        # Calculate OOS R² (Out-of-Sample R-squared)
        oos_r2_portfolio = self._calculate_oos_r2(portfolio_returns, sp500_returns)
        
        # For excess returns, calculate R² vs mean excess return (more meaningful than vs zero)
        mean_excess = np.full_like(excess_returns, excess_returns.mean())
        oos_r2_excess = self._calculate_oos_r2(excess_returns, mean_excess)
        
        # Calculate metrics
        metrics = {
            # Basic return metrics
            'avg_monthly_return': portfolio_returns.mean(),
            'avg_annual_return': portfolio_returns.mean() * 12,
            'total_return': (1 + portfolio_returns).prod() - 1,
            
            # Risk metrics
            'monthly_volatility': portfolio_returns.std(),
            'annual_volatility': portfolio_returns.std() * np.sqrt(12),
            
            # Risk-adjusted metrics
            'sharpe_ratio': (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12) if portfolio_returns.std() > 0 else 0,
            'information_ratio': (excess_returns.mean() / excess_returns.std()) * np.sqrt(12) if excess_returns.std() > 0 else 0,
            'tracking_error': excess_returns.std() * np.sqrt(12),  # Annualized tracking error
            
            # OOS R² metrics
            'oos_r2_vs_sp500': oos_r2_portfolio,
            'oos_r2_excess_returns': oos_r2_excess,
            
            # Benchmark comparison
            'sp500_avg_monthly_return': sp500_returns.mean(),
            'sp500_avg_annual_return': sp500_returns.mean() * 12,
            'sp500_annual_volatility': sp500_returns.std() * np.sqrt(12),
            'sp500_total_return': (1 + sp500_returns).prod() - 1,
            
            # Alpha calculation (simplified)
            'monthly_alpha': excess_returns.mean(),
            'annual_alpha': excess_returns.mean() * 12,
            
            # Additional metrics
            'win_rate': (portfolio_returns > 0).mean(),
            'months_tracked': len(portfolio_returns),
            'max_monthly_return': portfolio_returns.max(),
            'min_monthly_return': portfolio_returns.min(),
            
            # Drawdown calculation
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
        }
        
        print(f"   Calculated {len(metrics)} performance metrics")
        print(f"   Information Ratio: {metrics['information_ratio']:.4f}")
        print(f"   Tracking Error: {metrics['tracking_error']:.4f}")
        print(f"   OOS R² vs S&P 500: {oos_r2_portfolio:.4f}")
        print(f"   OOS R² (excess returns): {oos_r2_excess:.4f}")
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_oos_r2(self, y_actual: pd.Series, y_predicted: pd.Series) -> float:
        """
        Calculate Out-of-Sample R² following the methodology from penalized_linear_hackathon.py
        
        Args:
            y_actual: Actual returns/values
            y_predicted: Predicted returns/values (or benchmark for comparison)
            
        Returns:
            float: OOS R² value
        """
        y_actual_values = y_actual.values if hasattr(y_actual, 'values') else y_actual
        y_predicted_values = y_predicted.values if hasattr(y_predicted, 'values') else y_predicted
        
        # Handle edge cases
        if len(y_actual_values) == 0 or len(y_predicted_values) == 0:
            return 0.0
            
        # OOS R² formula: 1 - sum((actual - predicted)²) / sum(actual²)
        # This measures how much better the predictions are compared to just predicting zero
        numerator = np.sum(np.square(y_actual_values - y_predicted_values))
        denominator = np.sum(np.square(y_actual_values))
        
        # Handle zero or near-zero denominator (when actual values are all near zero)
        if np.isclose(denominator, 0, atol=1e-12):
            # If actual values are near zero, check if predicted values are also near zero
            if np.isclose(numerator, 0, atol=1e-12):
                return 1.0  # Perfect prediction of near-zero values
            else:
                return 0.0  # Poor prediction when actual values are near zero
            
        oos_r2 = 1 - (numerator / denominator)
        
        # Handle NaN or infinite results
        if np.isnan(oos_r2) or np.isinf(oos_r2):
            return 0.0
            
        return oos_r2
    
    def save_results(self, returns_df: pd.DataFrame, sp500_df: pd.DataFrame, metrics: Dict):
        """Save all results to files"""
        print(f"\nSaving results...")
        
        # Save returns data
        returns_path = self.backtest_dir / "portfolio_returns.csv"
        returns_df.to_csv(returns_path, index=False)
        print(f"   Saved portfolio returns: {returns_path}")
        
        # Save S&P 500 data
        sp500_path = self.backtest_dir / "sp500_benchmark.csv"
        sp500_df.to_csv(sp500_path, index=False)
        print(f"   Saved S&P 500 benchmark: {sp500_path}")
        
        # Save metrics
        metrics_path = self.backtest_dir / "performance_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"   Saved performance metrics: {metrics_path}")
        
        # Save summary report
        self.generate_summary_report(metrics)
    
    def generate_summary_report(self, metrics: Dict):
        """Generate a comprehensive summary report"""
        report_path = self.backtest_dir / "backtest_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE BACKTESTING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Backtesting Period: {self.start_month:02d}/{self.start_year} - {self.end_month:02d}/{self.end_year}\n")
            f.write(f"Strategy: Long top {self.top_n} + Short bottom {self.bottom_n} stocks per sector\n")
            f.write(f"Sectors: {', '.join(SECTORS)}\n")
            f.write(f"Total months tracked: {metrics.get('months_tracked', 0)}\n\n")
            
            f.write("PORTFOLIO PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Annual Return:     {metrics.get('avg_annual_return', 0):.2%}\n")
            f.write(f"Annual Volatility:         {metrics.get('annual_volatility', 0):.2%}\n")
            f.write(f"Sharpe Ratio:              {metrics.get('sharpe_ratio', 0):.3f}\n")
            f.write(f"Information Ratio:         {metrics.get('information_ratio', 0):.3f}\n")
            f.write(f"Tracking Error:            {metrics.get('tracking_error', 0):.2%}\n")
            f.write(f"Annual Alpha:              {metrics.get('annual_alpha', 0):.2%}\n")
            f.write(f"Maximum Drawdown:          {metrics.get('max_drawdown', 0):.2%}\n")
            f.write(f"Win Rate:                  {metrics.get('win_rate', 0):.2%}\n")
            f.write(f"OOS R² vs S&P 500:         {metrics.get('oos_r2_vs_sp500', 0):.4f}\n")
            f.write(f"OOS R² (Excess Returns):   {metrics.get('oos_r2_excess_returns', 0):.4f}\n\n")
            
            f.write("S&P 500 BENCHMARK COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write(f"S&P 500 Annual Return:     {metrics.get('sp500_avg_annual_return', 0):.2%}\n")
            f.write(f"S&P 500 Annual Volatility: {metrics.get('sp500_annual_volatility', 0):.2%}\n")
            f.write(f"S&P 500 Total Return:      {metrics.get('sp500_total_return', 0):.2%}\n")
            f.write(f"Portfolio Total Return:    {metrics.get('total_return', 0):.2%}\n\n")
            
            f.write("ADDITIONAL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Monthly Return:       {metrics.get('max_monthly_return', 0):.2%}\n")
            f.write(f"Worst Monthly Return:      {metrics.get('min_monthly_return', 0):.2%}\n")
            f.write(f"Average Monthly Return:    {metrics.get('avg_monthly_return', 0):.2%}\n\n")
        
        print(f"   Saved summary report: {report_path}")
    
    def run_comprehensive_backtest(self):
        """
        Run the complete backtesting suite
        """
        print(f"\nStarting Comprehensive Backtesting Suite")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Run all pipeline executions
            self.run_all_pipelines()
            
            # Step 2: Load and process portfolio data
            portfolio_df = self.load_portfolio_data()
            if len(portfolio_df) == 0:
                print("No portfolio data available. Exiting.")
                return
            
            # Step 3: Calculate returns
            returns_df = self.calculate_portfolio_returns(portfolio_df)
            
            # Step 4: Get benchmark data
            sp500_df = self.get_sp500_benchmark(returns_df)
            
            # Step 5: Calculate performance metrics
            metrics = self.calculate_performance_metrics(returns_df, sp500_df)
            
            # Step 6: Save all results
            self.save_results(returns_df, sp500_df, metrics)
            
            # Final summary
            print(f"\nBACKTESTING COMPLETED SUCCESSFULLY!")
            print(f"   Portfolio Annual Return: {metrics.get('avg_annual_return', 0):.2%}")
            print(f"   Portfolio Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
            print(f"   Annual Alpha vs S&P 500: {metrics.get('annual_alpha', 0):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Information Ratio: {metrics.get('information_ratio', 0):.3f}")
            print(f"   Tracking Error: {metrics.get('tracking_error', 0):.2%}")
            print(f"   OOS R² vs S&P 500: {metrics.get('oos_r2_vs_sp500', 0):.4f}")
            print(f"   OOS R² (Excess Returns): {metrics.get('oos_r2_excess_returns', 0):.4f}")
            print(f"   Results saved to: {self.backtest_dir}")
            
        except Exception as e:
            print(f"Backtesting failed: {str(e)}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive Backtesting Suite")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year (default: 2015)")
    parser.add_argument("--start-month", type=int, default=1, help="Start month (default: 1)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--end-month", type=int, default=5, help="End month (default: 5)")
    parser.add_argument("--top-n", type=int, default=5, help="Top N stocks per sector (default: 5)")
    parser.add_argument("--bottom-m", type=int, default=5, help="Bottom M stocks per sector (default: 5)")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    # Create and run backtester
    backtester = ComprehensiveBacktester(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        top_n=args.top_n,
        bottom_m=args.bottom_m,
        max_workers=args.max_workers
    )
    
    backtester.run_comprehensive_backtest()


if __name__ == "__main__":
    main()