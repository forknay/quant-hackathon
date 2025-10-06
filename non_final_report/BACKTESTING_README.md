# COMPREHENSIVE BACKTESTING SUITE

A complete backtesting framework for quantitative trading strategies that runs the full pipeline across all 11 GICS sectors and time periods, providing detailed performance analysis versus the S&P 500.

## 🚀 Quick Start

### Option 1: Quick Demo (Recommended First)
Test the system with a 6-month period:
```bash
python backtest_quick_demo.py
```

### Option 2: Full Backtest (Default: Jan 2015 - May 2025)
```bash
python run_complete_backtest.py
```

### Option 3: Custom Period
```bash
python run_complete_backtest.py --start-year 2018 --start-month 1 --end-year 2023 --end-month 12 --top-n 3 --bottom-n 3
```

## 📊 What It Does

### Strategy Overview
- **Sectors**: All 11 GICS sectors (Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Healthcare, Financials, Information Technology, Telecommunications, Utilities, Real Estate)
- **Portfolio Construction**: Long top N stocks + Short bottom N stocks per sector (default: 5 each)
- **Rebalancing**: Monthly using `main_pipeline.py`
- **Signal Generation**: Combined algorithmic + ML scores

### Analysis Pipeline

1. **Portfolio Construction**
   - Runs `main_pipeline.py` for each sector/month combination
   - Selects top/bottom stocks based on combined algo + ML scores
   - Constructs monthly portfolios with sector diversification

2. **Performance Calculation**
   - Calculates monthly portfolio returns
   - Fetches real S&P 500 benchmark data (via yfinance)
   - Computes comprehensive risk-adjusted metrics

3. **Statistical Analysis**
   - Alpha/Beta calculation using linear regression
   - Sharpe ratio, Sortino ratio, Information ratio
   - Maximum drawdown, Value at Risk (VaR)
   - Statistical significance testing

## 📈 Key Metrics Calculated

### Return Metrics
- **Average Annualized Portfolio Returns**
- **Total Return** over the entire period  
- **Monthly Return Statistics** (mean, best, worst)

### Risk Metrics
- **Annualized Portfolio Standard Deviation**
- **Maximum Drawdown** and drawdown duration
- **Downside Volatility** (Sortino ratio calculation)
- **Value at Risk (95%, 99%)**

### Risk-Adjusted Performance
- **Annualized Alpha** (market risk-adjusted return)
- **Beta** (market sensitivity)
- **Sharpe Ratio** (risk-adjusted return vs risk-free rate)
- **Sortino Ratio** (downside risk-adjusted return)
- **Information Ratio** (excess return vs tracking error)

### Benchmark Comparison
- **S&P 500 Returns** and volatility
- **Excess Returns** (portfolio - benchmark)
- **Win Rate** (% months outperforming S&P 500)
- **Tracking Error**

## 🛠️ Components

### Core Scripts

1. **`comprehensive_backtesting_suite.py`**
   - Main backtesting engine
   - Runs pipeline across all sectors/periods
   - Parallel execution with configurable workers
   - Portfolio construction and basic return calculation

2. **`enhanced_portfolio_analysis.py`**
   - Advanced performance analysis
   - Real market data integration
   - Statistical significance testing
   - Comprehensive metric calculation

3. **`run_complete_backtest.py`**
   - Master orchestrator script
   - Combines backtesting + analysis
   - Generates executive summary
   - Error handling and progress reporting

4. **`backtest_quick_demo.py`**
   - Quick testing script (6 months)
   - Perfect for initial validation
   - Fast execution for development

### Usage Examples

```bash
# Quick 6-month test
python backtest_quick_demo.py

# Full default backtest (Jan 2015 - May 2025)
python run_complete_backtest.py

# Custom period with different stock counts
python run_complete_backtest.py --start-year 2020 --start-month 1 --end-year 2024 --end-month 12 --top-n 3 --bottom-n 3

# Parallel execution with more workers
python run_complete_backtest.py --max-workers 8

# Analysis only (skip backtesting)
python run_complete_backtest.py --skip-backtest
```

## 📁 Output Files

All results are saved to `results/backtesting/`:

### Data Files
- **`portfolio_returns.csv`** - Monthly portfolio returns data
- **`sp500_benchmark.csv`** - S&P 500 benchmark returns
- **`performance_metrics.json`** - All calculated metrics
- **`enhanced_metrics.json`** - Detailed metrics with metadata

### Reports
- **`EXECUTIVE_SUMMARY.md`** - High-level performance summary with grades
- **`enhanced_performance_report.txt`** - Detailed technical analysis
- **`backtest_summary_report.txt`** - Basic backtesting results

### Individual Portfolio Files
- **`portfolio_YYYY_MM_sector.json`** - Monthly portfolio compositions

## 🔧 Configuration Options

### Command Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--start-year` | 2015 | Backtesting start year |
| `--start-month` | 1 | Backtesting start month |
| `--end-year` | 2025 | Backtesting end year |
| `--end-month` | 5 | Backtesting end month |
| `--top-n` | 5 | Top stocks per sector (long positions) |
| `--bottom-n` | 5 | Bottom stocks per sector (short positions) |
| `--max-workers` | 4 | Parallel execution workers |
| `--skip-backtest` | False | Skip backtesting, run analysis only |

### Advanced Configuration

Edit the scripts directly for:
- **Transaction costs** (default: 0.1%)
- **Risk-free rate** (default: 2%)
- **Timeout settings** (default: 5 min per pipeline run)
- **Batch sizes** for parallel execution

## 📊 Performance Grading System

The executive summary includes an automated grading system:

- **Grade A**: Excellent performance (excess return > 2%, statistically significant alpha)
- **Grade B**: Good performance (positive excess return, Sharpe > 1.0)
- **Grade C**: Satisfactory performance (positive excess return)
- **Grade D**: Needs improvement (negative excess return)

## 🚨 Prerequisites

### Required Data
1. **Sector Results**: All 11 sectors processed in `algo/results/`
   ```bash
   # Run this first if not done:
   python algo/run_sector.py --sector [sector_name]
   ```

2. **Python Packages**: Install requirements
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
- `yfinance` - For S&P 500 benchmark data
- `statsmodels` - For alpha/beta regression analysis
- `pandas`, `numpy` - Data processing
- `scipy` - Statistical calculations

## 🔍 Troubleshooting

### Common Issues

1. **"No portfolio data available"**
   - Ensure all sector processing is complete in `algo/results/`
   - Check that `main_pipeline.py` is working correctly

2. **"S&P 500 data fetch failed"**
   - System will automatically use simulated data
   - Check internet connection for real data

3. **Pipeline timeouts**
   - Increase timeout in `comprehensive_backtesting_suite.py`
   - Reduce `--max-workers` for stability

4. **Memory issues**
   - Reduce batch sizes in the backtesting suite
   - Process fewer periods at once

### Performance Tips

- **Start with quick demo** to validate setup
- **Use parallel workers** (`--max-workers`) based on CPU cores
- **Monitor disk space** - generates many portfolio files
- **Check existing files** - system skips already processed periods

## 📖 Understanding Results

### Key Metrics to Focus On

1. **Annualized Alpha**: Risk-adjusted excess return vs market
2. **Sharpe Ratio**: Risk-adjusted return vs risk-free rate  
3. **Information Ratio**: Consistency of outperformance
4. **Maximum Drawdown**: Worst peak-to-trough decline
5. **Statistical Significance**: p-value for alpha

### Interpretation Guidelines

- **Alpha > 2%**: Strong outperformance
- **Sharpe > 1.0**: Good risk-adjusted returns
- **Max Drawdown < -20%**: Acceptable risk level
- **p-value < 0.05**: Statistically significant alpha

## 🎯 Next Steps

After running the backtesting:

1. **Review Executive Summary** for key findings
2. **Analyze sector attribution** in detailed reports
3. **Optimize parameters** based on results
4. **Test different periods** for robustness
5. **Consider implementation** for live trading

---

**⚠️ Important Notes:**
- Results are based on simulated returns - validate with actual market data before live trading
- Past performance does not guarantee future results
- Consider transaction costs and market impact in live implementation
- Regular rebalancing may have significant tax implications

**📧 Support:**
Review the generated reports and logs for detailed debugging information.