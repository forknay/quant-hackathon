"""
Verify OHLCV data quality and generate individual CSV files for each company
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def verify_ohlcv_quality(ohlcv_data, dates, company_id, returns=None):
    """
    Comprehensive verification of OHLCV data quality.
    
    Returns:
        dict: Quality metrics and validation results
    """
    
    opens = ohlcv_data[:, 0]
    highs = ohlcv_data[:, 1]
    lows = ohlcv_data[:, 2]
    closes = ohlcv_data[:, 3]
    volumes = ohlcv_data[:, 4]
    
    quality_report = {
        'company_id': company_id,
        'total_days': len(ohlcv_data),
        'date_range': f"{dates[0]} to {dates[-1]}",
        'violations': [],
        'warnings': [],
        'statistics': {}
    }
    
    # 1. OHLC Relationship Validation
    ohlc_violations = 0
    for i in range(len(ohlcv_data)):
        # Check: Low <= Open <= High and Low <= Close <= High
        if not (lows[i] <= opens[i] <= highs[i]):
            ohlc_violations += 1
            quality_report['violations'].append(f"Day {i}: Low({lows[i]:.2f}) <= Open({opens[i]:.2f}) <= High({highs[i]:.2f})")
        
        if not (lows[i] <= closes[i] <= highs[i]):
            ohlc_violations += 1
            quality_report['violations'].append(f"Day {i}: Low({lows[i]:.2f}) <= Close({closes[i]:.2f}) <= High({highs[i]:.2f})")
    
    quality_report['ohlc_violations'] = ohlc_violations
    
    # 2. Price Range Validation
    price_stats = {
        'min_price': np.min([opens.min(), highs.min(), lows.min(), closes.min()]),
        'max_price': np.max([opens.max(), highs.max(), lows.max(), closes.max()]),
        'mean_close': closes.mean(),
        'close_volatility': closes.std()
    }
    
    # Check for reasonable price ranges
    if price_stats['min_price'] < 0:
        quality_report['warnings'].append(f"Negative prices detected: min={price_stats['min_price']:.2f}")
    
    if price_stats['max_price'] > 10000:
        quality_report['warnings'].append(f"Extremely high prices: max=${price_stats['max_price']:.2f}")
    
    if price_stats['min_price'] < 1:
        quality_report['warnings'].append(f"Very low prices: min=${price_stats['min_price']:.2f}")
    
    # 3. Volume Validation
    volume_stats = {
        'min_volume': volumes.min(),
        'max_volume': volumes.max(),
        'mean_volume': volumes.mean(),
        'zero_volume_days': np.sum(volumes == 0)
    }
    
    if volume_stats['min_volume'] < 0:
        quality_report['violations'].append(f"Negative volumes detected")
    
    if volume_stats['zero_volume_days'] > len(volumes) * 0.1:  # More than 10% zero volume
        quality_report['warnings'].append(f"High zero-volume days: {volume_stats['zero_volume_days']}")
    
    # 4. Return Analysis
    if len(closes) > 1:
        calculated_returns = np.diff(closes) / closes[:-1]
        return_stats = {
            'mean_return': calculated_returns.mean(),
            'return_volatility': calculated_returns.std(),
            'max_daily_return': calculated_returns.max(),
            'min_daily_return': calculated_returns.min(),
            'extreme_returns': np.sum(np.abs(calculated_returns) > 0.5)  # >50% daily moves
        }
        
        if return_stats['extreme_returns'] > len(calculated_returns) * 0.05:  # More than 5% extreme
            quality_report['warnings'].append(f"High extreme returns: {return_stats['extreme_returns']}")
        
        # Compare with original returns if available
        if returns is not None and len(returns) > 1:
            original_vol = np.std(returns)
            calculated_vol = return_stats['return_volatility']
            vol_preservation = calculated_vol / original_vol if original_vol > 0 else 0
            return_stats['volatility_preservation'] = vol_preservation
            
            if vol_preservation < 0.01 or vol_preservation > 100:
                quality_report['warnings'].append(f"Poor volatility preservation: {vol_preservation:.3f}")
    else:
        return_stats = {'note': 'Insufficient data for return analysis'}
    
    # 5. Daily Range Analysis
    daily_ranges = (highs - lows) / closes
    range_stats = {
        'mean_daily_range': daily_ranges.mean(),
        'max_daily_range': daily_ranges.max(),
        'days_no_range': np.sum(daily_ranges == 0)
    }
    
    if range_stats['days_no_range'] > len(daily_ranges) * 0.2:  # More than 20% no range
        quality_report['warnings'].append(f"Too many days with no range: {range_stats['days_no_range']}")
    
    # Compile all statistics
    quality_report['statistics'] = {
        'prices': price_stats,
        'volumes': volume_stats,
        'returns': return_stats,
        'ranges': range_stats
    }
    
    # Overall quality score
    violation_penalty = ohlc_violations * 10
    warning_penalty = len(quality_report['warnings']) * 5
    quality_score = max(0, 100 - violation_penalty - warning_penalty)
    quality_report['quality_score'] = quality_score
    
    return quality_report

def create_realistic_ohlcv_with_dates(stock_data):
    """
    Create realistic OHLCV data with improved methodology.
    
    Returns:
        tuple: (ohlcv_array, dates_array, quality_report)
    """
    
    # Extract data
    dates = stock_data['date'].values
    log_prices = stock_data['prc'].values
    returns = stock_data['stock_ret'].values
    company_id = stock_data['id'].iloc[0]
    
    # Convert log prices to realistic price levels
    # Use a more conservative scaling to avoid extreme prices
    base_price = 50.0
    price_scale = 0.2  # Reduced from 0.3 to avoid extreme values
    close_prices = base_price * np.exp(log_prices * price_scale)
    
    num_days = len(close_prices)
    open_prices = np.zeros_like(close_prices)
    high_prices = np.zeros_like(close_prices)
    low_prices = np.zeros_like(close_prices)
    
    for i in range(num_days):
        current_close = close_prices[i]
        current_return = returns[i] if not np.isnan(returns[i]) else 0.0
        
        # Generate Open with more realistic gaps
        if i == 0:
            # First day: small random gap
            gap = np.random.normal(0, 0.001)
            open_prices[i] = current_close * (1 + gap)
        else:
            # Subsequent days: use a smaller portion of return as overnight gap
            overnight_factor = 0.1  # Only 10% of return happens overnight
            overnight_gap = current_return * overnight_factor
            # Add small random component
            random_gap = np.random.normal(0, 0.002)
            gap_total = overnight_gap + random_gap
            # Limit gap to reasonable range
            gap_total = np.clip(gap_total, -0.05, 0.05)  # Max 5% gap
            open_prices[i] = close_prices[i-1] * (1 + gap_total)
        
        # Generate High and Low with realistic intraday movement
        # Use a smaller portion of return for intraday volatility
        intraday_return = current_return * 0.6  # 60% of return is intraday
        base_volatility = 0.01  # 1% base daily volatility
        intraday_vol = max(abs(intraday_return) * 0.3, base_volatility)
        
        # Limit volatility to reasonable range
        intraday_vol = min(intraday_vol, 0.1)  # Max 10% intraday range
        
        # Generate High and Low
        if intraday_return >= 0:  # Up day
            high_multiplier = 1 + intraday_vol
            low_multiplier = 1 - intraday_vol * 0.3  # Smaller downside on up days
        else:  # Down day
            high_multiplier = 1 + intraday_vol * 0.3  # Smaller upside on down days
            low_multiplier = 1 - intraday_vol
        
        high_prices[i] = max(open_prices[i], current_close) * high_multiplier
        low_prices[i] = min(open_prices[i], current_close) * low_multiplier
        
        # Ensure OHLC constraints are maintained
        high_prices[i] = max(high_prices[i], open_prices[i], current_close)
        low_prices[i] = min(low_prices[i], open_prices[i], current_close)
        
        # Additional safety check
        if low_prices[i] > high_prices[i]:
            low_prices[i], high_prices[i] = high_prices[i], low_prices[i]
    
    # Generate Volume with better methodology
    if 'turnover_126d' in stock_data.columns:
        turnover_norm = stock_data['turnover_126d'].values
        # More conservative volume scaling
        base_volume = 200000  # 200K shares base
        volume_scale = 0.2  # Reduced scaling
        volumes = base_volume * np.exp(turnover_norm * volume_scale)
        volumes = np.clip(volumes, 50000, 5000000)  # Reasonable bounds
        
        # Volume should increase with absolute returns
        volume_multiplier = 1 + np.abs(returns) * 0.1  # 10% increase per unit return
        volumes = volumes * volume_multiplier
    else:
        # Fallback volume calculation
        base_volume = 200000
        volume_multiplier = 1 + np.abs(returns) * 0.15
        volumes = base_volume * volume_multiplier
    
    volumes = volumes.astype(int)
    
    # Create OHLCV array
    ohlcv_data = np.column_stack([open_prices, high_prices, low_prices, close_prices, volumes])
    
    # Verify quality
    quality_report = verify_ohlcv_quality(ohlcv_data, dates, company_id, returns)
    
    return ohlcv_data, dates, quality_report

def generate_company_ohlcv_files():
    """
    Generate individual OHLCV CSV files for each company.
    """
    
    print("🏭 GENERATING INDIVIDUAL COMPANY OHLCV FILES")
    print("="*70)
    
    # Load data
    print("📂 Loading cleaned_all.parquet...")
    df = pd.read_parquet('../cleaned_all.parquet')
    
    print(f"📊 Dataset overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique companies: {df['id'].nunique():,}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create output directory
    output_dir = Path("company_ohlcv_data")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Get unique companies
    companies = df['id'].unique()
    
    # Quality tracking
    quality_summary = []
    successful_companies = 0
    failed_companies = 0
    total_violations = 0
    total_warnings = 0
    
    print(f"\n🔄 Processing {len(companies)} companies...")
    
    for i, company_id in enumerate(companies):
        try:
            # Get company data
            company_data = df[df['id'] == company_id].sort_values('date').copy()
            
            # Skip companies with too little data
            if len(company_data) < 5:
                print(f"⚠️  {i+1:4d}/{len(companies)} | {company_id:<20} | Skipped: Only {len(company_data)} days")
                failed_companies += 1
                continue
            
            # Generate OHLCV data
            ohlcv_data, dates, quality_report = create_realistic_ohlcv_with_dates(company_data)
            
            # Create DataFrame for CSV
            ohlcv_df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Open': ohlcv_data[:, 0],
                'High': ohlcv_data[:, 1],
                'Low': ohlcv_data[:, 2],
                'Close': ohlcv_data[:, 3],
                'Volume': ohlcv_data[:, 4].astype(int)
            })
            
            # Round prices to 2 decimal places
            for col in ['Open', 'High', 'Low', 'Close']:
                ohlcv_df[col] = ohlcv_df[col].round(2)
            
            # Save to CSV
            csv_filename = f"{company_id}_ohlcv.csv"
            csv_path = output_dir / csv_filename
            ohlcv_df.to_csv(csv_path, index=False)
            
            # Track quality
            quality_summary.append(quality_report)
            total_violations += quality_report['ohlc_violations']
            total_warnings += len(quality_report['warnings'])
            successful_companies += 1
            
            # Progress update
            if quality_report['quality_score'] >= 90:
                status = "✅ EXCELLENT"
            elif quality_report['quality_score'] >= 70:
                status = "⚠️  GOOD"
            else:
                status = "❌ POOR"
            
            if i % 100 == 0 or i < 20:  # Show first 20 and every 100th
                print(f"   {i+1:4d}/{len(companies)} | {company_id:<20} | {len(company_data):3d} days | Score: {quality_report['quality_score']:3.0f} | {status}")
        
        except Exception as e:
            print(f"❌ {i+1:4d}/{len(companies)} | {company_id:<20} | Error: {str(e)}")
            failed_companies += 1
            continue
    
    # Final summary
    print(f"\n📊 GENERATION SUMMARY")
    print("="*50)
    print(f"✅ Successful companies: {successful_companies:,}")
    print(f"❌ Failed companies: {failed_companies:,}")
    print(f"📁 CSV files created: {len(list(output_dir.glob('*.csv'))):,}")
    print(f"🔍 Total OHLC violations: {total_violations:,}")
    print(f"⚠️  Total warnings: {total_warnings:,}")
    
    # Quality distribution
    if quality_summary:
        scores = [q['quality_score'] for q in quality_summary]
        print(f"\n📈 QUALITY DISTRIBUTION")
        print(f"   Mean quality score: {np.mean(scores):.1f}")
        print(f"   Excellent (90+): {sum(1 for s in scores if s >= 90):,} ({sum(1 for s in scores if s >= 90)/len(scores)*100:.1f}%)")
        print(f"   Good (70-89): {sum(1 for s in scores if 70 <= s < 90):,} ({sum(1 for s in scores if 70 <= s < 90)/len(scores)*100:.1f}%)")
        print(f"   Poor (<70): {sum(1 for s in scores if s < 70):,} ({sum(1 for s in scores if s < 70)/len(scores)*100:.1f}%)")
    
    return quality_summary, output_dir

def validate_sample_files(output_dir):
    """Validate a sample of generated files."""
    
    print(f"\n🔍 VALIDATING SAMPLE FILES")
    print("="*40)
    
    csv_files = list(output_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    # Sample a few files for detailed validation
    sample_files = csv_files[:5]
    
    for csv_file in sample_files:
        print(f"\n📄 Validating: {csv_file.name}")
        
        try:
            # Load the CSV
            df = pd.read_csv(csv_file)
            
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Check data types
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Validate OHLC relationships
            violations = 0
            for i in range(len(df)):
                if not (df.loc[i, 'Low'] <= df.loc[i, 'Open'] <= df.loc[i, 'High']):
                    violations += 1
                if not (df.loc[i, 'Low'] <= df.loc[i, 'Close'] <= df.loc[i, 'High']):
                    violations += 1
            
            # Show sample data
            print(f"   OHLC violations: {violations}")
            print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            print(f"   Volume range: {df['Volume'].min():,} - {df['Volume'].max():,}")
            
            print("   Sample data:")
            print(df.head(3).to_string(index=False))
            
        except Exception as e:
            print(f"   ❌ Error validating {csv_file.name}: {e}")

def create_quality_report(quality_summary, output_dir):
    """Create a comprehensive quality report."""
    
    print(f"\n📊 CREATING QUALITY REPORT")
    print("="*40)
    
    if not quality_summary:
        print("❌ No quality data available")
        return
    
    # Create detailed quality report
    quality_df = pd.DataFrame([
        {
            'Company_ID': q['company_id'],
            'Total_Days': q['total_days'],
            'Date_Range': q['date_range'],
            'Quality_Score': q['quality_score'],
            'OHLC_Violations': q['ohlc_violations'],
            'Warnings': len(q['warnings']),
            'Min_Price': q['statistics']['prices']['min_price'],
            'Max_Price': q['statistics']['prices']['max_price'],
            'Mean_Close': q['statistics']['prices']['mean_close'],
            'Mean_Volume': q['statistics']['volumes']['mean_volume'],
            'Zero_Volume_Days': q['statistics']['volumes']['zero_volume_days']
        }
        for q in quality_summary
    ])
    
    # Save quality report
    quality_report_path = output_dir / "quality_report.csv"
    quality_df.to_csv(quality_report_path, index=False)
    
    print(f"✅ Quality report saved: {quality_report_path}")
    print(f"📊 Report contains {len(quality_df)} companies")
    
    # Show summary statistics
    print(f"\n📈 QUALITY STATISTICS:")
    print(f"   Companies with perfect score (100): {sum(quality_df['Quality_Score'] == 100):,}")
    print(f"   Companies with excellent score (90+): {sum(quality_df['Quality_Score'] >= 90):,}")
    print(f"   Companies with violations: {sum(quality_df['OHLC_Violations'] > 0):,}")
    print(f"   Companies with warnings: {sum(quality_df['Warnings'] > 0):,}")

if __name__ == "__main__":
    # Generate OHLCV files for all companies
    quality_summary, output_dir = generate_company_ohlcv_files()
    
    # Validate sample files
    validate_sample_files(output_dir)
    
    # Create quality report
    create_quality_report(quality_summary, output_dir)
    
    print(f"\n🎉 OHLCV GENERATION COMPLETE!")
    print(f"📁 All files saved in: {output_dir.absolute()}")
    print(f"📄 Individual CSV files: {len(list(output_dir.glob('*.csv')))} companies")
    print(f"📊 Quality report: quality_report.csv")