#!/usr/bin/env python3
"""
Fill Missing Portfolios Script

This script identifies missing portfolio files in the results directory for the period 
01/2015 to 05/2025 and creates them by copying the holdings from the previous existing 
portfolio of the same sector.

The script assumes monthly portfolios with the naming pattern:
portfolio_{YYYY}_{MM}_{sector}.json
"""

import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Configuration
RESULTS_DIR = "results"
START_YEAR = 2015
START_MONTH = 1
END_YEAR = 2025
END_MONTH = 5

SECTORS = [
    "energy", "materials", "industrials", "cons_discretionary", 
    "cons_staples", "healthcare", "financials", "it", 
    "telecoms", "utilities", "re"
]

def get_portfolio_filename(year, month, sector):
    """Generate portfolio filename for given year, month, and sector."""
    return f"portfolio_{year}_{month:02d}_{sector}.json"

def get_all_expected_portfolios():
    """Generate list of all expected portfolio files from START to END dates."""
    expected = []
    
    current_year = START_YEAR
    current_month = START_MONTH
    
    while (current_year < END_YEAR) or (current_year == END_YEAR and current_month <= END_MONTH):
        for sector in SECTORS:
            filename = get_portfolio_filename(current_year, current_month, sector)
            expected.append({
                'filename': filename,
                'year': current_year,
                'month': current_month,
                'sector': sector,
                'path': os.path.join(RESULTS_DIR, filename)
            })
        
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    return expected

def find_existing_portfolios():
    """Find all existing portfolio files in the results directory."""
    existing = []
    results_path = Path(RESULTS_DIR)
    
    if not results_path.exists():
        print(f"❌ Results directory '{RESULTS_DIR}' not found!")
        return existing
    
    portfolio_pattern = re.compile(r'portfolio_(\d{4})_(\d{2})_([a-z_]+)\.json')
    
    for file_path in results_path.glob("portfolio_*.json"):
        match = portfolio_pattern.match(file_path.name)
        if match:
            year, month, sector = match.groups()
            existing.append({
                'filename': file_path.name,
                'year': int(year),
                'month': int(month),
                'sector': sector,
                'path': str(file_path)
            })
    
    return sorted(existing, key=lambda x: (x['year'], x['month'], x['sector']))

def find_missing_portfolios():
    """Identify missing portfolios by comparing expected vs existing."""
    expected = get_all_expected_portfolios()
    existing = find_existing_portfolios()
    
    existing_set = {(p['year'], p['month'], p['sector']) for p in existing}
    
    missing = []
    for portfolio in expected:
        key = (portfolio['year'], portfolio['month'], portfolio['sector'])
        if key not in existing_set:
            missing.append(portfolio)
    
    return missing, existing

def find_previous_portfolio(sector, target_year, target_month, existing_portfolios):
    """Find the most recent existing portfolio for the given sector before target date."""
    sector_portfolios = [p for p in existing_portfolios if p['sector'] == sector]
    
    # Sort by year and month (descending)
    sector_portfolios.sort(key=lambda x: (x['year'], x['month']), reverse=True)
    
    # Find the most recent portfolio before target date
    for portfolio in sector_portfolios:
        if (portfolio['year'] < target_year) or (portfolio['year'] == target_year and portfolio['month'] < target_month):
            return portfolio
    
    return None

def update_portfolio_metadata(portfolio_data, new_year, new_month):
    """Update portfolio metadata with new date while keeping the same holdings."""
    if isinstance(portfolio_data, dict):
        # Update metadata
        if 'metadata' in portfolio_data:
            portfolio_data['metadata']['year'] = new_year
            portfolio_data['metadata']['month'] = new_month
            portfolio_data['metadata']['date'] = f"{new_year}-{new_month:02d}"
            portfolio_data['metadata']['generated_date'] = datetime.now().isoformat()
            portfolio_data['metadata']['note'] = f"Holdings copied from previous month (forward-fill)"
        
        # Update any date fields in the portfolio
        if 'generation_date' in portfolio_data:
            portfolio_data['generation_date'] = datetime.now().isoformat()
    
    return portfolio_data

def create_missing_portfolio(missing_portfolio, source_portfolio):
    """Create a missing portfolio by copying from source portfolio."""
    try:
        # Read source portfolio
        with open(source_portfolio['path'], 'r') as f:
            portfolio_data = json.load(f)
        
        # Update metadata
        portfolio_data = update_portfolio_metadata(
            portfolio_data, 
            missing_portfolio['year'], 
            missing_portfolio['month']
        )
        
        # Write new portfolio
        with open(missing_portfolio['path'], 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating {missing_portfolio['filename']}: {e}")
        return False

def main():
    """Main function to fill missing portfolios."""
    print("🔍 Portfolio Gap Analysis and Fill Script")
    print("=" * 60)
    
    # Find missing and existing portfolios
    missing_portfolios, existing_portfolios = find_missing_portfolios()
    
    # Summary statistics
    total_expected = len(get_all_expected_portfolios())
    total_existing = len(existing_portfolios)
    total_missing = len(missing_portfolios)
    
    print(f"📊 Portfolio Statistics:")
    print(f"   Expected portfolios: {total_expected:,}")
    print(f"   Existing portfolios: {total_existing:,}")
    print(f"   Missing portfolios: {total_missing:,}")
    print(f"   Coverage: {(total_existing/total_expected)*100:.1f}%")
    print()
    
    if total_missing == 0:
        print("✅ All portfolios are present! No gaps to fill.")
        return
    
    # Group missing portfolios by sector for better reporting
    missing_by_sector = {}
    for portfolio in missing_portfolios:
        sector = portfolio['sector']
        if sector not in missing_by_sector:
            missing_by_sector[sector] = []
        missing_by_sector[sector].append(portfolio)
    
    print("📋 Missing Portfolios by Sector:")
    for sector, portfolios in missing_by_sector.items():
        date_ranges = []
        current_range_start = None
        current_range_end = None
        
        for portfolio in sorted(portfolios, key=lambda x: (x['year'], x['month'])):
            if current_range_start is None:
                current_range_start = f"{portfolio['year']}-{portfolio['month']:02d}"
                current_range_end = current_range_start
            else:
                current_range_end = f"{portfolio['year']}-{portfolio['month']:02d}"
        
        print(f"   {sector:>18}: {len(portfolios):>3} missing ({current_range_start} to {current_range_end})")
    print()
    
    # Fill missing portfolios
    print("🔧 Filling Missing Portfolios...")
    print("-" * 40)
    
    successful_fills = 0
    failed_fills = 0
    
    for i, missing_portfolio in enumerate(missing_portfolios, 1):
        sector = missing_portfolio['sector']
        year = missing_portfolio['year']
        month = missing_portfolio['month']
        
        # Find previous portfolio for this sector
        source_portfolio = find_previous_portfolio(sector, year, month, existing_portfolios)
        
        if source_portfolio is None:
            print(f"⚠️  [{i:>3}/{total_missing}] No previous portfolio found for {sector} {year}-{month:02d}")
            failed_fills += 1
            continue
        
        # Create the missing portfolio
        success = create_missing_portfolio(missing_portfolio, source_portfolio)
        
        if success:
            print(f"✅ [{i:>3}/{total_missing}] Created {missing_portfolio['filename']} (from {source_portfolio['year']}-{source_portfolio['month']:02d})")
            successful_fills += 1
            
            # Add to existing portfolios list for future reference
            existing_portfolios.append(missing_portfolio)
        else:
            failed_fills += 1
    
    print()
    print("📈 Fill Results Summary:")
    print(f"   Successfully filled: {successful_fills:,}")
    print(f"   Failed to fill: {failed_fills:,}")
    print(f"   Total processed: {successful_fills + failed_fills:,}")
    
    if successful_fills > 0:
        new_coverage = ((total_existing + successful_fills) / total_expected) * 100
        print(f"   New coverage: {new_coverage:.1f}%")
    
    print()
    if failed_fills == 0:
        print("🎉 All missing portfolios successfully filled!")
    else:
        print(f"⚠️  {failed_fills} portfolios could not be filled (no previous data available)")

if __name__ == "__main__":
    main()