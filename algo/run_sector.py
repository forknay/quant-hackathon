#!/usr/bin/env python3
"""
Multi-Sector Algorithm Runner

Usage:
    python run_sector.py --sector healthcare
    python run_sector.py --sector energy --test-only
    python run_sector.py --list-sectors
"""

import argparse
import sys
import os
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

def main():
    parser = argparse.ArgumentParser(description='Run sector-specific quantitative analysis')
    parser.add_argument('--sector', '-s', 
                       choices=['energy', 'materials', 'industrials', 'cons_discretionary', 
                               'cons_staples', 'healthcare', 'financials', 'it', 
                               'telecoms', 'utilities', 're'],
                       help='Sector to analyze')
    parser.add_argument('--list-sectors', '-l', action='store_true',
                       help='List available sectors')
    parser.add_argument('--test-only', '-t', action='store_true',
                       help='Test sector mapping only')
    
    args = parser.parse_args()
    
    # GICS mapping
    SECTOR_GICS_MAPPING = {
        'energy': '10', 'materials': '15', 'industrials': '20',
        'cons_discretionary': '25', 'cons_staples': '30', 'healthcare': '35',
        'financials': '40', 'it': '45', 'telecoms': '50',
        'utilities': '55', 're': '60'
    }
    
    if args.list_sectors:
        print("Available sectors:")
        for sector, gics in SECTOR_GICS_MAPPING.items():
            print(f"  {sector:20} (GICS: {gics})")
        return
    
    if not args.sector:
        parser.print_help()
        return
    
    # Set environment variables for the sector
    os.environ['SECTOR_NAME'] = args.sector
    os.environ['GICS_PREFIX'] = SECTOR_GICS_MAPPING[args.sector]
    
    print(f"🎯 Configuring for {args.sector.upper()} sector (GICS: {SECTOR_GICS_MAPPING[args.sector]})")
    
    try:
        # Import after setting environment variables
        from config import SECTOR_NAME, GICS_PREFIX, MA_WINDOW, MOM_LAG
        from sector_mapper import test_sector_mapping
        from pipeline import run
        
        print(f"✓ Configuration loaded:")
        print(f"  - Sector: {SECTOR_NAME}")
        print(f"  - GICS Prefix: {GICS_PREFIX}")
        print(f"  - MA Window: {MA_WINDOW}")
        print(f"  - Momentum Lag: {MOM_LAG}")
        
        if args.test_only:
            print(f"\n🧪 Testing {args.sector} sector mapping...")
            success = test_sector_mapping(
                gics_prefix=GICS_PREFIX,
                sector_name=SECTOR_NAME
            )
            if success:
                print(f"✅ {args.sector} sector mapping test PASSED!")
            else:
                print(f"❌ {args.sector} sector mapping test FAILED!")
            sys.exit(0 if success else 1)
        
        # Run full pipeline
        print(f"\n🚀 Running {args.sector} sector analysis...")
        run()
        print(f"✅ {args.sector} analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running {args.sector} analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
