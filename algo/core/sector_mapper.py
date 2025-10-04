"""
Sector Mapping Strategy for Multi-Sector Pipeline

This module implements an efficient strategy for identifying companies by sector
and mapping them with the main financial data.

Strategy:
1. Convert sectorinfo.csv to parquet (one-time optimization)  
2. Create sector lookup table from sector data using GICS prefix
3. Filter main parquet data using sector lookup
4. Avoid large memory-intensive joins

Performance considerations:
- sectorinfo.csv: 6.9M rows -> parquet conversion ~10x faster reads
- Sector subset: ~few thousand companies vs 6.9M total
- Memory efficient: no large joins, streaming approach
"""

import os
import pandas as pd
import numpy as np
from typing import Set, Dict, List, Tuple
from datetime import datetime

# GICS Sector Mapping
SECTOR_GICS_MAPPING = {
    'energy': '10',
    'materials': '15', 
    'industrials': '20',
    'cons_discretionary': '25',
    'cons_staples': '30',
    'healthcare': '35',
    'financials': '40',
    'it': '45',
    'telecoms': '50',
    'utilities': '55',
    're': '60'  # Real Estate
}

class SectorMapper:
    """Efficient sector mapping and data filtering for any GICS sector."""
    
    def __init__(self, gics_prefix: str,
                 sector_name: str,
                 sector_csv_path: str = None, 
                 main_parquet_path: str = None,
                 cache_dir: str = "cache"):
        
        # Use absolute paths based on this file's location to avoid CWD issues
        if sector_csv_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sector_csv_path = os.path.join(script_dir, "..", "..", "sectorinfo.csv")
        
        if main_parquet_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            main_parquet_path = os.path.join(script_dir, "..", "..", "cleaned_all.parquet")
        
        self.sector_csv_path = os.path.abspath(sector_csv_path)
        self.main_parquet_path = os.path.abspath(main_parquet_path)
        self.cache_dir = cache_dir
        self.gics_prefix = gics_prefix
        self.sector_name = sector_name
        
        self.sector_parquet_path = os.path.join(cache_dir, "sectorinfo.parquet")
        self.sector_lookup_path = os.path.join(cache_dir, f"{sector_name}_lookup.parquet")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
    def convert_sector_csv_to_parquet(self, force_rebuild: bool = False) -> str:
        """
        Convert sectorinfo.csv to optimized parquet format.
        
        Args:
            force_rebuild: Force rebuild even if parquet exists
            
        Returns:
            Path to the parquet file
        """
        if os.path.exists(self.sector_parquet_path) and not force_rebuild:
            print(f"✓ Sector parquet already exists: {self.sector_parquet_path}")
            return self.sector_parquet_path
        
        print("Converting sectorinfo.csv to parquet (one-time optimization)...")
        
        # Read CSV with proper data types
        print("  - Reading CSV...")
        df_sector = pd.read_csv(
            self.sector_csv_path,
            dtype={
                'id': 'string',
                'gvkey': 'Int64',  # Nullable integer 
                'iid': 'string',
                'excntry': 'string',
                'gics': 'float64',  # Some missing values
                'sic': 'float64',
                'naics': 'float64',
                'year': 'int16',
                'month': 'int8'
            },
            parse_dates=['date', 'eom']
        )
        
        print(f"  - Loaded {len(df_sector):,} rows")
        
        # Data cleaning and optimization
        print("  - Optimizing data types...")
        
        # Clean and optimize
        df_sector = df_sector.dropna(subset=['id', 'date', 'gvkey'])  # Remove invalid rows
        df_sector['gics_str'] = df_sector['gics'].astype(str).str.replace('.0', '').str.replace('nan', '')
        
        # Save as parquet with compression
        print(f"  - Saving to parquet: {self.sector_parquet_path}")
        df_sector.to_parquet(
            self.sector_parquet_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        
        print(f"✓ Conversion complete. Saved {len(df_sector):,} rows")
        return self.sector_parquet_path
    
    def create_sector_lookup(self, force_rebuild: bool = False) -> pd.DataFrame:
        """
        Create optimized sector lookup table.
        
        Args:
            force_rebuild: Force rebuild lookup table
            
        Returns:
            DataFrame with sector companies information
        """
        if os.path.exists(self.sector_lookup_path) and not force_rebuild:
            print(f"✓ Loading existing {self.sector_name} lookup: {self.sector_lookup_path}")
            return pd.read_parquet(self.sector_lookup_path)
        
        print(f"Creating {self.sector_name} lookup table...")
        
        # Ensure sector parquet exists
        self.convert_sector_csv_to_parquet()
        
        # Load sector data
        print("  - Loading sector data...")
        df_sector = pd.read_parquet(self.sector_parquet_path)
        
        # Filter by GICS prefix
        print(f"  - Filtering {self.sector_name} companies (GICS prefix: {self.gics_prefix})...")
        sector_mask = df_sector['gics_str'].str.startswith(self.gics_prefix, na=False)
        df_sector_filtered = df_sector[sector_mask].copy()
        
        print(f"  - Found {len(df_sector_filtered):,} {self.sector_name} observations")
        print(f"  - Unique {self.sector_name} companies: {df_sector_filtered['id'].nunique():,}")
        print(f"  - Date range: {df_sector_filtered['date'].min()} to {df_sector_filtered['date'].max()}")
        
        # Optimize sector lookup
        df_sector_optimized = df_sector_filtered[[
            'id', 'date', 'gvkey', 'iid', 'gics', 'gics_str', 'year', 'month'
        ]].copy()
        
        # Add sector-specific metadata
        df_sector_optimized['sector_name'] = self.sector_name
        df_sector_optimized['gics_prefix'] = self.gics_prefix
        
        # Save sector lookup
        print(f"  - Saving {self.sector_name} lookup: {self.sector_lookup_path}")
        df_sector_optimized.to_parquet(
            self.sector_lookup_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        
        # Summary statistics
        sector_by_gics = df_sector_optimized['gics_str'].value_counts()
        print(f"  - {self.sector_name.title()} by GICS code:")
        for gics_code, count in sector_by_gics.head(10).items():
            print(f"    {gics_code}: {count:,} observations")
        
        print(f"✓ {self.sector_name.title()} lookup created: {len(df_sector_optimized):,} rows")
        return df_sector_optimized
    
    def get_sector_ids(self) -> Set[str]:
        """
        Get set of all sector company IDs for efficient filtering.
        
        Returns:
            Set of sector company ID strings
        """
        print(f"Getting {self.sector_name} company IDs...")
        
        # Load sector lookup
        df_sector = self.create_sector_lookup()
        
        # Get unique company IDs
        sector_ids = set(df_sector['id'].unique())
        
        print(f"✓ Found {len(sector_ids):,} unique {self.sector_name} companies")
        return sector_ids
    
    def load_sector_data(self, required_columns: List[str] = None) -> pd.DataFrame:
        """
        Efficiently load sector data from main parquet file.
        
        Args:
            required_columns: Specific columns to load (None = all columns)
            
        Returns:
            DataFrame with sector companies data
        """
        print(f"Loading {self.sector_name} data from main parquet...")
        
        # Get sector IDs for filtering
        sector_ids = self.get_sector_ids()
        
        # Load main data in chunks to manage memory
        print(f"  - Loading main parquet data...")
        
        # First, determine required columns
        if required_columns is None:
            # Load all columns for sector companies
            df_main = pd.read_parquet(self.main_parquet_path)
        else:
            # Load only required columns
            df_main = pd.read_parquet(self.main_parquet_path, columns=required_columns)
        
        print(f"  - Loaded {len(df_main):,} total observations")
        
        # Filter for sector companies
        print(f"  - Filtering for {self.sector_name} companies...")
        sector_mask = df_main['id'].isin(sector_ids)
        df_sector_main = df_main[sector_mask].copy()
        
        print(f"  - Filtered to {len(df_sector_main):,} {self.sector_name} observations")
        print(f"  - {self.sector_name.title()} companies found: {df_sector_main['id'].nunique():,}")
        print(f"  - Date range: {df_sector_main['date'].min()} to {df_sector_main['date'].max()}")
        
        # Add sector information
        print("  - Adding sector metadata...")
        df_sector_lookup = self.create_sector_lookup()
        
        # Merge with sector lookup for GICS codes
        df_result = df_sector_main.merge(
            df_sector_lookup[['id', 'date', 'gics', 'gics_str']],
            on=['id', 'date'],
            how='left'
        )
        
        print(f"✓ {self.sector_name.title()} data loaded: {len(df_result):,} rows with GICS codes")
        return df_result
    
    def validate_data_quality(self) -> Dict[str, any]:
        """
        Validate the sector data quality and coverage.
        
        Returns:
            Dictionary with validation metrics
        """
        print(f"Validating {self.sector_name} data quality...")
        
        # Load sector lookup and main data sample
        df_sector_lookup = self.create_sector_lookup()
        sector_ids = self.get_sector_ids()
        
        # Sample main data to check coverage
        df_main_sample = pd.read_parquet(
            self.main_parquet_path, 
            columns=['id', 'date', 'gvkey', 'iid', 'prc', 'stock_ret']
        ).head(10000)
        
        sector_in_sample = df_main_sample['id'].isin(sector_ids).sum()
        
        validation_results = {
            f'{self.sector_name}_companies_in_lookup': df_sector_lookup['id'].nunique(),
            f'total_{self.sector_name}_observations': len(df_sector_lookup),
            'date_range_lookup': {
                'start': str(df_sector_lookup['date'].min()),
                'end': str(df_sector_lookup['date'].max())
            },
            f'{self.sector_name}_in_main_sample': sector_in_sample,
            'sample_coverage_rate': sector_in_sample / len(df_main_sample),
            'gics_codes_found': sorted(df_sector_lookup['gics_str'].dropna().unique().tolist()),
            'sector_name': self.sector_name,
            'gics_prefix': self.gics_prefix,
            'status': 'success'
        }
        
        # Print validation summary
        print("Validation Results:")
        print(f"  - {self.sector_name.title()} companies: {validation_results[f'{self.sector_name}_companies_in_lookup']:,}")
        print(f"  - Total observations: {validation_results[f'total_{self.sector_name}_observations']:,}")
        print(f"  - Date range: {validation_results['date_range_lookup']['start']} to {validation_results['date_range_lookup']['end']}")
        print(f"  - Sample coverage: {validation_results['sample_coverage_rate']:.1%}")
        print(f"  - GICS codes: {validation_results['gics_codes_found']}")
        
        return validation_results

    # Backward compatibility methods (deprecated but maintained)
    def create_utilities_lookup(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Deprecated: Use create_sector_lookup() instead."""
        print("⚠ Warning: create_utilities_lookup() is deprecated. Use create_sector_lookup() instead.")
        return self.create_sector_lookup(force_rebuild)
    
    def get_utilities_ids(self) -> Set[str]:
        """Deprecated: Use get_sector_ids() instead."""
        print("⚠ Warning: get_utilities_ids() is deprecated. Use get_sector_ids() instead.")
        return self.get_sector_ids()
    
    def load_utilities_data(self, required_columns: List[str] = None) -> pd.DataFrame:
        """Deprecated: Use load_sector_data() instead."""
        print("⚠ Warning: load_utilities_data() is deprecated. Use load_sector_data() instead.")
        return self.load_sector_data(required_columns)


def test_sector_mapping(gics_prefix: str = "55", sector_name: str = "utilities"):
    """Test the sector mapping functionality for any sector."""
    print("="*60)
    print(f"TESTING SECTOR MAPPING STRATEGY - {sector_name.upper()}")
    print("="*60)
    
    # Initialize mapper with sector parameters
    mapper = SectorMapper(
        gics_prefix=gics_prefix,
        sector_name=sector_name
    )
    
    # Test the full pipeline
    try:
        # Step 1: Convert CSV to parquet
        print("\n1. Converting sector CSV to parquet...")
        mapper.convert_sector_csv_to_parquet()
        
        # Step 2: Create sector lookup
        print(f"\n2. Creating {sector_name} lookup...")
        sector_lookup = mapper.create_sector_lookup()
        
        # Step 3: Validate data quality
        print("\n3. Validating data quality...")
        validation = mapper.validate_data_quality()
        
        # Step 4: Test sector data loading (with limited columns for speed)
        print(f"\n4. Testing {sector_name} data loading...")
        required_cols = ['id', 'date', 'gvkey', 'iid', 'prc', 'stock_ret']
        sector_data = mapper.load_sector_data(required_columns=required_cols)
        
        print("\n" + "="*60)
        print(f"✅ SECTOR MAPPING TEST SUCCESSFUL - {sector_name.upper()}!")
        print(f"Ready to process {len(sector_data):,} {sector_name} observations")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ SECTOR MAPPING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test with utilities by default (backward compatibility)
    test_sector_mapping() 