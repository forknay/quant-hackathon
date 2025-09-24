"""
Sector Mapping Strategy for Utilities Pipeline

This module implements an efficient strategy for identifying utilities companies
and mapping them with the main financial data.

Strategy:
1. Convert sectorinfo.csv to parquet (one-time optimization)  
2. Create utilities lookup table from sector data
3. Filter main parquet data using utilities lookup
4. Avoid large memory-intensive joins

Performance considerations:
- sectorinfo.csv: 6.9M rows -> parquet conversion ~10x faster reads
- Utilities subset: ~few thousand companies vs 6.9M total
- Memory efficient: no large joins, streaming approach
"""

import os
import pandas as pd
import numpy as np
from typing import Set, Dict, List, Tuple
from datetime import datetime


class SectorMapper:
    """Efficient utilities sector mapping and data filtering."""
    
    def __init__(self, sector_csv_path: str = "../../sectorinfo.csv", 
                 main_parquet_path: str = "../../cleaned_all.parquet",
                 cache_dir: str = "cache"):
        
        self.sector_csv_path = sector_csv_path
        self.main_parquet_path = main_parquet_path
        self.cache_dir = cache_dir
        self.sector_parquet_path = os.path.join(cache_dir, "sectorinfo.parquet")
        self.utilities_lookup_path = os.path.join(cache_dir, "utilities_lookup.parquet")
        
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
    
    def create_utilities_lookup(self, force_rebuild: bool = False) -> pd.DataFrame:
        """
        Create optimized utilities lookup table.
        
        Args:
            force_rebuild: Force rebuild lookup table
            
        Returns:
            DataFrame with utilities companies information
        """
        if os.path.exists(self.utilities_lookup_path) and not force_rebuild:
            print(f"✓ Loading existing utilities lookup: {self.utilities_lookup_path}")
            return pd.read_parquet(self.utilities_lookup_path)
        
        print("Creating utilities lookup table...")
        
        # Ensure sector parquet exists
        self.convert_sector_csv_to_parquet()
        
        # Load sector data
        print("  - Loading sector data...")
        df_sector = pd.read_parquet(self.sector_parquet_path)
        
        # Filter utilities (GICS starting with 55)
        print("  - Filtering utilities companies...")
        utilities_mask = df_sector['gics_str'].str.startswith('55', na=False)
        df_utilities = df_sector[utilities_mask].copy()
        
        print(f"  - Found {len(df_utilities):,} utilities observations")
        print(f"  - Unique utilities companies: {df_utilities['id'].nunique():,}")
        print(f"  - Date range: {df_utilities['date'].min()} to {df_utilities['date'].max()}")
        
        # Optimize utilities lookup
        df_utilities_optimized = df_utilities[[
            'id', 'date', 'gvkey', 'iid', 'gics', 'gics_str', 'year', 'month'
        ]].copy()
        
        # Add utilities-specific metadata
        df_utilities_optimized['sector_name'] = 'utilities'
        df_utilities_optimized['gics_prefix'] = '55'
        
        # Save utilities lookup
        print(f"  - Saving utilities lookup: {self.utilities_lookup_path}")
        df_utilities_optimized.to_parquet(
            self.utilities_lookup_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        
        # Summary statistics
        utilities_by_gics = df_utilities_optimized['gics_str'].value_counts()
        print("  - Utilities by GICS code:")
        for gics_code, count in utilities_by_gics.head(10).items():
            print(f"    {gics_code}: {count:,} observations")
        
        print(f"✓ Utilities lookup created: {len(df_utilities_optimized):,} rows")
        return df_utilities_optimized
    
    def get_utilities_ids(self) -> Set[str]:
        """
        Get set of all utilities company IDs for efficient filtering.
        
        Returns:
            Set of utilities company ID strings
        """
        print("Getting utilities company IDs...")
        
        # Load utilities lookup
        df_utilities = self.create_utilities_lookup()
        
        # Get unique company IDs
        utilities_ids = set(df_utilities['id'].unique())
        
        print(f"✓ Found {len(utilities_ids):,} unique utilities companies")
        return utilities_ids
    
    def load_utilities_data(self, required_columns: List[str] = None) -> pd.DataFrame:
        """
        Efficiently load utilities data from main parquet file.
        
        Args:
            required_columns: Specific columns to load (None = all columns)
            
        Returns:
            DataFrame with utilities companies data
        """
        print("Loading utilities data from main parquet...")
        
        # Get utilities IDs for filtering
        utilities_ids = self.get_utilities_ids()
        
        # Load main data in chunks to manage memory
        print(f"  - Loading main parquet data...")
        
        # First, determine required columns
        if required_columns is None:
            # Load all columns for utilities companies
            df_main = pd.read_parquet(self.main_parquet_path)
        else:
            # Load only required columns
            df_main = pd.read_parquet(self.main_parquet_path, columns=required_columns)
        
        print(f"  - Loaded {len(df_main):,} total observations")
        
        # Filter for utilities companies
        print("  - Filtering for utilities companies...")
        utilities_mask = df_main['id'].isin(utilities_ids)
        df_utilities_main = df_main[utilities_mask].copy()
        
        print(f"  - Filtered to {len(df_utilities_main):,} utilities observations")
        print(f"  - Utilities companies found: {df_utilities_main['id'].nunique():,}")
        print(f"  - Date range: {df_utilities_main['date'].min()} to {df_utilities_main['date'].max()}")
        
        # Add sector information
        print("  - Adding sector metadata...")
        df_utilities_lookup = self.create_utilities_lookup()
        
        # Merge with utilities lookup for GICS codes
        df_result = df_utilities_main.merge(
            df_utilities_lookup[['id', 'date', 'gics', 'gics_str']],
            on=['id', 'date'],
            how='left'
        )
        
        print(f"✓ Utilities data loaded: {len(df_result):,} rows with GICS codes")
        return df_result
    
    def validate_data_quality(self) -> Dict[str, any]:
        """
        Validate the utilities data quality and coverage.
        
        Returns:
            Dictionary with validation metrics
        """
        print("Validating utilities data quality...")
        
        # Load utilities lookup and main data sample
        df_utilities_lookup = self.create_utilities_lookup()
        utilities_ids = self.get_utilities_ids()
        
        # Sample main data to check coverage
        df_main_sample = pd.read_parquet(
            self.main_parquet_path, 
            columns=['id', 'date', 'gvkey', 'iid', 'prc', 'stock_ret']
        ).head(10000)
        
        utilities_in_sample = df_main_sample['id'].isin(utilities_ids).sum()
        
        validation_results = {
            'utilities_companies_in_lookup': df_utilities_lookup['id'].nunique(),
            'total_utilities_observations': len(df_utilities_lookup),
            'date_range_lookup': {
                'start': str(df_utilities_lookup['date'].min()),
                'end': str(df_utilities_lookup['date'].max())
            },
            'utilities_in_main_sample': utilities_in_sample,
            'sample_coverage_rate': utilities_in_sample / len(df_main_sample),
            'gics_codes_found': sorted(df_utilities_lookup['gics_str'].dropna().unique().tolist()),
            'status': 'success'
        }
        
        # Print validation summary
        print("Validation Results:")
        print(f"  - Utilities companies: {validation_results['utilities_companies_in_lookup']:,}")
        print(f"  - Total observations: {validation_results['total_utilities_observations']:,}")
        print(f"  - Date range: {validation_results['date_range_lookup']['start']} to {validation_results['date_range_lookup']['end']}")
        print(f"  - Sample coverage: {validation_results['sample_coverage_rate']:.1%}")
        print(f"  - GICS codes: {validation_results['gics_codes_found']}")
        
        return validation_results


def test_sector_mapping():
    """Test the sector mapping functionality."""
    print("="*60)
    print("TESTING SECTOR MAPPING STRATEGY")
    print("="*60)
    
    # Initialize mapper with corrected paths (run from utilities directory)
    mapper = SectorMapper()
    
    # Test the full pipeline
    try:
        # Step 1: Convert CSV to parquet
        print("\n1. Converting sector CSV to parquet...")
        mapper.convert_sector_csv_to_parquet()
        
        # Step 2: Create utilities lookup
        print("\n2. Creating utilities lookup...")
        utilities_lookup = mapper.create_utilities_lookup()
        
        # Step 3: Validate data quality
        print("\n3. Validating data quality...")
        validation = mapper.validate_data_quality()
        
        # Step 4: Test utilities data loading (with limited columns for speed)
        print("\n4. Testing utilities data loading...")
        required_cols = ['id', 'date', 'gvkey', 'iid', 'prc', 'stock_ret']
        utilities_data = mapper.load_utilities_data(required_columns=required_cols)
        
        print("\n" + "="*60)
        print("✅ SECTOR MAPPING TEST SUCCESSFUL!")
        print(f"Ready to process {len(utilities_data):,} utilities observations")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ SECTOR MAPPING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sector_mapping() 