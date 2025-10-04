#!/usr/bin/env python3
"""Convert pickle files in TextData year folders to Parquet format.

This script scans TextData year folders for .pkl files and converts them to .parquet
format in-place. It handles compression detection and provides fallback loading methods.

Usage:
    python tools/convert_pickles_to_parquet.py --years 2007 2008 2009
    python tools/convert_pickles_to_parquet.py  # converts all years
"""

import argparse
import gzip
import bz2
import pickle
import pandas as pd
from pathlib import Path
import sys


def safe_unpickle(filepath: Path) -> pd.DataFrame:
    """Attempt to unpickle with multiple methods and compression detection."""
    
    # Try reading raw bytes first to detect compression
    with open(filepath, 'rb') as f:
        head = f.read(8)
        f.seek(0)
        data = f.read()
    
    # Try different decompression methods
    loaders = [
        ("raw", lambda d: d),
        ("gzip", gzip.decompress),
        ("bz2", bz2.decompress)
    ]
    
    for method_name, decompress_func in loaders:
        try:
            decompressed_data = decompress_func(data)
            df = pickle.loads(decompressed_data)
            print(f"    Successfully loaded with {method_name} decompression")
            return df
        except Exception as e:
            continue
    
    # If all methods fail, try pandas directly (sometimes handles compression automatically)
    try:
        df = pd.read_pickle(filepath)
        print(f"    Successfully loaded with pandas.read_pickle")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read {filepath}: {e}")


def convert_pickle_to_parquet(pickle_path: Path, output_path: Path) -> bool:
    """Convert a single pickle file to Parquet format."""
    try:
        print(f"Converting {pickle_path} -> {output_path}")
        
        # Load the pickle file
        df = safe_unpickle(pickle_path)
        
        # Validate it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Pickle contains {type(df)}, not a DataFrame")
        
        print(f"    Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Write to Parquet
        df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        
        print(f"    Saved Parquet: {output_path}")
        
        # Verify the Parquet file can be read
        test_df = pd.read_parquet(output_path)
        if test_df.shape != df.shape:
            raise ValueError(f"Shape mismatch after conversion: {df.shape} -> {test_df.shape}")
        
        print(f"    Verification passed: {test_df.shape}")
        return True
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert TextData pickle files to Parquet")
    parser.add_argument("--years", nargs="*", type=int, 
                       help="Specific years to convert (e.g., 2007 2008 2009)")
    parser.add_argument("--base-dir", default="TextData", 
                       help="Base directory containing year folders")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"Error: Base directory {base_path} does not exist")
        return 1
    
    # Determine which years to process
    if args.years:
        years = args.years
    else:
        # Auto-detect years from subdirectories
        years = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                years.append(int(item.name))
        years.sort()
    
    if not years:
        print("No years found to process")
        return 1
    
    print(f"Processing years: {years}")
    
    converted = []
    failed = []
    
    for year in years:
        year_dir = base_path / str(year)
        if not year_dir.exists():
            print(f"Year directory {year_dir} does not exist, skipping")
            continue
        
        # Look for pickle files
        pickle_files = list(year_dir.glob("*.pkl"))
        if not pickle_files:
            print(f"No pickle files found in {year_dir}")
            continue
        
        for pickle_file in pickle_files:
            # Generate output path (same name but .parquet extension)
            parquet_file = pickle_file.with_suffix('.parquet')
            
            # Skip if Parquet already exists and is newer
            if parquet_file.exists() and parquet_file.stat().st_mtime > pickle_file.stat().st_mtime:
                print(f"Parquet file {parquet_file} is up to date, skipping")
                continue
            
            # Attempt conversion
            success = convert_pickle_to_parquet(pickle_file, parquet_file)
            
            if success:
                converted.append((year, str(pickle_file), str(parquet_file)))
            else:
                failed.append((year, str(pickle_file), "Conversion failed"))
    
    # Summary
    print(f"\nConversion Summary:")
    print(f"Converted: {len(converted)}")
    for year, pkl, pq in converted:
        print(f"  ✓ {year}: {Path(pkl).name} -> {Path(pq).name}")
    
    print(f"Failed: {len(failed)}")
    for year, pkl, error in failed:
        print(f"  ✗ {year}: {Path(pkl).name} - {error}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())