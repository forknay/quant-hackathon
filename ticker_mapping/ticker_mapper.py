#!/usr/bin/env python3
"""
Ticker Mapping Script

This script converts company identifiers (gvkey, iid, datadate) from algo results
to ticker symbols using the provided CSV mapping files.

"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ticker_mapping.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TickerMapper:
    """Main class for mapping company identifiers to ticker symbols."""

    def __init__(self, algo_results_path: str, mapping_files_path: str, output_path: str):
        """
        Initialize the ticker mapper.

        Args:
            algo_results_path: Path to algo results directory
            mapping_files_path: Path to directory containing CSV mapping files
            output_path: Path to output directory for results
        """
        self.algo_results_path = Path(algo_results_path)
        self.mapping_files_path = Path(mapping_files_path)
        self.output_path = Path(output_path)

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Mapping data storage
        self.global_mapping = None
        self.na_mapping = None
        self.combined_mapping = None

        logger.info(f"Initialized TickerMapper with paths:")
        logger.info(f"  Algo results: {self.algo_results_path}")
        logger.info(f"  Mapping files: {self.mapping_files_path}")
        logger.info(f"  Output: {self.output_path}")

    def load_mapping_files(self) -> None:
        """Load and merge the CSV mapping files."""
        logger.info("Loading mapping files...")

        # Define mapping file paths
        global_file = self.mapping_files_path / "Global (ex Canada and US) Company Name Merge by DataDate-GVKEY-IID.csv"
        na_file = self.mapping_files_path / "North America Company Name Merge by DataDate-GVKEY-IID.csv"

        # Load Global mapping (excludes Canada and US)
        logger.info(f"Loading Global mapping file: {global_file}")
        try:
            # Use chunking for large files
            chunks = []
            for chunk in pd.read_csv(
                global_file,
                chunksize=100000,
                low_memory=False,
                dtype={'gvkey': str, 'iid': str, 'fic': str}
            ):
                chunks.append(chunk)
            self.global_mapping = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(self.global_mapping)} records from Global mapping")
        except Exception as e:
            logger.error(f"Error loading Global mapping: {e}")
            self.global_mapping = pd.DataFrame()

        # Load North America mapping
        logger.info(f"Loading North America mapping file: {na_file}")
        try:
            chunks = []
            for chunk in pd.read_csv(
                na_file,
                chunksize=100000,
                low_memory=False,
                dtype={'gvkey': str, 'iid': str, 'tic': str, 'cusip': str, 'cik': str}
            ):
                chunks.append(chunk)
            self.na_mapping = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(self.na_mapping)} records from North America mapping")
        except Exception as e:
            logger.error(f"Error loading North America mapping: {e}")
            self.na_mapping = pd.DataFrame()

        # Combine mappings with priority for North America data
        self._create_combined_mapping()

    def _create_combined_mapping(self) -> None:
        """Create combined mapping with North America data taking priority."""
        logger.info("Creating combined mapping...")

        combined_dfs = []

        # Start with North America data (has ticker symbols)
        if not self.na_mapping.empty:
            combined_dfs.append(self.na_mapping.copy())

        # Add Global data where not already covered by North America
        if not self.global_mapping.empty:
            # Find records in Global that are not in North America
            na_keys = set()
            if not self.na_mapping.empty:
                na_keys = set(zip(
                    self.na_mapping['gvkey'].astype(str),
                    self.na_mapping['iid'].astype(str),
                    pd.to_datetime(self.na_mapping['datadate']).dt.date
                ))

            global_filtered = self.global_mapping.copy()
            if na_keys:
                global_keys = set(zip(
                    self.global_mapping['gvkey'].astype(str),
                    self.global_mapping['iid'].astype(str),
                    pd.to_datetime(self.global_mapping['datadate']).dt.date
                ))
                # Keep only Global records not in North America
                mask = [key not in na_keys for key in global_keys]
                global_filtered = self.global_mapping[mask].copy()

            if not global_filtered.empty:
                combined_dfs.append(global_filtered)

        if combined_dfs:
            self.combined_mapping = pd.concat(combined_dfs, ignore_index=True)

            # Ensure consistent data types
            self.combined_mapping['gvkey'] = self.combined_mapping['gvkey'].astype(str)
            self.combined_mapping['iid'] = self.combined_mapping['iid'].astype(str)
            self.combined_mapping['datadate'] = pd.to_datetime(self.combined_mapping['datadate'])

            # Create composite key for faster lookups
            self.combined_mapping['lookup_key'] = (
                self.combined_mapping['gvkey'] + '|' +
                self.combined_mapping['iid'] + '|' +
                self.combined_mapping['datadate'].dt.date.astype(str)
            )

            logger.info(f"Combined mapping has {len(self.combined_mapping)} records")
        else:
            logger.warning("No mapping data loaded!")
            self.combined_mapping = pd.DataFrame()

    def find_parquet_files(self) -> List[Path]:
        """Find all parquet files in the algo results directory."""
        logger.info("Finding parquet files...")

        # Look for all .parquet files in the algo results directory
        pattern = "**/*.parquet"
        parquet_files = list(self.algo_results_path.glob(pattern))

        logger.info(f"Found {len(parquet_files)} parquet files")
        return parquet_files

    def process_parquet_files(self) -> pd.DataFrame:
        """Process all parquet files and extract unique identifier combinations."""
        logger.info("Processing parquet files...")

        parquet_files = self.find_parquet_files()
        all_identifiers = []

        for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
            try:
                # Read parquet file
                table = pq.read_table(str(parquet_file))
                df = table.to_pandas()

                # Extract required columns
                if all(col in df.columns for col in ['gvkey', 'iid', 'date']):
                    # Convert date to datadate format if needed
                    identifiers = df[['gvkey', 'iid', 'date']].copy()

                    # Ensure data types
                    identifiers['gvkey'] = identifiers['gvkey'].astype(str)
                    identifiers['iid'] = identifiers['iid'].astype(str)
                    identifiers['date'] = pd.to_datetime(identifiers['date'])

                    # Rename date to datadate for consistency
                    identifiers = identifiers.rename(columns={'date': 'datadate'})

                    # Create lookup key
                    identifiers['lookup_key'] = (
                        identifiers['gvkey'] + '|' +
                        identifiers['iid'] + '|' +
                        identifiers['datadate'].dt.date.astype(str)
                    )

                    # Keep only unique combinations
                    identifiers = identifiers[['gvkey', 'iid', 'datadate', 'lookup_key']].drop_duplicates()
                    all_identifiers.append(identifiers)

            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                continue

        if all_identifiers:
            combined_identifiers = pd.concat(all_identifiers, ignore_index=True)
            # Remove duplicates across all files
            unique_identifiers = combined_identifiers.drop_duplicates(subset=['lookup_key'])
            logger.info(f"Extracted {len(unique_identifiers)} unique identifier combinations")
            return unique_identifiers
        else:
            logger.warning("No valid identifiers found in parquet files")
            return pd.DataFrame(columns=['gvkey', 'iid', 'datadate', 'lookup_key'])

    def map_to_tickers(self, identifiers_df: pd.DataFrame) -> pd.DataFrame:
        """Map identifiers to ticker symbols."""
        logger.info("Mapping identifiers to tickers...")

        if self.combined_mapping is None or self.combined_mapping.empty:
            logger.error("No mapping data available")
            return pd.DataFrame()

        if identifiers_df.empty:
            logger.warning("No identifiers to map")
            return pd.DataFrame()

        # Create mapping dictionary for faster lookups
        mapping_dict = {}
        for _, row in self.combined_mapping.iterrows():
            mapping_dict[row['lookup_key']] = {
                'tic': row.get('tic', ''),
                'cusip': row.get('cusip', ''),
                'conm': row.get('conm', ''),
                'cik': row.get('cik', ''),
                'fic': row.get('fic', '')
            }

        # Map the identifiers
        mapped_data = []
        unmapped_count = 0

        for _, row in tqdm(identifiers_df.iterrows(), desc="Mapping to tickers", total=len(identifiers_df)):
            lookup_key = row['lookup_key']
            mapping_info = mapping_dict.get(lookup_key, {})

            if mapping_info:
                mapped_row = {
                    'gvkey': row['gvkey'],
                    'iid': row['iid'],
                    'datadate': row['datadate'],
                    'lookup_key': lookup_key,
                    'ticker': mapping_info.get('tic', ''),
                    'cusip': mapping_info.get('cusip', ''),
                    'company_name': mapping_info.get('conm', ''),
                    'cik': mapping_info.get('cik', ''),
                    'fic': mapping_info.get('fic', ''),
                    'mapped': True
                }
            else:
                mapped_row = {
                    'gvkey': row['gvkey'],
                    'iid': row['iid'],
                    'datadate': row['datadate'],
                    'lookup_key': lookup_key,
                    'ticker': '',
                    'cusip': '',
                    'company_name': '',
                    'cik': '',
                    'fic': '',
                    'mapped': False
                }
                unmapped_count += 1

            mapped_data.append(mapped_row)

        result_df = pd.DataFrame(mapped_data)

        logger.info(f"Successfully mapped {len(result_df[result_df['mapped']])} identifiers")
        logger.info(f"Failed to map {unmapped_count} identifiers")

        return result_df

    def save_results(self, mapped_df: pd.DataFrame, filename_prefix: str = "ticker_mapped") -> None:
        """Save the mapped results to various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        full_filename = f"{filename_prefix}_full_{timestamp}.csv"
        full_path = self.output_path / full_filename
        mapped_df.to_csv(full_path, index=False)
        logger.info(f"Saved full results to {full_path}")

        # Save only successfully mapped results
        mapped_only = mapped_df[mapped_df['mapped']].copy()
        if not mapped_only.empty:
            mapped_filename = f"{filename_prefix}_mapped_{timestamp}.csv"
            mapped_path = self.output_path / mapped_filename
            mapped_only.to_csv(mapped_path, index=False)
            logger.info(f"Saved mapped results to {mapped_path}")

        # Save unmapped results for analysis
        unmapped_only = mapped_df[~mapped_df['mapped']].copy()
        if not unmapped_only.empty:
            unmapped_filename = f"{filename_prefix}_unmapped_{timestamp}.csv"
            unmapped_path = self.output_path / unmapped_filename
            unmapped_only.to_csv(unmapped_path, index=False)
            logger.info(f"Saved unmapped results to {unmapped_path}")

        # Save summary statistics
        self._save_summary_stats(mapped_df, timestamp, filename_prefix)

    def _save_summary_stats(self, mapped_df: pd.DataFrame, timestamp: str, filename_prefix: str) -> None:
        """Save summary statistics about the mapping process."""
        total = len(mapped_df)
        mapped = len(mapped_df[mapped_df['mapped']])
        unmapped = len(mapped_df[~mapped_df['mapped']])

        summary = {
            'total_identifiers': total,
            'successfully_mapped': mapped,
            'failed_to_map': unmapped,
            'mapping_success_rate': mapped / total if total > 0 else 0,
            'unique_tickers': mapped_df[mapped_df['mapped']]['ticker'].nunique() if mapped > 0 else 0,
            'unique_companies': mapped_df[mapped_df['mapped']]['company_name'].nunique() if mapped > 0 else 0,
            'processing_timestamp': timestamp
        }

        summary_df = pd.DataFrame([summary])
        summary_filename = f"{filename_prefix}_summary_{timestamp}.csv"
        summary_path = self.output_path / summary_filename
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary statistics to {summary_path}")

    def run(self) -> None:
        """Run the complete ticker mapping process."""
        logger.info("Starting ticker mapping process...")

        # Step 1: Load mapping files
        self.load_mapping_files()

        if self.combined_mapping is None or self.combined_mapping.empty:
            logger.error("Failed to load mapping files. Exiting.")
            return

        # Step 2: Process parquet files
        identifiers_df = self.process_parquet_files()

        if identifiers_df.empty:
            logger.error("No identifiers found in parquet files. Exiting.")
            return

        # Step 3: Map to tickers
        mapped_df = self.map_to_tickers(identifiers_df)

        # Step 4: Save results
        self.save_results(mapped_df)

        logger.info("Ticker mapping process completed successfully!")


def main():
    """Main function to run the ticker mapper."""
    # Define paths
    project_root = Path(__file__).parent.parent
    algo_results_path = project_root / "algo" / "results"
    mapping_files_path = project_root  # CSV files are in root
    output_path = project_root / "ticker_mapping"

    # Create and run the mapper
    mapper = TickerMapper(algo_results_path, mapping_files_path, output_path)
    mapper.run()


if __name__ == "__main__":
    main()
