"""
Real Data Structure Analysis Script

This script analyzes the actual Data folder and TextData folder structure
to understand how to properly map stocks to their corresponding text data.
It doesn't process the full data (which is in GBs) but analyzes the structure.
"""

import pandas as pd
import os
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class RealDataAnalyzer:
    def __init__(self, base_path="C:/Users/positive/Documents/GitHub/quant-hackathon"):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "Data"
        self.textdata_path = self.base_path / "TextData"
        self.analysis_results = {}
        
    def analyze_data_folder(self):
        """Analyze the Data folder to understand stock identifiers and performance data"""
        print("=" * 60)
        print("ANALYZING DATA FOLDER")
        print("=" * 60)
        
        if not self.data_path.exists():
            print(f"❌ Data folder not found: {self.data_path}")
            return
            
        # List all files in Data folder
        data_files = list(self.data_path.glob("*"))
        print(f"📁 Found {len(data_files)} files in Data folder:")
        for file in data_files:
            print(f"  - {file.name} ({self.get_file_size(file)})")
        
        # Analyze key mapping files
        key_files = {
            'link_table': 'cik_gvkey_linktable_USA_only.csv',
            'returns': 'ret_sample.csv', 
            'market_ind': 'mkt_ind.csv',
            'factor_chars': 'factor_char_list.csv',
            'stock_features': 'Stock_features__Feature__Acronym_.csv',
            'na_company': 'North America Company Name Merge by DataDate-GVKEY-IID.csv',
            'global_company': 'Global (ex Canada and US) Company Name Merge by DataDate-GVKEY-IID.csv'
        }
        
        self.analysis_results['data_files'] = {}
        
        for key, filename in key_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                print(f"\n📊 Analyzing {filename}:")
                try:
                    # Read first few rows to understand structure
                    df_sample = pd.read_csv(file_path, nrows=1000)
                    
                    info = {
                        'columns': list(df_sample.columns),
                        'shape_sample': df_sample.shape,
                        'dtypes': df_sample.dtypes.to_dict()
                    }
                    
                    # Look for key identifier columns
                    key_cols = []
                    for col in df_sample.columns:
                        col_lower = col.lower()
                        if any(identifier in col_lower for identifier in ['gvkey', 'cik', 'iid', 'ticker', 'company', 'name']):
                            key_cols.append(col)
                    
                    if key_cols:
                        info['key_columns'] = key_cols
                        print(f"  🔑 Key columns: {key_cols}")
                        
                        # Show sample values for key columns
                        for col in key_cols[:3]:  # Limit to first 3 key columns
                            unique_vals = df_sample[col].unique()
                            print(f"    {col}: {len(unique_vals)} unique values")
                            print(f"      Sample: {list(unique_vals[:5])}")
                    
                    # Look for performance/return columns
                    perf_cols = []
                    for col in df_sample.columns:
                        col_lower = col.lower()
                        if any(perf in col_lower for perf in ['ret', 'return', 'price', 'performance', 'vol']):
                            perf_cols.append(col)
                    
                    if perf_cols:
                        info['performance_columns'] = perf_cols
                        print(f"  📈 Performance columns: {perf_cols}")
                    
                    # Look for date columns
                    date_cols = []
                    for col in df_sample.columns:
                        col_lower = col.lower()
                        if any(date in col_lower for date in ['date', 'datadate', 'year', 'month']):
                            date_cols.append(col)
                    
                    if date_cols:
                        info['date_columns'] = date_cols
                        print(f"  📅 Date columns: {date_cols}")
                    
                    print(f"  📏 Shape: {info['shape_sample']} (sample)")
                    print(f"  📋 Columns: {len(info['columns'])}")
                    
                    self.analysis_results['data_files'][key] = info
                    
                except Exception as e:
                    print(f"  ❌ Error reading {filename}: {e}")
                    self.analysis_results['data_files'][key] = {'error': str(e)}
            else:
                print(f"  ❌ File not found: {filename}")
        
        return self.analysis_results['data_files']
    
    def analyze_textdata_folder(self):
        """Analyze TextData folder structure to understand how text files are organized"""
        print("\n" + "=" * 60)
        print("ANALYZING TEXTDATA FOLDER")
        print("=" * 60)
        
        if not self.textdata_path.exists():
            print(f"❌ TextData folder not found: {self.textdata_path}")
            return
            
        # Get year directories
        year_dirs = [d for d in self.textdata_path.iterdir() if d.is_dir() and d.name.isdigit()]
        year_dirs.sort()
        
        print(f"📁 Found {len(year_dirs)} year directories:")
        for year_dir in year_dirs:
            file_count = len(list(year_dir.glob("*.parquet")))
            total_size = sum(f.stat().st_size for f in year_dir.glob("*.parquet"))
            print(f"  - {year_dir.name}: {file_count} parquet files ({self.format_size(total_size)})")
        
        # Analyze recent years in detail (2023, 2024)
        recent_years = [d for d in year_dirs if int(d.name) >= 2023]
        
        self.analysis_results['textdata'] = {}
        
        for year_dir in recent_years:
            print(f"\n📊 Analyzing {year_dir.name} TextData:")
            year_key = f"year_{year_dir.name}"
            
            parquet_files = list(year_dir.glob("*.parquet"))
            if not parquet_files:
                print(f"  ❌ No parquet files found in {year_dir.name}")
                continue
                
            # Analyze first few parquet files to understand structure
            sample_file = parquet_files[0]
            print(f"  🔍 Sampling structure from: {sample_file.name}")
            
            try:
                # Read small sample to understand structure
                df_sample = pd.read_parquet(sample_file, engine='pyarrow')
                if len(df_sample) > 1000:
                    df_sample = df_sample.sample(1000, random_state=42)
                
                info = {
                    'total_files': len(parquet_files),
                    'sample_file': sample_file.name,
                    'columns': list(df_sample.columns),
                    'sample_shape': df_sample.shape,
                    'dtypes': df_sample.dtypes.to_dict()
                }
                
                print(f"  📏 Sample shape: {df_sample.shape}")
                print(f"  📋 Columns: {list(df_sample.columns)}")
                
                # Analyze key identifier columns
                key_cols = []
                for col in df_sample.columns:
                    col_lower = col.lower()
                    if any(identifier in col_lower for identifier in ['gvkey', 'cik', 'iid', 'company']):
                        key_cols.append(col)
                
                if key_cols:
                    print(f"  🔑 Identifier columns: {key_cols}")
                    info['identifier_columns'] = key_cols
                    
                    # Show sample identifier values
                    for col in key_cols:
                        unique_vals = df_sample[col].unique()
                        print(f"    {col}: {len(unique_vals)} unique values in sample")
                        if len(unique_vals) <= 10:
                            print(f"      Values: {list(unique_vals)}")
                        else:
                            print(f"      Sample: {list(unique_vals[:5])}")
                
                # Look for text columns
                text_cols = []
                for col in df_sample.columns:
                    col_lower = col.lower()
                    if any(text in col_lower for text in ['text', 'rf', 'mgmt', 'content']):
                        text_cols.append(col)
                
                if text_cols:
                    print(f"  📝 Text columns: {text_cols}")
                    info['text_columns'] = text_cols
                    
                    # Analyze text content
                    for col in text_cols[:2]:  # Limit to first 2 text columns
                        if col in df_sample.columns:
                            text_series = df_sample[col].dropna()
                            if len(text_series) > 0:
                                text_lengths = text_series.str.len()
                                print(f"    {col}: avg length {text_lengths.mean():.0f} chars")
                                print(f"      Range: {text_lengths.min()}-{text_lengths.max()} chars")
                                # Show short sample
                                sample_text = str(text_series.iloc[0])[:100]
                                print(f"      Sample: '{sample_text}...'")
                
                # Look for date columns
                date_cols = []
                for col in df_sample.columns:
                    col_lower = col.lower()
                    if any(date in col_lower for date in ['date', 'datadate', 'filing']):
                        date_cols.append(col)
                
                if date_cols:
                    print(f"  📅 Date columns: {date_cols}")
                    info['date_columns'] = date_cols
                
                self.analysis_results['textdata'][year_key] = info
                
            except Exception as e:
                print(f"  ❌ Error analyzing {year_dir.name}: {e}")
                self.analysis_results['textdata'][year_key] = {'error': str(e)}
    
    def identify_mapping_strategy(self):
        """Based on analysis, identify how to map stocks to text data"""
        print("\n" + "=" * 60)
        print("MAPPING STRATEGY IDENTIFICATION")
        print("=" * 60)
        
        mapping_strategy = {
            'timestamp': datetime.now().isoformat(),
            'primary_identifiers': [],
            'performance_data_source': None,
            'text_data_structure': {},
            'recommended_approach': []
        }
        
        # Identify primary stock identifiers
        print("🔍 Identifying stock identifier strategy:")
        
        # Check if we found gvkey in both data and textdata
        data_gvkeys = set()
        textdata_gvkeys = set()
        
        for file_key, file_info in self.analysis_results.get('data_files', {}).items():
            if 'key_columns' in file_info:
                for col in file_info['key_columns']:
                    if 'gvkey' in col.lower():
                        data_gvkeys.add(f"{file_key}.{col}")
        
        for year_key, year_info in self.analysis_results.get('textdata', {}).items():
            if 'identifier_columns' in year_info:
                for col in year_info['identifier_columns']:
                    if 'gvkey' in col.lower():
                        textdata_gvkeys.add(f"{year_key}.{col}")
        
        if data_gvkeys and textdata_gvkeys:
            print("  ✅ GVKEY found in both Data and TextData")
            mapping_strategy['primary_identifiers'].append('gvkey')
            print(f"    Data sources: {list(data_gvkeys)}")
            print(f"    TextData sources: {list(textdata_gvkeys)}")
        
        # Identify performance data source
        print("\n🎯 Identifying performance data source:")
        perf_sources = []
        for file_key, file_info in self.analysis_results.get('data_files', {}).items():
            if 'performance_columns' in file_info:
                perf_sources.append(file_key)
                print(f"  📈 {file_key}: {file_info['performance_columns']}")
        
        if perf_sources:
            mapping_strategy['performance_data_source'] = perf_sources[0]  # Use first found
            print(f"  ✅ Recommended performance source: {perf_sources[0]}")
        
        # Recommended processing approach
        print("\n💡 Recommended processing approach:")
        approach = []
        
        if mapping_strategy['primary_identifiers'] and mapping_strategy['performance_data_source']:
            approach.extend([
                "1. Load performance data to identify top/bottom stocks by GVKEY",
                "2. Use GVKEY to filter TextData files for relevant companies", 
                "3. Extract text from recent years (2023-2024) for selected GVKEYs",
                "4. Process extracted text through FinBERT for sentiment analysis",
                "5. Aggregate sentiment scores by stock and rank"
            ])
        else:
            approach.extend([
                "⚠️ Missing key components for mapping",
                "Need both: stock identifiers + performance data + text data structure"
            ])
        
        mapping_strategy['recommended_approach'] = approach
        for step in approach:
            print(f"  {step}")
        
        self.analysis_results['mapping_strategy'] = mapping_strategy
        return mapping_strategy
    
    def save_analysis_results(self):
        """Save analysis results to JSON file"""
        output_file = self.base_path / "sentiment_analysis" / "real_data_analysis.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_safe_results = self.make_json_safe(self.analysis_results)
            json.dump(json_safe_results, f, indent=2)
        
        print(f"\n💾 Analysis results saved to: {output_file}")
        return output_file
    
    def make_json_safe(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_safe(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_file_size(self, file_path):
        """Get human-readable file size"""
        try:
            size = file_path.stat().st_size
            return self.format_size(size)
        except:
            return "unknown"
    
    def format_size(self, size_bytes):
        """Format size in bytes to human readable string"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def run_complete_analysis(self):
        """Run complete data analysis"""
        print("🚀 STARTING REAL DATA ANALYSIS")
        print("=" * 60)
        
        # Run all analysis steps
        self.analyze_data_folder()
        self.analyze_textdata_folder()
        self.identify_mapping_strategy()
        
        # Save results
        output_file = self.save_analysis_results()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"📊 Results saved to: {output_file}")
        
        return self.analysis_results

def main():
    """Main function to run the analysis"""
    analyzer = RealDataAnalyzer()
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n📋 SUMMARY:")
    data_files = len(results.get('data_files', {}))
    textdata_years = len(results.get('textdata', {}))
    
    print(f"  - Analyzed {data_files} data files")
    print(f"  - Analyzed {textdata_years} TextData years") 
    print(f"  - Generated mapping strategy")
    
    mapping = results.get('mapping_strategy', {})
    if mapping.get('primary_identifiers'):
        print(f"  ✅ Found identifiers: {mapping['primary_identifiers']}")
    if mapping.get('performance_data_source'):
        print(f"  ✅ Found performance data: {mapping['performance_data_source']}")
        
    print(f"\n🎯 Next: Use analysis results to build real data extraction pipeline")

if __name__ == "__main__":
    main()