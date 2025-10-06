#!/usr/bin/env python3
"""
TOP HOLDINGS ANALYZER
====================

Analyzes all portfolio JSON files from 2015-2025 to identify:
1. Top 10 best holdings based on multiple criteria
2. Holdings frequency and performance metrics
3. Sector distribution of top performers
4. Consistency and risk-adjusted performance

Author: GitHub Copilot
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TopHoldingsAnalyzer:
    """
    Analyze portfolio holdings across the entire 10-year backtesting period
    """
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.portfolio_files = []
        self.holdings_data = []
        self.company_stats = defaultdict(lambda: {
            'appearances': 0,
            'total_weight': 0.0,
            'avg_weight': 0.0,
            'combined_scores': [],
            'portfolio_weights': [],
            'sectors': set(),
            'position_types': [],
            'monthly_data': []
        })
        
    def load_all_portfolios(self):
        """Load all portfolio JSON files from the results directory"""
        print("Loading all portfolio files...")
        
        # Find all portfolio JSON files
        pattern = "portfolio_*.json"
        self.portfolio_files = list(self.results_dir.glob(pattern))
        
        print(f"Found {len(self.portfolio_files)} portfolio files")
        
        # Load each portfolio file
        for i, portfolio_file in enumerate(self.portfolio_files):
            if i % 50 == 0:
                print(f"  Processing: {i+1}/{len(self.portfolio_files)}")
                
            try:
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                
                # Extract date info from filename
                filename_parts = portfolio_file.stem.split('_')
                if len(filename_parts) >= 3:
                    year = int(filename_parts[1])
                    month = int(filename_parts[2])
                    sector = '_'.join(filename_parts[3:])
                    
                    # Process long positions
                    if 'long_positions' in portfolio_data:
                        for position in portfolio_data['long_positions']:
                            self._process_position(position, year, month, sector, 'long')
                    
                    # Process short positions
                    if 'short_positions' in portfolio_data:
                        for position in portfolio_data['short_positions']:
                            self._process_position(position, year, month, sector, 'short')
                            
            except Exception as e:
                print(f"  Error processing {portfolio_file}: {e}")
                continue
        
        print(f"Loaded data for {len(self.company_stats)} unique companies")
        
    def _process_position(self, position, year, month, sector, position_type):
        """Process individual position data"""
        company_id = position.get('company_id', '')
        if not company_id:
            return
            
        # Update company statistics
        company = self.company_stats[company_id]
        company['appearances'] += 1
        company['sectors'].add(sector)
        company['position_types'].append(position_type)
        
        # Extract metrics
        portfolio_weight = position.get('portfolio_weight', 0.0)
        combined_score = position.get('combined_score', 0.0)
        
        company['total_weight'] += abs(portfolio_weight)  # Use absolute value for comparison
        company['combined_scores'].append(combined_score)
        company['portfolio_weights'].append(portfolio_weight)
        
        # Store monthly data
        company['monthly_data'].append({
            'year': year,
            'month': month,
            'sector': sector,
            'position_type': position_type,
            'portfolio_weight': portfolio_weight,
            'combined_score': combined_score
        })
        
    def calculate_company_metrics(self):
        """Calculate comprehensive metrics for each company"""
        print("Calculating company performance metrics...")
        
        for company_id, stats in self.company_stats.items():
            if stats['appearances'] == 0:
                continue
                
            # Basic averages
            stats['avg_weight'] = stats['total_weight'] / stats['appearances']
            stats['avg_combined_score'] = np.mean(stats['combined_scores']) if stats['combined_scores'] else 0
            
            # Volatility and consistency metrics
            if len(stats['combined_scores']) > 1:
                stats['score_volatility'] = np.std(stats['combined_scores'])
                stats['weight_volatility'] = np.std([abs(w) for w in stats['portfolio_weights']])
            else:
                stats['score_volatility'] = 0
                stats['weight_volatility'] = 0
            
            # Risk-adjusted performance
            if stats['score_volatility'] > 0:
                stats['risk_adjusted_score'] = stats['avg_combined_score'] / stats['score_volatility']
            else:
                stats['risk_adjusted_score'] = stats['avg_combined_score']
            
            # Frequency and consistency
            stats['frequency_score'] = stats['appearances']
            stats['primary_sector'] = max(stats['sectors'], key=lambda s: sum(1 for md in stats['monthly_data'] if md['sector'] == s))
            stats['primary_position_type'] = max(set(stats['position_types']), key=stats['position_types'].count)
            
            # Performance consistency (lower is better)
            stats['consistency_score'] = 1 / (1 + stats['score_volatility'])
            
            # Combined ranking score (higher is better)
            stats['overall_score'] = (
                stats['avg_combined_score'] * 0.4 +           # Performance weight
                stats['risk_adjusted_score'] * 0.3 +          # Risk-adjusted performance
                (stats['frequency_score'] / 100) * 0.2 +      # Frequency (normalized)
                stats['consistency_score'] * 0.1              # Consistency
            )
    
    def get_top_holdings(self, n=10, criteria='overall_score'):
        """Get top N holdings based on specified criteria"""
        # Create a list of companies with their metrics
        companies = []
        for company_id, stats in self.company_stats.items():
            if stats['appearances'] >= 3:  # Minimum 3 appearances for significance
                companies.append({
                    'company_id': company_id,
                    'appearances': stats['appearances'],
                    'avg_combined_score': stats['avg_combined_score'],
                    'avg_weight': stats['avg_weight'],
                    'risk_adjusted_score': stats['risk_adjusted_score'],
                    'frequency_score': stats['frequency_score'],
                    'consistency_score': stats['consistency_score'],
                    'overall_score': stats['overall_score'],
                    'primary_sector': stats['primary_sector'],
                    'primary_position_type': stats['primary_position_type'],
                    'score_volatility': stats['score_volatility'],
                    'weight_volatility': stats['weight_volatility']
                })
        
        # Sort by the specified criteria
        companies_df = pd.DataFrame(companies)
        if len(companies_df) == 0:
            return pd.DataFrame()
            
        top_companies = companies_df.nlargest(n, criteria)
        return top_companies
    
    def analyze_sector_performance(self):
        """Analyze performance by sector"""
        sector_stats = defaultdict(lambda: {
            'companies': 0,
            'total_appearances': 0,
            'avg_scores': [],
            'avg_weights': []
        })
        
        for company_id, stats in self.company_stats.items():
            if stats['appearances'] >= 3:
                primary_sector = stats['primary_sector']
                sector_stats[primary_sector]['companies'] += 1
                sector_stats[primary_sector]['total_appearances'] += stats['appearances']
                sector_stats[primary_sector]['avg_scores'].append(stats['avg_combined_score'])
                sector_stats[primary_sector]['avg_weights'].append(stats['avg_weight'])
        
        # Calculate sector averages
        sector_summary = []
        for sector, stats in sector_stats.items():
            if stats['companies'] > 0:
                sector_summary.append({
                    'sector': sector,
                    'companies': stats['companies'],
                    'total_appearances': stats['total_appearances'],
                    'avg_score': np.mean(stats['avg_scores']),
                    'avg_weight': np.mean(stats['avg_weights']),
                    'score_std': np.std(stats['avg_scores']) if len(stats['avg_scores']) > 1 else 0
                })
        
        return pd.DataFrame(sector_summary).sort_values('avg_score', ascending=False)
    
    def generate_detailed_report(self, top_n=10):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("TOP HOLDINGS ANALYSIS REPORT")
        print("Period: January 2015 - May 2025")
        print("="*80)
        
        # Overall statistics
        total_companies = len(self.company_stats)
        significant_companies = len([c for c in self.company_stats.values() if c['appearances'] >= 3])
        
        print(f"\nOVERALL STATISTICS:")
        print(f"   Total unique companies analyzed: {total_companies:,}")
        print(f"   Companies with ≥3 appearances: {significant_companies:,}")
        print(f"   Total portfolio files processed: {len(self.portfolio_files):,}")
        
        # Top holdings by different criteria
        criteria_list = [
            ('overall_score', 'Overall Performance Score'),
            ('avg_combined_score', 'Average Combined Score'),
            ('risk_adjusted_score', 'Risk-Adjusted Performance'),
            ('frequency_score', 'Frequency (Most Selected)'),
            ('consistency_score', 'Consistency (Lowest Volatility)')
        ]
        
        for criteria, title in criteria_list:
            print(f"\n{title.upper()}:")
            print("-" * 60)
            
            top_holdings = self.get_top_holdings(n=top_n, criteria=criteria)
            if len(top_holdings) > 0:
                for i, (_, row) in enumerate(top_holdings.iterrows(), 1):
                    print(f"   {i:2d}. {row['company_id']:<25} | {row['primary_sector']:<15} | {row[criteria]:.4f}")
                    print(f"       Appearances: {row['appearances']:3d} | Avg Score: {row['avg_combined_score']:7.4f} | Risk-Adj: {row['risk_adjusted_score']:7.4f}")
            else:
                print("   No data available")
        
        # Sector analysis
        print(f"\nSECTOR PERFORMANCE ANALYSIS:")
        print("-" * 60)
        sector_perf = self.analyze_sector_performance()
        if len(sector_perf) > 0:
            for _, row in sector_perf.iterrows():
                print(f"   {row['sector']:<20} | Companies: {row['companies']:3d} | Avg Score: {row['avg_score']:7.4f} | Std: {row['score_std']:.4f}")
        
        # Detailed top 10 analysis
        print(f"\nTOP 10 BEST HOLDINGS DETAILED ANALYSIS:")
        print("="*80)
        
        top_10 = self.get_top_holdings(n=10, criteria='overall_score')
        if len(top_10) > 0:
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                company_id = row['company_id']
                stats = self.company_stats[company_id]
                
                print(f"\n{i:2d}. {company_id}")
                print(f"    Overall Score: {row['overall_score']:.4f}")
                print(f"    Primary Sector: {row['primary_sector']}")
                print(f"    Position Type: {row['primary_position_type']}")
                print(f"    Appearances: {row['appearances']} times")
                print(f"    Average Combined Score: {row['avg_combined_score']:.4f}")
                print(f"    Average Portfolio Weight: {row['avg_weight']:.4f}")
                print(f"    Risk-Adjusted Score: {row['risk_adjusted_score']:.4f}")
                print(f"    Consistency Score: {row['consistency_score']:.4f}")
                print(f"    Score Volatility: {row['score_volatility']:.4f}")
                
                # Recent performance trend
                recent_data = sorted(stats['monthly_data'], key=lambda x: (x['year'], x['month']))[-12:]
                if recent_data:
                    recent_scores = [d['combined_score'] for d in recent_data]
                    recent_avg = np.mean(recent_scores)
                    print(f"    Recent 12-month Avg Score: {recent_avg:.4f}")
        
        return top_10
    
    def save_results(self, top_holdings, filename="top_holdings_analysis.csv"):
        """Save detailed results to CSV"""
        if len(top_holdings) > 0:
            # Add detailed monthly data for top holdings
            detailed_results = []
            
            for _, row in top_holdings.iterrows():
                company_id = row['company_id']
                stats = self.company_stats[company_id]
                
                # Create base record
                base_record = row.to_dict()
                
                # Add sector breakdown
                sector_counts = Counter(md['sector'] for md in stats['monthly_data'])
                base_record['sectors_appeared'] = ', '.join([f"{s}({c})" for s, c in sector_counts.most_common()])
                
                # Add position type breakdown
                position_counts = Counter(stats['position_types'])
                base_record['position_breakdown'] = f"Long: {position_counts.get('long', 0)}, Short: {position_counts.get('short', 0)}"
                
                # Add time range
                years = [md['year'] for md in stats['monthly_data']]
                base_record['first_appearance'] = min(years)
                base_record['last_appearance'] = max(years)
                
                detailed_results.append(base_record)
            
            results_df = pd.DataFrame(detailed_results)
            output_path = self.results_dir / filename
            results_df.to_csv(output_path, index=False)
            print(f"\nDetailed results saved to: {output_path}")
            
            return results_df
        
        return pd.DataFrame()

def main():
    """Main analysis execution"""
    print("Starting Top Holdings Analysis...")
    
    # Initialize analyzer
    analyzer = TopHoldingsAnalyzer()
    
    # Load all portfolio data
    analyzer.load_all_portfolios()
    
    # Calculate metrics
    analyzer.calculate_company_metrics()
    
    # Generate comprehensive report
    top_holdings = analyzer.generate_detailed_report(top_n=10)
    
    # Save results
    detailed_results = analyzer.save_results(top_holdings)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()