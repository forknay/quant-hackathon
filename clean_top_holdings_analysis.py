#!/usr/bin/env python3
"""
CLEAN TOP HOLDINGS ANALYZER
===========================

Provides a clean, interpretable analysis of the best holdings over the 10-year period
using normalized scoring and meaningful metrics.

Author: GitHub Copilot
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class CleanTopHoldingsAnalyzer:
    """
    Clean analyzer for top holdings with interpretable metrics
    """
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.company_stats = defaultdict(lambda: {
            'appearances': 0,
            'total_weight': 0.0,
            'combined_scores': [],
            'portfolio_weights': [],
            'sectors': set(),
            'position_types': [],
            'monthly_data': []
        })
        
    def load_all_portfolios(self):
        """Load all portfolio JSON files"""
        print("Loading portfolio data...")
        
        portfolio_files = list(self.results_dir.glob("portfolio_*.json"))
        print(f"Found {len(portfolio_files)} portfolio files")
        
        for i, portfolio_file in enumerate(portfolio_files):
            if i % 100 == 0:
                print(f"  Processing: {i+1}/{len(portfolio_files)}")
                
            try:
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                
                filename_parts = portfolio_file.stem.split('_')
                if len(filename_parts) >= 3:
                    year = int(filename_parts[1])
                    month = int(filename_parts[2])
                    sector = '_'.join(filename_parts[3:])
                    
                    # Process positions
                    for position_type in ['long_positions', 'short_positions']:
                        if position_type in portfolio_data:
                            for position in portfolio_data[position_type]:
                                self._process_position(position, year, month, sector, 
                                                     position_type.replace('_positions', ''))
                            
            except Exception as e:
                continue
        
        print(f"Loaded data for {len(self.company_stats)} unique companies")
        
    def _process_position(self, position, year, month, sector, position_type):
        """Process individual position"""
        company_id = position.get('company_id', '')
        if not company_id:
            return
            
        company = self.company_stats[company_id]
        company['appearances'] += 1
        company['sectors'].add(sector)
        company['position_types'].append(position_type)
        
        portfolio_weight = position.get('portfolio_weight', 0.0)
        combined_score = position.get('combined_score', 0.0)
        
        company['total_weight'] += abs(portfolio_weight)
        company['combined_scores'].append(combined_score)
        company['portfolio_weights'].append(portfolio_weight)
        
        company['monthly_data'].append({
            'year': year,
            'month': month,
            'sector': sector,
            'position_type': position_type,
            'portfolio_weight': portfolio_weight,
            'combined_score': combined_score
        })
        
    def calculate_clean_metrics(self):
        """Calculate clean, interpretable metrics"""
        print("Calculating performance metrics...")
        
        for company_id, stats in self.company_stats.items():
            if stats['appearances'] == 0:
                continue
            
            # Basic metrics
            stats['avg_weight'] = stats['total_weight'] / stats['appearances']
            stats['avg_combined_score'] = np.mean(stats['combined_scores']) if stats['combined_scores'] else 0
            stats['max_combined_score'] = np.max(stats['combined_scores']) if stats['combined_scores'] else 0
            stats['min_combined_score'] = np.min(stats['combined_scores']) if stats['combined_scores'] else 0
            
            # Volatility metrics
            if len(stats['combined_scores']) > 1:
                stats['score_std'] = np.std(stats['combined_scores'])
                stats['weight_std'] = np.std([abs(w) for w in stats['portfolio_weights']])
            else:
                stats['score_std'] = 0
                stats['weight_std'] = 0
            
            # Clean risk-adjusted score (handle zero volatility)
            if stats['score_std'] > 1e-10:  # Avoid division by very small numbers
                stats['sharpe_like_ratio'] = stats['avg_combined_score'] / stats['score_std']
            else:
                stats['sharpe_like_ratio'] = stats['avg_combined_score'] * 1000  # Boost for zero volatility
            
            # Performance consistency (0-1 scale)
            stats['consistency'] = 1 / (1 + stats['score_std']) if stats['score_std'] > 0 else 1.0
            
            # Sector and position analysis
            stats['primary_sector'] = max(stats['sectors'], 
                                        key=lambda s: sum(1 for md in stats['monthly_data'] if md['sector'] == s))
            stats['primary_position_type'] = max(set(stats['position_types']), 
                                                key=stats['position_types'].count)
            
            # Time span analysis
            years = [md['year'] for md in stats['monthly_data']]
            stats['first_year'] = min(years)
            stats['last_year'] = max(years)
            stats['time_span'] = stats['last_year'] - stats['first_year'] + 1
            
            # Position type distribution
            long_count = stats['position_types'].count('long')
            short_count = stats['position_types'].count('short')
            stats['long_ratio'] = long_count / stats['appearances'] if stats['appearances'] > 0 else 0
            
            # Create a clean overall score (0-100 scale)
            performance_score = min(stats['avg_combined_score'] * 50, 40)  # Cap at 40 points
            frequency_score = min(stats['appearances'] / 10, 20)            # Cap at 20 points
            consistency_score = stats['consistency'] * 20                   # Max 20 points
            longevity_score = min(stats['time_span'] / 5, 20)              # Cap at 20 points
            
            stats['overall_score'] = performance_score + frequency_score + consistency_score + longevity_score
    
    def get_top_holdings(self, n=10, min_appearances=3):
        """Get top N holdings with clean metrics"""
        companies = []
        
        for company_id, stats in self.company_stats.items():
            if stats['appearances'] >= min_appearances:
                companies.append({
                    'company_id': company_id,
                    'appearances': stats['appearances'],
                    'avg_combined_score': round(stats['avg_combined_score'], 4),
                    'max_combined_score': round(stats['max_combined_score'], 4),
                    'avg_weight': round(stats['avg_weight'], 4),
                    'sharpe_like_ratio': round(stats['sharpe_like_ratio'], 2),
                    'consistency': round(stats['consistency'], 3),
                    'overall_score': round(stats['overall_score'], 2),
                    'primary_sector': stats['primary_sector'],
                    'primary_position_type': stats['primary_position_type'],
                    'long_ratio': round(stats['long_ratio'], 2),
                    'time_span': stats['time_span'],
                    'first_year': stats['first_year'],
                    'last_year': stats['last_year'],
                    'score_std': round(stats['score_std'], 4)
                })
        
        companies_df = pd.DataFrame(companies)
        if len(companies_df) == 0:
            return pd.DataFrame()
        
        # Sort by overall score
        return companies_df.nlargest(n, 'overall_score')
    
    def analyze_by_different_criteria(self, top_n=10):
        """Analyze top holdings by different criteria"""
        companies_df = pd.DataFrame([
            {
                'company_id': company_id,
                'appearances': stats['appearances'],
                'avg_combined_score': stats['avg_combined_score'],
                'sharpe_like_ratio': stats['sharpe_like_ratio'],
                'consistency': stats['consistency'],
                'overall_score': stats['overall_score'],
                'primary_sector': stats['primary_sector'],
                'primary_position_type': stats['primary_position_type']
            }
            for company_id, stats in self.company_stats.items()
            if stats['appearances'] >= 3
        ])
        
        if len(companies_df) == 0:
            return {}
        
        results = {}
        
        # Top by different criteria
        criteria = [
            ('overall_score', 'Overall Performance'),
            ('avg_combined_score', 'Highest Average Score'),
            ('sharpe_like_ratio', 'Best Risk-Adjusted Performance'),
            ('appearances', 'Most Frequently Selected'),
            ('consistency', 'Most Consistent Performance')
        ]
        
        for criterion, title in criteria:
            top_companies = companies_df.nlargest(top_n, criterion)
            results[criterion] = {
                'title': title,
                'companies': top_companies[['company_id', 'primary_sector', criterion, 'appearances']].to_dict('records')
            }
        
        return results
    
    def generate_clean_report(self):
        """Generate clean, readable report"""
        print("\n" + "="*80)
        print("TOP 10 BEST HOLDINGS ANALYSIS")
        print("Period: January 2015 - May 2025")
        print("="*80)
        
        # Get top holdings by different criteria
        analysis_results = self.analyze_by_different_criteria()
        
        for criterion, data in analysis_results.items():
            print(f"\n{data['title'].upper()}:")
            print("-" * 60)
            
            for i, company in enumerate(data['companies'], 1):
                metric_value = company[criterion]
                if isinstance(metric_value, float):
                    metric_str = f"{metric_value:.4f}"
                else:
                    metric_str = str(metric_value)
                    
                print(f"  {i:2d}. {company['company_id']:<20} | {company['primary_sector']:<15} | {metric_str:>8} | ({company['appearances']} times)")
        
        # Detailed top 10 analysis
        print(f"\n\nDETAILED TOP 10 ANALYSIS:")
        print("="*80)
        
        top_10 = self.get_top_holdings(10)
        
        if len(top_10) > 0:
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                company_id = row['company_id']
                stats = self.company_stats[company_id]
                
                print(f"\n{i:2d}. {company_id}")
                print(f"    Overall Score: {row['overall_score']:.2f}/100")
                print(f"    Primary Sector: {row['primary_sector']}")
                print(f"    Position Preference: {row['primary_position_type']} ({row['long_ratio']:.0%} long)")
                print(f"    Selection Frequency: {row['appearances']} times")
                print(f"    Time Period: {row['first_year']}-{row['last_year']} ({row['time_span']} years)")
                print(f"    Average Combined Score: {row['avg_combined_score']:.4f}")
                print(f"    Peak Performance: {row['max_combined_score']:.4f}")
                print(f"    Average Portfolio Weight: {row['avg_weight']:.4f}")
                print(f"    Risk-Adjusted Performance: {row['sharpe_like_ratio']:.2f}")
                print(f"    Consistency Score: {row['consistency']:.3f}")
                print(f"    Score Volatility: {row['score_std']:.4f}")
                
                # Show sector breakdown
                sector_counts = Counter(md['sector'] for md in stats['monthly_data'])
                top_sectors = sector_counts.most_common(3)
                sectors_str = ', '.join([f"{s}({c})" for s, c in top_sectors])
                print(f"    Top Sectors: {sectors_str}")
        
        return top_10
    
    def save_clean_results(self, top_holdings):
        """Save clean results"""
        if len(top_holdings) > 0:
            output_path = self.results_dir / "top_10_best_holdings.csv"
            top_holdings.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            return top_holdings
        
        return pd.DataFrame()

def main():
    """Main execution"""
    print("Starting Clean Top Holdings Analysis...")
    
    analyzer = CleanTopHoldingsAnalyzer()
    analyzer.load_all_portfolios()
    analyzer.calculate_clean_metrics()
    
    top_holdings = analyzer.generate_clean_report()
    analyzer.save_clean_results(top_holdings)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()