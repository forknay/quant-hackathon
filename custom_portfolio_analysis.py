#!/usr/bin/env python3
"""
PORTFOLIO ANALYSIS WITH CUSTOM DATA
==================================

This script analyzes portfolio JSON files using only the internal data
without relying on external stock price feeds. It focuses on:

1. Portfolio construction quality metrics
2. Algorithmic scoring analysis  
3. ML model performance evaluation
4. Sector and time-based comparisons
5. Risk distribution analysis

Author: GitHub Copilot
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
import statistics

warnings.filterwarnings('ignore')

class CustomPortfolioAnalyzer:
    """
    Portfolio analyzer using only custom internal data
    """
    
    def __init__(self, portfolio_dir: str = "results"):
        """Initialize the analyzer"""
        self.portfolio_dir = Path(portfolio_dir)
        self.portfolios = []
        
    def discover_portfolios(self) -> List[Dict]:
        """Discover all portfolio JSON files in the results directory"""
        portfolio_files = list(self.portfolio_dir.glob("portfolio_*.json"))
        portfolios = []
        
        print(f"📁 Discovering portfolios in {self.portfolio_dir}")
        print(f"   Found {len(portfolio_files)} portfolio files")
        
        for file_path in sorted(portfolio_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    portfolio_data = json.load(f)
                
                # Parse filename to extract metadata
                filename = file_path.stem  # e.g., "portfolio_2023_01_energy"
                parts = filename.split('_')
                if len(parts) >= 4:
                    year = int(parts[1])
                    month = int(parts[2])
                    sector = '_'.join(parts[3:])
                    
                    portfolio_info = {
                        'file_path': file_path,
                        'year': year,
                        'month': month,
                        'sector': sector,
                        'date': datetime(year, month, 1),
                        'data': portfolio_data
                    }
                    portfolios.append(portfolio_info)
                    
                    long_pos = len(portfolio_data.get('long_positions', []))
                    short_pos = len(portfolio_data.get('short_positions', []))
                    print(f"   ✓ {year}-{month:02d} {sector}: {long_pos} long + {short_pos} short")
                    
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")
                
        print(f"✅ Discovered {len(portfolios)} valid portfolios")
        return portfolios
    
    def analyze_portfolio_construction(self, portfolio: Dict) -> Dict:
        """Analyze portfolio construction quality using internal data"""
        data = portfolio['data']
        long_positions = data.get('long_positions', [])
        short_positions = data.get('short_positions', [])
        
        analysis = {
            'portfolio_id': f"{portfolio['year']}-{portfolio['month']:02d}_{portfolio['sector']}",
            'num_long': len(long_positions),
            'num_short': len(short_positions),
            'total_positions': len(long_positions) + len(short_positions),
        }
        
        # Analyze long positions
        if long_positions:
            long_weights = [pos['portfolio_weight'] for pos in long_positions]
            long_algo_scores = [pos['algo_score_raw'] for pos in long_positions]
            long_ml_scores = [pos['ml_score_raw'] for pos in long_positions]
            long_combined = [pos['combined_score'] for pos in long_positions]
            
            analysis.update({
                'long_weight_sum': sum(long_weights),
                'long_weight_concentration': max(long_weights) if long_weights else 0,
                'long_algo_score_mean': statistics.mean(long_algo_scores),
                'long_algo_score_std': statistics.stdev(long_algo_scores) if len(long_algo_scores) > 1 else 0,
                'long_ml_score_mean': statistics.mean(long_ml_scores),
                'long_ml_score_std': statistics.stdev(long_ml_scores) if len(long_ml_scores) > 1 else 0,
                'long_combined_mean': statistics.mean(long_combined),
                'long_combined_std': statistics.stdev(long_combined) if len(long_combined) > 1 else 0,
            })
        
        # Analyze short positions
        if short_positions:
            short_weights = [abs(pos['portfolio_weight']) for pos in short_positions]
            short_algo_scores = [pos['algo_score_raw'] for pos in short_positions]
            short_ml_scores = [pos['ml_score_raw'] for pos in short_positions]
            short_combined = [pos['combined_score'] for pos in short_positions]
            
            analysis.update({
                'short_weight_sum': sum(short_weights),
                'short_weight_concentration': max(short_weights) if short_weights else 0,
                'short_algo_score_mean': statistics.mean(short_algo_scores),
                'short_algo_score_std': statistics.stdev(short_algo_scores) if len(short_algo_scores) > 1 else 0,
                'short_ml_score_mean': statistics.mean(short_ml_scores),
                'short_ml_score_std': statistics.stdev(short_ml_scores) if len(short_ml_scores) > 1 else 0,
                'short_combined_mean': statistics.mean(short_combined),
                'short_combined_std': statistics.stdev(short_combined) if len(short_combined) > 1 else 0,
            })
        
        # Portfolio balance metrics
        analysis.update({
            'weight_balance': abs(analysis.get('long_weight_sum', 0) - analysis.get('short_weight_sum', 0)),
            'position_balance': abs(analysis['num_long'] - analysis['num_short']) / max(analysis['num_long'] + analysis['num_short'], 1),
        })
        
        return analysis
    
    def analyze_scoring_quality(self, portfolios: List[Dict]) -> Dict:
        """Analyze the quality of algorithmic and ML scoring across portfolios"""
        all_long_algo = []
        all_long_ml = []
        all_short_algo = []
        all_short_ml = []
        
        for portfolio in portfolios:
            data = portfolio['data']
            
            # Collect long position scores
            for pos in data.get('long_positions', []):
                all_long_algo.append(pos['algo_score_raw'])
                all_long_ml.append(pos['ml_score_raw'])
            
            # Collect short position scores
            for pos in data.get('short_positions', []):
                all_short_algo.append(pos['algo_score_raw'])
                all_short_ml.append(pos['ml_score_raw'])
        
        analysis = {}
        
        if all_long_algo and all_short_algo:
            # Analyze score separation (higher scores for long, lower for short is good)
            analysis.update({
                'long_algo_mean': statistics.mean(all_long_algo),
                'short_algo_mean': statistics.mean(all_short_algo),
                'algo_separation': statistics.mean(all_long_algo) - statistics.mean(all_short_algo),
                'algo_separation_ratio': statistics.mean(all_long_algo) / statistics.mean(all_short_algo) if statistics.mean(all_short_algo) != 0 else float('inf'),
            })
        
        if all_long_ml and all_short_ml:
            analysis.update({
                'long_ml_mean': statistics.mean(all_long_ml),
                'short_ml_mean': statistics.mean(all_short_ml),
                'ml_separation': statistics.mean(all_long_ml) - statistics.mean(all_short_ml),
                'ml_separation_ratio': statistics.mean(all_long_ml) / statistics.mean(all_short_ml) if statistics.mean(all_short_ml) != 0 else float('inf'),
            })
        
        # Score consistency analysis
        if len(all_long_algo) > 1:
            analysis['long_algo_consistency'] = 1 / (statistics.stdev(all_long_algo) + 1e-8)
        if len(all_short_algo) > 1:
            analysis['short_algo_consistency'] = 1 / (statistics.stdev(all_short_algo) + 1e-8)
        if len(all_long_ml) > 1:
            analysis['long_ml_consistency'] = 1 / (statistics.stdev(all_long_ml) + 1e-8)
        if len(all_short_ml) > 1:
            analysis['short_ml_consistency'] = 1 / (statistics.stdev(all_short_ml) + 1e-8)
        
        return analysis
    
    def analyze_temporal_patterns(self, portfolios: List[Dict]) -> Dict:
        """Analyze patterns across time periods"""
        # Group by time periods
        by_year = {}
        by_month = {}
        by_sector = {}
        
        for portfolio in portfolios:
            year = portfolio['year']
            month = portfolio['month']
            sector = portfolio['sector']
            
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(portfolio)
            
            if month not in by_month:
                by_month[month] = []
            by_month[month].append(portfolio)
            
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(portfolio)
        
        analysis = {
            'years_covered': sorted(by_year.keys()),
            'months_covered': sorted(by_month.keys()),
            'sectors_covered': sorted(by_sector.keys()),
            'portfolios_per_year': {year: len(portfolios) for year, portfolios in by_year.items()},
            'portfolios_per_month': {month: len(portfolios) for month, portfolios in by_month.items()},
            'portfolios_per_sector': {sector: len(portfolios) for sector, portfolios in by_sector.items()},
        }
        
        # Calculate average positions by grouping
        for group_name, group_data in [('year', by_year), ('month', by_month), ('sector', by_sector)]:
            avg_positions = {}
            for key, group_portfolios in group_data.items():
                total_positions = sum(len(p['data'].get('long_positions', [])) + len(p['data'].get('short_positions', [])) 
                                    for p in group_portfolios)
                avg_positions[key] = total_positions / len(group_portfolios) if group_portfolios else 0
            analysis[f'avg_positions_per_{group_name}'] = avg_positions
        
        return analysis
    
    def calculate_comprehensive_metrics(self, portfolio_analyses: List[Dict]) -> Dict:
        """Calculate comprehensive metrics from portfolio analyses"""
        if not portfolio_analyses:
            return {}
        
        # Portfolio construction quality
        total_positions = sum(p['total_positions'] for p in portfolio_analyses)
        avg_positions = total_positions / len(portfolio_analyses)
        
        # Weight balance quality (closer to 0 is better)
        weight_balances = [p['weight_balance'] for p in portfolio_analyses]
        avg_weight_balance = statistics.mean(weight_balances)
        
        # Position balance quality (closer to 0 is better)
        position_balances = [p['position_balance'] for p in portfolio_analyses]
        avg_position_balance = statistics.mean(position_balances)
        
        # Score quality metrics
        long_algo_means = [p.get('long_algo_score_mean', 0) for p in portfolio_analyses if 'long_algo_score_mean' in p]
        short_algo_means = [p.get('short_algo_score_mean', 0) for p in portfolio_analyses if 'short_algo_score_mean' in p]
        
        long_ml_means = [p.get('long_ml_score_mean', 0) for p in portfolio_analyses if 'long_ml_score_mean' in p]
        short_ml_means = [p.get('short_ml_score_mean', 0) for p in portfolio_analyses if 'short_ml_score_mean' in p]
        
        metrics = {
            'total_portfolios': len(portfolio_analyses),
            'avg_positions_per_portfolio': avg_positions,
            'avg_weight_balance_error': avg_weight_balance,
            'avg_position_balance_error': avg_position_balance,
            'portfolio_construction_quality': 1 / (1 + avg_weight_balance + avg_position_balance),  # Higher is better
        }
        
        if long_algo_means and short_algo_means:
            metrics.update({
                'avg_long_algo_score': statistics.mean(long_algo_means),
                'avg_short_algo_score': statistics.mean(short_algo_means),
                'algo_score_separation': statistics.mean(long_algo_means) - statistics.mean(short_algo_means),
            })
        
        if long_ml_means and short_ml_means:
            metrics.update({
                'avg_long_ml_score': statistics.mean(long_ml_means),
                'avg_short_ml_score': statistics.mean(short_ml_means),
                'ml_score_separation': statistics.mean(long_ml_means) - statistics.mean(short_ml_means),
            })
        
        return metrics
    
    def generate_report(self, metrics: Dict, scoring_analysis: Dict, temporal_analysis: Dict, 
                       portfolio_analyses: List[Dict]) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("📊 COMPREHENSIVE PORTFOLIO ANALYSIS REPORT")
        report.append("   (Based on Custom Algorithm and ML Data)")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("🎯 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"📈 Total Portfolios Analyzed:        {metrics.get('total_portfolios', 0)}")
        report.append(f"📊 Average Positions per Portfolio:  {metrics.get('avg_positions_per_portfolio', 0):.1f}")
        report.append(f"⚖️  Portfolio Construction Quality:   {metrics.get('portfolio_construction_quality', 0):.3f}")
        report.append(f"🎯 Algorithm Score Separation:       {metrics.get('algo_score_separation', 0):.4f}")
        report.append(f"🧠 ML Score Separation:              {metrics.get('ml_score_separation', 0):.4f}")
        report.append("")
        
        # Temporal Coverage
        report.append("📅 TEMPORAL COVERAGE")
        report.append("-" * 40)
        report.append(f"Years Covered: {', '.join(map(str, temporal_analysis.get('years_covered', [])))}")
        report.append(f"Months Covered: {', '.join(map(str, temporal_analysis.get('months_covered', [])))}")
        report.append(f"Sectors Covered: {', '.join(temporal_analysis.get('sectors_covered', []))}")
        report.append("")
        
        # Portfolio Construction Analysis
        report.append("🏗️  PORTFOLIO CONSTRUCTION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average Weight Balance Error:    {metrics.get('avg_weight_balance_error', 0):.4f}")
        report.append(f"Average Position Balance Error:  {metrics.get('avg_position_balance_error', 0):.4f}")
        report.append("(Lower values indicate better balance)")
        report.append("")
        
        # Scoring System Analysis
        report.append("🔢 SCORING SYSTEM ANALYSIS")
        report.append("-" * 40)
        if 'avg_long_algo_score' in metrics:
            report.append(f"Algorithm Scores:")
            report.append(f"  Long Positions Average:   {metrics['avg_long_algo_score']:.4f}")
            report.append(f"  Short Positions Average:  {metrics['avg_short_algo_score']:.4f}")
            report.append(f"  Separation (Long-Short):  {metrics['algo_score_separation']:.4f}")
            report.append("")
        
        if 'avg_long_ml_score' in metrics:
            report.append(f"ML Model Scores:")
            report.append(f"  Long Positions Average:   {metrics['avg_long_ml_score']:.4f}")
            report.append(f"  Short Positions Average:  {metrics['avg_short_ml_score']:.4f}")
            report.append(f"  Separation (Long-Short):  {metrics['ml_score_separation']:.4f}")
            report.append("")
        
        # Sector Distribution
        report.append("🏭 SECTOR DISTRIBUTION")
        report.append("-" * 40)
        sector_counts = temporal_analysis.get('portfolios_per_sector', {})
        for sector, count in sorted(sector_counts.items()):
            avg_pos = temporal_analysis.get('avg_positions_per_sector', {}).get(sector, 0)
            report.append(f"{sector:20}: {count:2d} portfolios, {avg_pos:.1f} avg positions")
        report.append("")
        
        # Time Distribution
        report.append("📆 TIME DISTRIBUTION")
        report.append("-" * 40)
        year_counts = temporal_analysis.get('portfolios_per_year', {})
        for year, count in sorted(year_counts.items()):
            report.append(f"Year {year}: {count} portfolios")
        report.append("")
        
        month_counts = temporal_analysis.get('portfolios_per_month', {})
        for month, count in sorted(month_counts.items()):
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
            report.append(f"Month {month:2d} ({month_name}): {count} portfolios")
        report.append("")
        
        # Individual Portfolio Details
        report.append("📁 INDIVIDUAL PORTFOLIO DETAILS")
        report.append("-" * 40)
        for analysis in sorted(portfolio_analyses, key=lambda x: x['portfolio_id']):
            report.append(f"{analysis['portfolio_id']:20} | "
                         f"Pos: {analysis['total_positions']:2d} "
                         f"({analysis['num_long']:2d}L/{analysis['num_short']:2d}S) | "
                         f"Wgt Bal: {analysis.get('weight_balance', 0):.3f} | "
                         f"Pos Bal: {analysis.get('position_balance', 0):.3f}")
        report.append("")
        
        # Quality Assessment
        report.append("✅ QUALITY ASSESSMENT")
        report.append("-" * 40)
        construction_quality = metrics.get('portfolio_construction_quality', 0)
        if construction_quality > 0.8:
            report.append("🟢 Portfolio Construction: EXCELLENT")
        elif construction_quality > 0.6:
            report.append("🟡 Portfolio Construction: GOOD")
        else:
            report.append("🔴 Portfolio Construction: NEEDS IMPROVEMENT")
        
        algo_sep = abs(metrics.get('algo_score_separation', 0))
        if algo_sep > 0.1:
            report.append("🟢 Algorithm Score Separation: EXCELLENT")
        elif algo_sep > 0.05:
            report.append("🟡 Algorithm Score Separation: GOOD") 
        else:
            report.append("🔴 Algorithm Score Separation: NEEDS IMPROVEMENT")
        
        ml_sep = abs(metrics.get('ml_score_separation', 0))
        if ml_sep > 5.0:
            report.append("🟢 ML Score Separation: EXCELLENT")
        elif ml_sep > 1.0:
            report.append("🟡 ML Score Separation: GOOD")
        else:
            report.append("🔴 ML Score Separation: NEEDS IMPROVEMENT")
        report.append("")
        
        # Methodology Notes
        report.append("📝 METHODOLOGY NOTES")
        report.append("-" * 40)
        report.append("• Analysis based on internal portfolio construction data")
        report.append("• Scores from algorithmic indicators and ML model predictions")
        report.append("• Portfolio quality measured by position balance and weight distribution")
        report.append("• Score separation indicates model effectiveness (higher = better)")
        report.append("• Long positions should have higher scores than short positions")
        report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def run_analysis(self) -> Dict:
        """Run the complete custom portfolio analysis"""
        print("🚀 STARTING CUSTOM PORTFOLIO ANALYSIS")
        print("=" * 50)
        
        # Discover portfolios
        self.portfolios = self.discover_portfolios()
        if not self.portfolios:
            print("❌ No portfolios found")
            return {}
        
        # Analyze each portfolio
        print(f"\n🔍 ANALYZING PORTFOLIO CONSTRUCTION")
        print("-" * 30)
        portfolio_analyses = []
        for portfolio in self.portfolios:
            analysis = self.analyze_portfolio_construction(portfolio)
            portfolio_analyses.append(analysis)
            print(f"   ✓ {analysis['portfolio_id']}: {analysis['total_positions']} positions")
        
        # Analyze scoring quality
        print(f"\n🎯 ANALYZING SCORING SYSTEMS")
        print("-" * 30)
        scoring_analysis = self.analyze_scoring_quality(self.portfolios)
        print(f"   ✓ Algorithm score separation: {scoring_analysis.get('algo_separation', 0):.4f}")
        print(f"   ✓ ML score separation: {scoring_analysis.get('ml_separation', 0):.4f}")
        
        # Analyze temporal patterns
        print(f"\n📊 ANALYZING TEMPORAL PATTERNS")
        print("-" * 30)
        temporal_analysis = self.analyze_temporal_patterns(self.portfolios)
        print(f"   ✓ {len(temporal_analysis['years_covered'])} years, "
              f"{len(temporal_analysis['sectors_covered'])} sectors")
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(portfolio_analyses)
        
        # Generate and display report
        report = self.generate_report(metrics, scoring_analysis, temporal_analysis, portfolio_analyses)
        print(f"\n{report}")
        
        # Save report to file
        output_file = self.portfolio_dir / "custom_portfolio_analysis_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📁 Report saved to: {output_file}")
        
        return {
            'metrics': metrics,
            'scoring_analysis': scoring_analysis, 
            'temporal_analysis': temporal_analysis,
            'portfolio_analyses': portfolio_analyses
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom Portfolio Analysis")
    parser.add_argument("--portfolio-dir", default="results", 
                       help="Directory containing portfolio JSON files")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = CustomPortfolioAnalyzer(portfolio_dir=args.portfolio_dir)
    results = analyzer.run_analysis()
    
    return results

if __name__ == "__main__":
    main()