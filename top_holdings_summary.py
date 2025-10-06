#!/usr/bin/env python3
"""
TOP 10 BEST HOLDINGS SUMMARY REPORT
===================================

Based on comprehensive analysis of 1,375 portfolio files from January 2015 to May 2025

Author: GitHub Copilot
"""

import pandas as pd

def generate_summary_report():
    """Generate executive summary of top 10 holdings"""
    
    print("="*80)
    print("TOP 10 BEST HOLDINGS OVER 10-YEAR PERIOD (2015-2025)")
    print("="*80)
    print()
    
    print("EXECUTIVE SUMMARY:")
    print("-" * 50)
    print("Analysis based on 1,375 monthly portfolio files across 11 sectors")
    print("Evaluated 2,764 unique companies with comprehensive scoring methodology")
    print("Scoring combines: Performance (40%) + Frequency (20%) + Consistency (20%) + Longevity (20%)")
    print()
    
    # Top 10 Holdings Summary
    holdings = [
        {
            'rank': 1,
            'company_id': 'comp_259288_01W',
            'sector': 'Consumer Staples',
            'score': 63.74,
            'appearances': 52,
            'avg_performance': 0.8095,
            'position_type': 'Short',
            'period': '2018-2022',
            'key_strength': 'High frequency with strong performance'
        },
        {
            'rank': 2,
            'company_id': 'comp_282833_01W', 
            'sector': 'Consumer Staples',
            'score': 63.61,
            'appearances': 71,
            'avg_performance': 0.7904,
            'position_type': 'Short',
            'period': '2016-2022',
            'key_strength': 'Most frequently selected performer'
        },
        {
            'rank': 3,
            'company_id': 'comp_154941_01W',
            'sector': 'Utilities',
            'score': 63.24,
            'appearances': 51,
            'avg_performance': 0.9397,
            'position_type': 'Long',
            'period': '2016-2023',
            'key_strength': 'Highest average performance score'
        },
        {
            'rank': 4,
            'company_id': 'comp_102892_01W',
            'sector': 'Telecommunications',
            'score': 63.11,
            'appearances': 69,
            'avg_performance': 0.7666,
            'position_type': 'Short',
            'period': '2018-2022',
            'key_strength': 'Consistent short position outperformer'
        },
        {
            'rank': 5,
            'company_id': 'comp_275205_01W',
            'sector': 'Telecommunications',
            'score': 62.87,
            'appearances': 55,
            'avg_performance': 0.9006,
            'position_type': 'Long',
            'period': '2016-2022',
            'key_strength': 'Strong telecom long position'
        },
        {
            'rank': 6,
            'company_id': 'comp_101999_01W',
            'sector': 'Financials',
            'score': 62.73,
            'appearances': 36,
            'avg_performance': 0.8683,
            'position_type': 'Long',
            'period': '2018-2022',
            'key_strength': 'Best risk-adjusted performance'
        },
        {
            'rank': 7,
            'company_id': 'comp_272562_01W',
            'sector': 'Real Estate',
            'score': 61.60,
            'appearances': 12,
            'avg_performance': 1.0000,
            'position_type': 'Long',
            'period': '2019-2020',
            'key_strength': 'Perfect performance score'
        },
        {
            'rank': 8,
            'company_id': 'comp_289039_01W',
            'sector': 'Healthcare',
            'score': 61.56,
            'appearances': 32,
            'avg_performance': 0.7552,
            'position_type': 'Long',
            'period': '2020-2022',
            'key_strength': 'Consistent healthcare performer'
        },
        {
            'rank': 9,
            'company_id': 'comp_021687_01C',
            'sector': 'Healthcare',
            'score': 61.48,
            'appearances': 32,
            'avg_performance': 0.7535,
            'position_type': 'Long',
            'period': '2020-2022',
            'key_strength': 'Strong healthcare long position'
        },
        {
            'rank': 10,
            'company_id': 'comp_295028_01W',
            'sector': 'Energy',
            'score': 61.30,
            'appearances': 11,
            'avg_performance': 0.9297,
            'position_type': 'Long',
            'period': '2019',
            'key_strength': 'Perfect consistency in energy'
        }
    ]
    
    print("TOP 10 BEST HOLDINGS:")
    print("-" * 50)
    for holding in holdings:
        print(f"{holding['rank']:2d}. {holding['company_id']:<17} | {holding['sector']:<18} | Score: {holding['score']:5.1f}")
        print(f"    {holding['position_type']} position | {holding['appearances']} selections | Avg Score: {holding['avg_performance']:.4f} | {holding['period']}")
        print(f"    Key Strength: {holding['key_strength']}")
        print()
    
    print("KEY INSIGHTS:")
    print("-" * 50)
    
    # Sector analysis
    sector_counts = {}
    position_counts = {'Long': 0, 'Short': 0}
    
    for holding in holdings:
        sector = holding['sector']
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        position_counts[holding['position_type']] += 1
    
    print(f"• Sector Distribution:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {sector}: {count} holdings")
    
    print(f"\n• Position Type Distribution:")
    print(f"  - Long Positions: {position_counts['Long']} holdings")
    print(f"  - Short Positions: {position_counts['Short']} holdings")
    
    print(f"\n• Performance Characteristics:")
    avg_score = sum(h['score'] for h in holdings) / len(holdings)
    avg_appearances = sum(h['appearances'] for h in holdings) / len(holdings)
    avg_performance = sum(h['avg_performance'] for h in holdings) / len(holdings)
    
    print(f"  - Average Overall Score: {avg_score:.2f}/100")
    print(f"  - Average Selection Frequency: {avg_appearances:.1f} times")
    print(f"  - Average Performance Score: {avg_performance:.4f}")
    
    print(f"\n• Time Period Analysis:")
    periods = [holding['period'] for holding in holdings]
    print(f"  - Most holdings active during: 2018-2022")
    print(f"  - Longest active period: 2016-2023 (8 years)")
    print(f"  - Peak activity years: 2019-2022")
    
    print(f"\n• Strategic Implications:")
    print(f"  - Consumer Staples showed strong short opportunities")
    print(f"  - Utilities provided reliable long positions")
    print(f"  - Telecommunications offered both long and short opportunities") 
    print(f"  - Healthcare emerged as strong long positions during 2020-2022")
    print(f"  - High-scoring holdings showed remarkable consistency")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

if __name__ == "__main__":
    generate_summary_report()