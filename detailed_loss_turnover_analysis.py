#!/usr/bin/env python3
"""
DETAILED PORTFOLIO LOSS AND TURNOVER ANALYSIS
==============================================

Analyzes actual portfolio returns and composition data to provide precise
maximum loss and turnover calculations.

Author: GitHub Copilot
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_portfolio_losses_and_turnover():
    """Analyze maximum losses and portfolio turnover from actual data"""
    
    results_dir = Path("results")
    
    print("="*80)
    print("DETAILED PORTFOLIO LOSS AND TURNOVER ANALYSIS")
    print("="*80)
    
    # 1. ANALYZE MAXIMUM MONTHLY LOSSES
    print("\n1. ANALYZING MAXIMUM MONTHLY LOSSES")
    print("-" * 50)
    
    # Load portfolio returns from backtesting
    portfolio_returns_file = results_dir / "backtesting" / "portfolio_returns.csv"
    
    if portfolio_returns_file.exists():
        print(f"Loading data from: {portfolio_returns_file}")
        df_returns = pd.read_csv(portfolio_returns_file)
        df_returns['date'] = pd.to_datetime(df_returns['date'])
        
        # Find maximum monthly loss
        min_return = df_returns['portfolio_return'].min()
        max_return = df_returns['portfolio_return'].max()
        
        # Find the specific month with maximum loss
        worst_month = df_returns.loc[df_returns['portfolio_return'].idxmin()]
        best_month = df_returns.loc[df_returns['portfolio_return'].idxmax()]
        
        print(f"MAXIMUM ONE-MONTH LOSS: {min_return*100:.3f}%")
        print(f"   Date: {worst_month['date'].strftime('%B %Y')}")
        print(f"   Long return: {worst_month['long_return']*100:.3f}%")
        print(f"   Short return: {worst_month['short_return']*100:.3f}%")
        print(f"   Positions: {worst_month['positions_count']}")
        
        print(f"\nMaximum one-month gain: {max_return*100:.3f}%")
        print(f"   Date: {best_month['date'].strftime('%B %Y')}")
        
        # Calculate additional loss statistics
        negative_returns = df_returns[df_returns['portfolio_return'] < 0]
        print(f"\nLoss Statistics:")
        print(f"   Number of negative months: {len(negative_returns)}")
        print(f"   Percentage of negative months: {len(negative_returns)/len(df_returns)*100:.1f}%")
        
        if len(negative_returns) > 0:
            print(f"   Average loss in negative months: {negative_returns['portfolio_return'].mean()*100:.3f}%")
            print(f"   Median loss in negative months: {negative_returns['portfolio_return'].median()*100:.3f}%")
            
            print(f"\nWorst 5 months:")
            worst_5 = df_returns.nsmallest(5, 'portfolio_return')
            for i, (_, row) in enumerate(worst_5.iterrows(), 1):
                print(f"   {i}. {row['date'].strftime('%b %Y')}: {row['portfolio_return']*100:.3f}%")
        
        # Value at Risk calculations
        var_5 = np.percentile(df_returns['portfolio_return'], 5)
        var_1 = np.percentile(df_returns['portfolio_return'], 1)
        
        print(f"\nValue at Risk:")
        print(f"   VaR 5%: {var_5*100:.3f}% (worst 5% of months)")
        print(f"   VaR 1%: {var_1*100:.3f}% (worst 1% of months)")
        
        # Maximum drawdown analysis
        cumulative_returns = (1 + df_returns['portfolio_return']).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        print(f"\nDrawdown Analysis:")
        print(f"   Maximum drawdown: {max_drawdown*100:.3f}%")
        
        if max_drawdown < 0:
            max_dd_date = df_returns.loc[drawdowns.idxmin(), 'date']
            print(f"   Max drawdown date: {max_dd_date.strftime('%B %Y')}")
    
    else:
        print(f"Portfolio returns file not found: {portfolio_returns_file}")
    
    # 2. ANALYZE PORTFOLIO TURNOVER
    print(f"\n2. ANALYZING PORTFOLIO TURNOVER")
    print("-" * 50)
    
    # Load portfolio compositions to calculate turnover
    portfolio_files = sorted(list(results_dir.glob("portfolio_*.json")))
    
    if len(portfolio_files) > 0:
        print(f"Analyzing {len(portfolio_files)} portfolio files...")
        
        monthly_compositions = {}
        
        # Load all portfolio compositions
        for portfolio_file in portfolio_files:
            try:
                filename_parts = portfolio_file.stem.split('_')
                if len(filename_parts) >= 3:
                    year = int(filename_parts[1])
                    month = int(filename_parts[2])
                    sector = '_'.join(filename_parts[3:])
                    
                    date_key = f"{year}-{month:02d}"
                    
                    if date_key not in monthly_compositions:
                        monthly_compositions[date_key] = {
                            'all_positions': set(),
                            'long_positions': set(),
                            'short_positions': set(),
                            'year': year,
                            'month': month
                        }
                    
                    with open(portfolio_file, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    # Extract positions
                    if 'long_positions' in portfolio_data:
                        for pos in portfolio_data['long_positions']:
                            company_id = pos.get('company_id', '')
                            if company_id:
                                monthly_compositions[date_key]['long_positions'].add(company_id)
                                monthly_compositions[date_key]['all_positions'].add(f"L_{company_id}")
                    
                    if 'short_positions' in portfolio_data:
                        for pos in portfolio_data['short_positions']:
                            company_id = pos.get('company_id', '')
                            if company_id:
                                monthly_compositions[date_key]['short_positions'].add(company_id)
                                monthly_compositions[date_key]['all_positions'].add(f"S_{company_id}")
                                
            except Exception as e:
                continue
        
        # Calculate monthly turnover
        sorted_dates = sorted(monthly_compositions.keys())
        turnover_data = []
        
        for i in range(1, len(sorted_dates)):
            current_date = sorted_dates[i]
            previous_date = sorted_dates[i-1]
            
            current_positions = monthly_compositions[current_date]['all_positions']
            previous_positions = monthly_compositions[previous_date]['all_positions']
            
            # Calculate turnover
            if len(previous_positions) > 0:
                positions_exited = previous_positions - current_positions
                positions_entered = current_positions - previous_positions
                
                total_changes = len(positions_exited) + len(positions_entered)
                average_positions = (len(current_positions) + len(previous_positions)) / 2
                
                turnover_rate = total_changes / average_positions if average_positions > 0 else 0
            else:
                turnover_rate = 1.0 if len(current_positions) > 0 else 0.0
            
            turnover_data.append({
                'date': current_date,
                'turnover_rate': turnover_rate,
                'positions_entered': len(positions_entered) if 'positions_entered' in locals() else 0,
                'positions_exited': len(positions_exited) if 'positions_exited' in locals() else 0,
                'total_positions': len(current_positions),
                'net_change': len(positions_entered) - len(positions_exited) if 'positions_entered' in locals() and 'positions_exited' in locals() else 0
            })
        
        # Analyze turnover statistics
        if turnover_data:
            turnover_df = pd.DataFrame(turnover_data)
            
            avg_turnover = turnover_df['turnover_rate'].mean()
            median_turnover = turnover_df['turnover_rate'].median()
            max_turnover = turnover_df['turnover_rate'].max()
            min_turnover = turnover_df['turnover_rate'].min()
            
            print(f"PORTFOLIO TURNOVER ANALYSIS:")
            print(f"   Average monthly turnover: {avg_turnover*100:.1f}%")
            print(f"   ANNUAL TURNOVER: {avg_turnover*12*100:.1f}%")
            print(f"   Median monthly turnover: {median_turnover*100:.1f}%")
            print(f"   Maximum monthly turnover: {max_turnover*100:.1f}%")
            print(f"   Minimum monthly turnover: {min_turnover*100:.1f}%")
            
            # Find months with highest turnover
            high_turnover_months = turnover_df.nlargest(5, 'turnover_rate')
            print(f"\nHighest turnover months:")
            for i, (_, row) in enumerate(high_turnover_months.iterrows(), 1):
                print(f"   {i}. {row['date']}: {row['turnover_rate']*100:.1f}% turnover")
                print(f"      Entered: {row['positions_entered']}, Exited: {row['positions_exited']}")
            
            # Average portfolio size
            avg_positions = turnover_df['total_positions'].mean()
            print(f"\nPortfolio size:")
            print(f"   Average positions per month: {avg_positions:.1f}")
            print(f"   Position range: {turnover_df['total_positions'].min()}-{turnover_df['total_positions'].max()}")
            
            # Turnover interpretation
            annual_turnover = avg_turnover * 12
            print(f"\nTURNOVER INTERPRETATION:")
            if annual_turnover > 5:
                print(f"   Very high turnover ({annual_turnover*100:.0f}% annually)")
                print(f"   Indicates frequent rebalancing and active management")
            elif annual_turnover > 3:
                print(f"   High turnover ({annual_turnover*100:.0f}% annually)")  
                print(f"   Active portfolio management with regular rebalancing")
            elif annual_turnover > 2:
                print(f"   Moderate turnover ({annual_turnover*100:.0f}% annually)")
                print(f"   Balanced approach between buy-and-hold and active trading")
            else:
                print(f"   Low turnover ({annual_turnover*100:.0f}% annually)")
                print(f"   More buy-and-hold approach with infrequent changes")
    
    print(f"\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    
    if 'min_return' in locals():
        print(f"MAXIMUM ONE-MONTH LOSS: {min_return*100:.3f}%")
        if abs(min_return) < 0.01:  # Less than 1%
            print(f"   Very low maximum loss indicates strong risk management")
        elif abs(min_return) < 0.05:  # Less than 5%
            print(f"   Moderate maximum loss within acceptable range")
        else:
            print(f"   High maximum loss indicates significant risk exposure")
    
    if 'annual_turnover' in locals():
        print(f"ANNUAL PORTFOLIO TURNOVER: {annual_turnover*100:.1f}%")
        if annual_turnover > 4:
            print(f"   Very high turnover may indicate overtrading")
        elif annual_turnover > 2:
            print(f"   High turnover indicates active management")
        else:
            print(f"   Moderate turnover suggests disciplined approach")

if __name__ == "__main__":
    analyze_portfolio_losses_and_turnover()