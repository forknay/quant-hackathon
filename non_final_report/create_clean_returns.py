#!/usr/bin/env python3
"""
Create a clean CSV with just the essential monthly returns data
"""

import pandas as pd
from pathlib import Path

# Load the comprehensive data
input_file = Path("results/mixed_sector_monthly_returns.csv")
df = pd.read_csv(input_file)

# Create clean version with essential columns
clean_df = df[['date', 'year', 'month', 'portfolio_return']].copy()

# Rename column for clarity
clean_df = clean_df.rename(columns={
    'portfolio_return': 'monthly_return'
})

# Add percentage format column for easy reading
clean_df['monthly_return_pct'] = (clean_df['monthly_return'] * 100).round(4)

# Save clean version
output_file = Path("results/monthly_portfolio_returns_2015_2025.csv")
clean_df.to_csv(output_file, index=False)

print(f"Clean monthly returns CSV created: {output_file}")
print(f"Columns: {list(clean_df.columns)}")
print(f"Period: {clean_df['date'].min()} to {clean_df['date'].max()}")
print(f"Total months: {len(clean_df)}")

# Show first few rows
print("\nFirst 5 rows:")
print(clean_df.head().to_string(index=False))

print("\nLast 5 rows:")
print(clean_df.tail().to_string(index=False))