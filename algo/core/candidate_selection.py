"""
Enhanced candidate selection logic per technical recommendations.

Implements sector-aware ranking and composite scoring to select top/bottom candidates
for portfolio construction, with risk filtering and turnover considerations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def select_candidates(df_month: pd.DataFrame, 
                     sector_config: Dict[str, Any],
                     selection_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Simplified candidate selection using only the core technical indicators.
    
    Args:
        df_month: Monthly indicators data for all stocks in the sector
        sector_config: Sector-specific configuration (MA_WINDOW, MOM_LAG, etc.)
        selection_config: Selection parameters (TOP_K_RATIO, BOTTOM_K_RATIO, etc.)
    
    Returns:
        DataFrame with selected candidates and their composite scores
    """
    if df_month.empty:
        return pd.DataFrame()
    
    df_filtered = df_month.copy()
    
    # 1. Check for required technical indicator columns
    ma_col = f"ma_{sector_config['ma_window']}"
    ma_slope_col = f"ma_slope_{sector_config['ma_window']}"
    mom_col = f"mom_{sector_config['mom_lag']}"
    
    required_cols = [ma_col, ma_slope_col, mom_col, 'garch_vol']
    missing_cols = [col for col in required_cols if col not in df_filtered.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols} in candidate selection")
        return pd.DataFrame()
    
    # Remove rows with all NaN indicators (but keep partial data)
    df_filtered = df_filtered.dropna(subset=required_cols, how='all')
    
    if len(df_filtered) == 0:
        print("Warning: No valid indicator data found")
        return pd.DataFrame()
    
    initial_count = len(df_filtered)
    print(f"Processing {initial_count} stocks with valid indicators")
    
    # 2. Create technical signals for ranking (no filtering, just scoring)
    df_filtered = df_filtered.copy()
    
    # MA slope (trend signal) - raw values
    df_filtered['ma_slope_signal'] = df_filtered[ma_slope_col].fillna(0)
    
    # Momentum signal - cross-sectional standardization
    mom_values = df_filtered[mom_col].dropna()
    if len(mom_values) > 1:
        df_filtered['mom_zscore'] = (
            (df_filtered[mom_col] - mom_values.mean()) / 
            (mom_values.std() + 1e-8)
        ).fillna(0)
    else:
        df_filtered['mom_zscore'] = 0
    
    # Volatility score (prefer moderate volatility) - inverted rank
    garch_values = df_filtered['garch_vol'].dropna()
    if len(garch_values) > 1:
        df_filtered['vol_rank'] = df_filtered['garch_vol'].rank(method='average', na_option='keep')
        df_filtered['vol_score'] = (
            1.0 - (df_filtered['vol_rank'] - 1) / (len(garch_values) - 1)
        ).fillna(0.5)  # Neutral score for missing volatility
    else:
        df_filtered['vol_score'] = 0.5  # Neutral score
    
    # 3. Composite score using core technical indicators
    # Weights: 50% momentum + 30% MA slope + 20% volatility preference
    df_filtered['composite_score'] = (
        0.5 * df_filtered['mom_zscore'] + 
        0.3 * df_filtered['ma_slope_signal'] + 
        0.2 * df_filtered['vol_score']
    )
    
    # Handle any remaining NaN values in composite score
    df_filtered['composite_score'] = df_filtered['composite_score'].fillna(0)
    
    # 4. Select top and bottom candidates based on composite score
    n_total = len(df_filtered)
    n_top = max(1, int(n_total * selection_config['top_k_ratio']))
    n_bottom = max(1, int(n_total * selection_config['bottom_k_ratio']))
    
    # Sort by composite score and select candidates
    df_sorted = df_filtered.sort_values('composite_score', ascending=False)
    
    top_candidates = df_sorted.head(n_top).copy()
    bottom_candidates = df_sorted.tail(n_bottom).copy()
    
    # 5. Label candidates for downstream use
    top_candidates['candidate_type'] = 'long'
    bottom_candidates['candidate_type'] = 'short'
    
    # Add selection metadata
    top_candidates['selection_rank'] = range(1, len(top_candidates) + 1)
    bottom_candidates['selection_rank'] = range(1, len(bottom_candidates) + 1)
    
    # Combine and add additional metadata
    selected_candidates = pd.concat([top_candidates, bottom_candidates], ignore_index=True)
    selected_candidates['selection_date'] = df_month['date'].iloc[0] if len(df_month) > 0 else pd.NaT
    selected_candidates['total_universe'] = initial_count
    selected_candidates['filtered_universe'] = n_total
    
    # Add sector information for multi-sector compatibility
    if 'gics' in df_month.columns:
        selected_candidates['sector'] = df_month['gics'].iloc[0] if len(df_month) > 0 else None
    
    print(f"Selected {len(top_candidates)} long + {len(bottom_candidates)} short candidates "
          f"from {n_total} valid stocks")
    
    return selected_candidates


def aggregate_monthly_candidates(df_indicators: pd.DataFrame,
                               sector_config: Dict[str, Any], 
                               selection_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply candidate selection month by month to the indicators data.
    
    Args:
        df_indicators: Multi-month indicators data
        sector_config: Sector configuration
        selection_config: Selection parameters
        
    Returns:
        DataFrame with all monthly candidate selections
    """
    if df_indicators.empty:
        return pd.DataFrame()
    
    # Group by year-month and apply selection
    df_indicators['year_month'] = df_indicators['date'].dt.to_period('M')
    
    monthly_candidates = []
    
    for period, month_data in df_indicators.groupby('year_month'):
        candidates = select_candidates(month_data, sector_config, selection_config)
        if not candidates.empty:
            candidates['year_month'] = period
            monthly_candidates.append(candidates)
    
    if monthly_candidates:
        result = pd.concat(monthly_candidates, ignore_index=True)
        # Clean up temporary column
        result = result.drop('year_month', axis=1, errors='ignore')
        return result
    else:
        return pd.DataFrame()


def validate_candidate_selection(candidates_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the candidate selection results and provide summary statistics.
    
    Args:
        candidates_df: Selected candidates dataframe
        
    Returns:
        Dictionary with validation metrics and summary statistics
    """
    if candidates_df.empty:
        return {"status": "error", "message": "No candidates selected"}
    
    long_candidates = candidates_df[candidates_df['candidate_type'] == 'long']
    short_candidates = candidates_df[candidates_df['candidate_type'] == 'short']
    
    validation_results = {
        "status": "success",
        "total_candidates": len(candidates_df),
        "long_candidates": len(long_candidates),
        "short_candidates": len(short_candidates),
        "unique_dates": candidates_df['selection_date'].nunique(),
        "date_range": {
            "start": candidates_df['selection_date'].min(),
            "end": candidates_df['selection_date'].max()
        },
        "composite_score_stats": {
            "mean": candidates_df['composite_score'].mean(),
            "std": candidates_df['composite_score'].std(),
            "min": candidates_df['composite_score'].min(),
            "max": candidates_df['composite_score'].max()
        }
    }
    
    # Check for potential issues
    warnings = []
    if len(long_candidates) == 0 or len(short_candidates) == 0:
        warnings.append("Imbalanced long/short selection")
    
    if candidates_df['composite_score'].std() < 0.01:
        warnings.append("Very low score dispersion - may indicate poor signal")
        
    validation_results["warnings"] = warnings
    
    return validation_results 