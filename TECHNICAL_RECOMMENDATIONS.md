# Technical Recommendations & Architecture Improvements

## Immediate Critical Issues to Address

### 1. Layer 1 Architecture Flaws

**Problem**: Current utilities implementation has several issues:
- MIN_TRAIN=250 is too small for GARCH (should be 500+ as per implementation.md)
- No MAX_TRAIN_WINDOW cap (can cause memory issues with long series)
- Missing candidate selection logic (just computes indicators)
- No sector-aware extremes detection

**Solutions**:
```python
# Updated config.py for all sectors
GARCH_PARAMS = {
    "p": 1, "q": 1, "dist": "t", 
    "MIN_TRAIN": 500,
    "MAX_TRAIN_WINDOW": 750  # Add this cap
}

# Add candidate selection parameters
SELECTION_PARAMS = {
    "TOP_K_RATIO": 0.15,    # Top 15% uptrending
    "BOTTOM_K_RATIO": 0.15, # Bottom 15% downtrending
    "VOL_BAND_LOWER": 0.05, # 5th percentile vol
    "VOL_BAND_UPPER": 0.95, # 95th percentile vol
    "MIN_MARKET_CAP": 100,  # $100M minimum
    "MIN_LIQUIDITY": 0.001  # Min daily volume ratio
}
```

### 2. Data Leakage Prevention

**Problem**: Current ML template doesn't guarantee strict temporal alignment

**Solutions**:
- Implement data versioning with explicit time-t snapshots
- Add automatic leakage detection tests
- Strict separation of train/validation/test with temporal gaps

### 3. Memory & Performance Issues

**Problems**:
- Large dataset loading without chunking in ML pipeline
- Inefficient groupby operations in portfolio analysis
- No caching between layers

**Solutions**:
```python
# Efficient data loading pattern
def load_monthly_data(date_range, features, cache_dir="cache/"):
    cache_file = f"{cache_dir}/monthly_{date_range[0]}_{date_range[1]}.parquet"
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    # Load and cache
    data = process_monthly_range(date_range, features)
    data.to_parquet(cache_file)
    return data
```

## Enhanced Architecture Proposal

### 1. Unified Configuration System

**Current**: Each sector has separate config files
**Proposed**: Centralized configuration with inheritance

```python
# config/base_config.py
@dataclass
class BaseConfig:
    # Common parameters
    INPUT_PARQUET: str = "cleaned_all.parquet"
    RESULTS_BASE_DIR: str = "results"
    N_JOBS: int = -1
    BATCH_SIZE: int = 64

@dataclass  
class SectorConfig(BaseConfig):
    sector_name: str
    gics_prefix: str
    ma_window: int
    mom_lag: int
    garch_params: dict
    
    def __post_init__(self):
        self.results_dir = f"{self.RESULTS_BASE_DIR}/{self.sector_name}"

# config/sector_configs.py
SECTOR_CONFIGS = {
    "utilities": SectorConfig(
        sector_name="utilities", gics_prefix="55",
        ma_window=60, mom_lag=120, 
        garch_params={"p": 1, "q": 1, "dist": "t", "min_train": 500}
    ),
    "it": SectorConfig(
        sector_name="it", gics_prefix="45",
        ma_window=30, mom_lag=90,
        garch_params={"p": 1, "q": 1, "dist": "t", "min_train": 500}
    ),
    # ... other sectors
}
```

### 2. Layered Data Architecture

**Proposed Pipeline**:
```
Raw Data (cleaned_all.parquet) 
    ↓
Layer 1 Processing (sector-specific signals)
    ↓
Monthly Aggregation Cache (monthly_features_YYYY_MM.parquet)
    ↓
Layer 2 ML Pipeline (147 factors + Layer 1 signals)
    ↓
Layer 3 Text Features (optional augmentation)
    ↓
Layer 4 Meta-Learning & Portfolio Construction
```

### 3. Improved Candidate Selection Logic

```python
def select_candidates(df_month, sector_config, selection_config):
    """
    Enhanced candidate selection with multiple filters
    """
    # 1. Basic filters
    df_filtered = df_month[
        (df_month['market_cap'] >= selection_config.MIN_MARKET_CAP) &
        (df_month['liquidity_ratio'] >= selection_config.MIN_LIQUIDITY) &
        (df_month['garch_vol'].between(
            df_month['garch_vol'].quantile(0.05),
            df_month['garch_vol'].quantile(0.95)
        ))
    ]
    
    # 2. Sector-aware ranking
    df_filtered['ma_slope'] = df_filtered.groupby('gics_sector')[f'ma_{sector_config.ma_window}'].pct_change(5)
    df_filtered['mom_zscore'] = df_filtered.groupby('gics_sector')[f'mom_{sector_config.mom_lag}'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # 3. Composite score
    df_filtered['composite_score'] = (
        0.4 * df_filtered['ma_slope'] + 
        0.4 * df_filtered['mom_zscore'] + 
        0.2 * (1 / df_filtered['garch_vol'])  # Prefer lower volatility
    )
    
    # 4. Select top/bottom candidates
    n_total = len(df_filtered)
    n_top = int(n_total * selection_config.TOP_K_RATIO)
    n_bottom = int(n_total * selection_config.BOTTOM_K_RATIO)
    
    top_candidates = df_filtered.nlargest(n_top, 'composite_score')
    bottom_candidates = df_filtered.nsmallest(n_bottom, 'composite_score')
    
    # 5. Label for downstream use
    top_candidates['candidate_type'] = 'long'
    bottom_candidates['candidate_type'] = 'short'
    
    return pd.concat([top_candidates, bottom_candidates])
```

### 4. Enhanced ML Pipeline Integration

**Current**: Separate ML template
**Proposed**: Integrated pipeline with proper validation

```python
class IntegratedMLPipeline:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.feature_importance_history = []
        
    def prepare_features(self, month_date):
        # Load Layer 1 candidates
        layer1_data = self.load_layer1_candidates(month_date)
        
        # Load 147 characteristics  
        factor_data = self.load_factor_characteristics(month_date)
        
        # Merge with strict temporal alignment
        features = self.merge_with_alignment(layer1_data, factor_data, month_date)
        
        return features
        
    def expanding_window_train(self, current_date):
        # Implement proper expanding window with validation gap
        train_end = current_date - pd.DateOffset(months=2)  # 2-month gap
        val_start = train_end + pd.DateOffset(days=1)
        val_end = current_date - pd.DateOffset(months=1)
        
        train_data = self.prepare_features_range(self.config.TRAIN_START, train_end)
        val_data = self.prepare_features_range(val_start, val_end)
        
        # Train ensemble of models
        self.train_ensemble(train_data, val_data)
        
    def predict_oos(self, prediction_date):
        features = self.prepare_features(prediction_date)
        candidates_only = features[features['is_candidate']]
        
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(candidates_only[self.feature_columns])
            predictions[model_name] = pred
            
        # Ensemble prediction
        final_pred = np.mean(list(predictions.values()), axis=0)
        return candidates_only[['gvkey', 'iid']], final_pred
```

### 5. Advanced Risk Management

```python
class RiskManager:
    def __init__(self, config):
        self.max_sector_exposure = config.MAX_SECTOR_EXPOSURE
        self.max_position_size = config.MAX_POSITION_SIZE
        self.vol_target = config.VOLATILITY_TARGET
        
    def apply_risk_constraints(self, portfolio_weights, features):
        # Sector exposure limits
        sector_exposures = portfolio_weights.groupby(features['gics_sector']).sum()
        for sector, exposure in sector_exposures.items():
            if abs(exposure) > self.max_sector_exposure:
                # Scale down sector positions
                sector_mask = features['gics_sector'] == sector
                scale_factor = self.max_sector_exposure / abs(exposure)
                portfolio_weights[sector_mask] *= scale_factor
        
        # Position size limits
        portfolio_weights = np.clip(portfolio_weights, -self.max_position_size, self.max_position_size)
        
        # Volatility targeting
        portfolio_vol = self.estimate_portfolio_vol(portfolio_weights, features)
        vol_scale = self.vol_target / portfolio_vol
        portfolio_weights *= vol_scale
        
        return portfolio_weights
        
    def estimate_portfolio_vol(self, weights, features):
        # Use GARCH forecasts for position-level risk
        individual_vols = features['garch_vol']
        # Simplified: assume correlation = 0.3 within sectors, 0.1 across sectors
        return np.sqrt(np.sum((weights * individual_vols) ** 2) * 1.3)  # Rough correlation adjustment
```

### 6. Performance Monitoring & Attribution

```python
class PerformanceAnalyzer:
    def __init__(self):
        self.benchmark_returns = self.load_benchmark_data()
        
    def comprehensive_attribution(self, portfolio_returns, holdings_history):
        results = {
            # Return metrics
            'total_return': portfolio_returns.sum(),
            'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(12),
            'max_drawdown': self.calculate_max_drawdown(portfolio_returns),
            
            # Risk metrics
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            
            # Factor attribution
            'factor_exposures': self.calculate_factor_exposures(holdings_history),
            'alpha_vs_benchmark': self.calculate_alpha(portfolio_returns),
            
            # Transaction costs
            'turnover': self.calculate_turnover(holdings_history),
            'estimated_transaction_costs': self.estimate_transaction_costs(holdings_history)
        }
        return results
```

## Implementation Priority Queue

1. **Week 1-2**: Fix Layer 1 architecture and implement all sectors
2. **Week 3-4**: Build unified data pipeline and caching system  
3. **Week 5-6**: Integrate Layer 2 ML with proper validation
4. **Week 7-8**: Implement enhanced portfolio construction and risk management
5. **Week 9-10**: Add Layer 3 text processing (if time permits)
6. **Week 11-12**: Layer 4 meta-learning and final optimization

## Quality Assurance Checklist

- [ ] All date operations use explicit timezone handling
- [ ] No data leakage in train/validation/test splits
- [ ] Memory usage stays under 16GB for full backtest
- [ ] All random seeds are fixed for reproducibility
- [ ] Comprehensive logging for debugging
- [ ] Unit tests for all core functions
- [ ] Integration tests for end-to-end pipeline 