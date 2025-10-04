import pandas as pd
from pathlib import Path

outdir = Path('text_features_out')
filing = outdir / 'filing_features.parquet'
monthly = outdir / 'text_features_monthly.parquet'

print('Checking Phase 1 outputs in', outdir.resolve())

if not filing.exists() or not monthly.exists():
    print('Missing outputs:')
    print(' - filing:', filing.exists())
    print(' - monthly:', monthly.exists())
    raise SystemExit(1)

f = pd.read_parquet(filing)
m = pd.read_parquet(monthly)

print('filing:', f.shape)
print('monthly:', m.shape)

has_pca = any(c.startswith('emb_pca_') for c in f.columns)
print('PCA columns present:', has_pca)

cols_show = [c for c in ['gvkey','date','length_words','sent_pos_mean','sent_pos_std','pos_ratio','emb_pca_1'] if c in f.columns]
print('filing head:')
print(f[cols_show].head())

cols_show_m = [c for c in ['gvkey','month_end','has_recent_text','days_since_last_filing','emb_pca_1'] if c in m.columns]
print('monthly head:')
print(m[cols_show_m].head())
