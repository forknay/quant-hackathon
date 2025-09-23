"""Second pass: streaming clean of ret_sample.csv into a single Parquet.
Operations per feature:
1. Winsorize using sampled quantiles (returns tighter)
2. Transform (log1p or signed_log by category)
3. Impute missing with global median + add *_was_missing flag
4. Per-chunk month grouping -> per-month robust scaling (median/MAD) (no cross-chunk carry)
5. Optional final absolute clip
"""
from __future__ import annotations
import json, numpy as np, pandas as pd, time, sys, pathlib, os
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pyarrow as pa, pyarrow.parquet as pq
from typing import Dict, Set, List
from collections import defaultdict
from cleaning.config import (
    RAW_FILE, PROFILE_JSON, OUTPUT_PARQUET, DATE_COL, TARGET_COL, ID_COLS,
    CLIP_RULES, RETURNS_TAGS, LOG1P_CATEGORIES, SIGNED_LOG_CATEGORIES,
    CHUNKSIZE, FINAL_ABS_CLIP, ACRONYM_FILE, infer_category,
    FLUSH_EACH_CHUNK, ADD_MISSING_FLAGS, CARRY_MAX_ROWS, CAST_FLOAT32
)

# Safety valve: if the first month never ends (data all one month) and grows too large,
# flush partial segments early to avoid huge single-table conversions / memory spikes.
DEFRAG_INTERVAL_ROWS = 300_000  # make a full copy occasionally to defragment

# ---------- Helpers ----------

def load_profile() -> Dict[str, Dict[str, float]]:
    with open(PROFILE_JSON) as f:
        return json.load(f)["profile"]


def load_acronym_map() -> Dict[str, str]:
    try:
        df = pd.read_csv(ACRONYM_FILE)
        # expecting 'Acronym' maybe other column names; unify to lowercase keys
        if "Acronym" in df.columns and "Feature" in df.columns:
            # no explicit theme column in provided sample; map to placeholder 'other'
            return {a.strip().lower(): 'other' for a in df['Acronym']}
        if 'feature' in df.columns and 'theme' in df.columns:
            return dict(zip(df['feature'].str.lower(), df['theme'].str.lower()))
    except Exception:
        pass
    return {}


def winsorize(series: pd.Series, cat: str, stats: Dict[str, float]) -> pd.Series:
    if not stats:
        return series
    if cat == 'returns':
        low, high = stats.get('q0005'), stats.get('q995')
    else:
        low, high = stats.get('q01'), stats.get('q99')
    if low is None or high is None:
        return series
    return series.clip(low, high)


def log1p_positive(s: pd.Series) -> pd.Series:
    out = s.copy()
    mask = out > 0
    if mask.any():
        out.loc[mask] = np.log1p(out.loc[mask])
    return out


def signed_log(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def transform(series: pd.Series, cat: str) -> pd.Series:
    if cat in LOG1P_CATEGORIES and (series > 0).any():
        return log1p_positive(series)
    if cat in SIGNED_LOG_CATEGORIES:
        return signed_log(series)
    return series


def robust_scale_month(df_month: pd.DataFrame, feature_cols) -> pd.DataFrame:
    if not feature_cols:
        return df_month
    med = df_month[feature_cols].median()
    mad = (df_month[feature_cols] - med).abs().median().replace(0, 1e-9)
    df_month.loc[:, feature_cols] = (df_month[feature_cols] - med) / mad
    if FINAL_ABS_CLIP:
        df_month.loc[:, feature_cols] = df_month[feature_cols].clip(-FINAL_ABS_CLIP, FINAL_ABS_CLIP)
    return df_month


def write_status(status: Dict):
    try:
        with open('cleaning/status.json','w') as f:
            json.dump(status, f)
    except Exception:
        pass

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with duplicate column names removed (keeping first)."""
    if df.columns.has_duplicates:
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df

# ---------- Main Cleaning ----------

def main():
    t0 = time.time()
    profile = load_profile()
    acronym_map = load_acronym_map()
    writer = None

    # QA accumulators
    feature_set: Set[str] = set()
    clipped_counts = defaultdict(int)
    missing_counts = defaultdict(int)
    non_null_counts = defaultdict(int)
    months_processed: List[str] = []
    rows = 0
    # No cross-chunk month carry; each chunk's months scaled independently
    detected_date_col = DATE_COL

    def process_month_group(df_group: pd.DataFrame, numeric_cols: List[str]):
        nonlocal writer
        if df_group.empty:
            return
        feat_cols = [c for c in df_group.columns if c in numeric_cols]
        df_group = robust_scale_month(df_group, feat_cols)
        table = pa.Table.from_pandas(df_group, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema)
        writer.write_table(table)

    # Limit math library threads to reduce system freeze perception
    os_environ_set = False
    for var in ["OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        if var not in os.environ:
            os.environ[var] = "2"
            os_environ_set = True
    if os_environ_set:
        write_status({'note': 'thread_limits_set=2'})

    max_chunks = int(os.environ.get('MAX_CHUNKS', '0')) or None
    chunk_counter = 0
    for chunk in pd.read_csv(RAW_FILE, chunksize=CHUNKSIZE, low_memory=False):
        chunk_counter += 1
        if max_chunks and chunk_counter > max_chunks:
            break
        if detected_date_col not in chunk.columns:
            if 'date' in chunk.columns:
                detected_date_col = 'date'
            else:
                raise KeyError(f"Date column '{DATE_COL}' not found. Columns: {list(chunk.columns)[:20]}")
        chunk[detected_date_col] = pd.to_datetime(
            chunk[detected_date_col].astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce"
        )
        # Ensure ID columns remain string (prevents pyarrow bytes/float conversion error)
        for idc in ID_COLS:
            if idc in chunk.columns:
                chunk[idc] = chunk[idc].astype(str)
        chunk = chunk.sort_values(detected_date_col)

        # Ensure no duplicate columns in fresh chunk
        chunk = drop_duplicate_columns(chunk)

    # No carry merge (simplified): treat each chunk independently

        # --- Minimal object->numeric coercion (helps mixed-type columns) ---
        # Attempt to coerce object columns that appear numeric (>=90% convertible)
        obj_candidates = [c for c in chunk.select_dtypes(include=['object']).columns
                          if c not in ID_COLS and c not in (TARGET_COL, detected_date_col) and not c.endswith('_was_missing')]
        for c in obj_candidates:
            coerced = pd.to_numeric(chunk[c], errors='coerce')
            if coerced.notna().mean() >= 0.9:  # treat as numeric
                chunk[c] = coerced

        # Opportunistic defragment copy every few hundred K rows processed
        if rows and rows % DEFRAG_INTERVAL_ROWS == 0:
            chunk = chunk.copy()

        numeric = [c for c in chunk.select_dtypes(include=['number']).columns
                   if c not in ID_COLS and c != TARGET_COL and not c.endswith('_was_missing')]
        feature_set.update(numeric)
        cat_map = {col: infer_category(col, acronym_map) for col in numeric}

        # Optional float32 cast to cut memory
        if CAST_FLOAT32:
            for c in chunk.select_dtypes(include=['float64','int64']).columns:
                if c not in ID_COLS:
                    chunk[c] = pd.to_numeric(chunk[c], errors='coerce').astype('float32')

        # Vectorized per-column transforms
        missing_flags = {}
        if months_processed:
            # Normal processing once at least one month written
            for col in numeric:
                stats = profile.get(col, {})
                cat = cat_map[col]
                before = chunk[col]
                before_notna = before.notna()

                # Winsorize and count only real clips (exclude NaNs)
                after = winsorize(before, cat, stats)
                clipped_counts[col] += (before_notna & before.ne(after)).sum()

                # Transform
                after = transform(after, cat)

                # Missing counting based on pre-impute state
                missing_counts[col] += (~before_notna).sum()
                non_null_counts[col] += before_notna.sum()

                # Optional flags (post-transform NaNs are fine)
                if ADD_MISSING_FLAGS:
                    miss_mask = after.isna()
                    missing_flags[f"{col}_was_missing"] = miss_mask.astype(int)

                fill_val = stats.get('median', 0.0)
                chunk[col] = after.fillna(fill_val)

        else:
            # Fast path for first months to reduce overhead
            if numeric:
                sub = chunk[numeric]
                med_map = {col: profile.get(col, {}).get('median', 0.0) for col in numeric}
                miss_mask_df = sub.isna()
                for col in numeric:
                    missing_counts[col] += miss_mask_df[col].sum()
                    non_null_counts[col] += (~miss_mask_df[col]).sum()
                chunk[numeric] = sub.fillna(value=med_map)
                if ADD_MISSING_FLAGS:
                    for col in numeric:
                        missing_flags[f"{col}_was_missing"] = miss_mask_df[col].astype(int)
        if ADD_MISSING_FLAGS and missing_flags:
            flag_df = pd.DataFrame(missing_flags, index=chunk.index)
            chunk = pd.concat([chunk, flag_df], axis=1)
        # Drop any accidental duplicate columns created during flag concat
        chunk = drop_duplicate_columns(chunk)

        # Add year-month key and immediately process all months within this chunk
        ym = chunk[detected_date_col].dt.to_period('M')
        chunk['_ym_'] = ym
        for month_val, df_month in chunk.groupby('_ym_'):
            df_month = df_month.drop(columns=['_ym_'])
            process_month_group(df_month, numeric)
            months_processed.append(str(month_val))

        # Row & status accounting (inside loop)
        rows += len(chunk)
        elapsed = time.time() - t0
        if rows % (CHUNKSIZE) == 0:
            write_status({'rows': rows, 'months': len(months_processed), 'elapsed_sec': elapsed})
            print(f"[clean] rows={rows:,} months={len(months_processed)} elapsed={elapsed/60:.1f}m")

    # No carry finalization needed

    if writer:
        writer.close()

    qa = {}
    for col in sorted(feature_set):
        nn = int(non_null_counts.get(col, 0))
        mm = int(missing_counts.get(col, 0))
        total = nn + mm
        qa[col] = {
            'clipped': int(clipped_counts.get(col, 0)),
            'clipped_pct_non_null': float(clipped_counts.get(col, 0) / nn) if nn else 0.0,
            'missing': mm,
            'missing_rate': float(mm / total) if total else 0.0
        }
    summary = {
        'months_processed': months_processed,
        'n_features': len(feature_set),
        'qa': qa,
        'total_rows': rows,
        'elapsed_sec': time.time() - t0
    }
    with open('cleaning/qa_summary.json','w') as f:
        json.dump(summary, f)
    write_status({'done': True, 'rows': rows, 'months': len(months_processed)})
    print("[clean] done ->", OUTPUT_PARQUET, '| QA: cleaning/qa_summary.json')


if __name__ == "__main__":
    main()
