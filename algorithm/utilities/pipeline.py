import os, glob
import pandas as pd
from joblib import Parallel, delayed

from config import (
    INPUT_PARQUET, INPUT_DAYS_DIR, RESULTS_DIR,
    UTILITIES_GICS_PREFIX,
    MA_WINDOW, MOM_LAG, GARCH_PARAMS, MIN_TRAIN,
    COL_DATE, COL_GVKEY, COL_IID, COL_GICS, COL_PRICE, COL_RET_RAW,
    N_JOBS, BATCH_SIZE
)
from indicators import compute_indicators

USE_COLS = [COL_DATE, COL_GVKEY, COL_IID, COL_GICS, COL_PRICE, COL_RET_RAW]

def _is_utilities(g):
    # robust even if g is numeric: string-ify then prefix match "55"
    return str(g).startswith(UTILITIES_GICS_PREFIX)

def load_utilities_dataframe() -> pd.DataFrame:
    if INPUT_PARQUET and os.path.exists(INPUT_PARQUET):
        df = pd.read_parquet(INPUT_PARQUET, columns=USE_COLS)
    else:
        # Fallback: daily CSVs (only if you decide to bypass the cleaner)
        files = sorted(glob.glob(os.path.join(INPUT_DAYS_DIR, "*.csv")))
        parts = []
        for fp in files:
            parts.append(pd.read_csv(fp, usecols=lambda c: c in USE_COLS, parse_dates=[COL_DATE]))
        if not parts:
            raise RuntimeError("No input found. Set INPUT_PARQUET or INPUT_DAYS_DIR correctly.")
        df = pd.concat(parts, ignore_index=True)

    # Utilities-only
    df = df[df[COL_GICS].apply(_is_utilities)]

    # Basic type hygiene (no extra cleaning)
    df[COL_GVKEY] = df[COL_GVKEY].astype("category")
    df[COL_IID]   = df[COL_IID].astype("category")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[COL_DATE]):
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

    # ---- RAW price & return guarantees ----
    # 1) Price must be raw level (not scaled/transformed). If your cleaned file stored negative CRSP prices, keep abs().
    df[COL_PRICE] = pd.to_numeric(df[COL_PRICE], errors="coerce").abs()

    # 2) Return must be raw (decimal). If the cleaner replaced it with a scaled value,
    #    rebuild a raw daily return from price within each (gvkey,iid) as a fallback.
    df[COL_RET_RAW] = pd.to_numeric(df[COL_RET_RAW], errors="coerce")

    # Heuristic: if returns look "too big" or too centered (scaled), rebuild from price
    # (Daily raw returns rarely have std >> 0.2 across a long panel; robust scale typically yields ~unit scale.)
    if df[COL_RET_RAW].abs().median() > 0.5 or df[COL_RET_RAW].abs().quantile(0.99) > 5:
        df[COL_RET_RAW] = df.sort_values([COL_GVKEY, COL_IID, COL_DATE]) \
                            .groupby([COL_GVKEY, COL_IID])[COL_PRICE] \
                            .pct_change()

    # Minimal NA handling: keep rows where we can compute signals
    df = df.dropna(subset=[COL_DATE, COL_GVKEY, COL_IID, COL_PRICE])  # keep return NaNs; GARCH will drop internally

    df = df.drop_duplicates(subset=[COL_DATE, COL_GVKEY, COL_IID]) \
           .sort_values([COL_GVKEY, COL_IID, COL_DATE])

    # Convenience partitions
    df["year"]  = df[COL_DATE].dt.year.astype("int16")
    df["month"] = df[COL_DATE].dt.month.astype("int8")
    return df

def _process_one(group_key_df):
    (_, _), df_sec = group_key_df
    # compute indicators on the minimal columns
    cols = [COL_DATE, COL_GVKEY, COL_IID, COL_GICS, COL_PRICE, COL_RET_RAW]
    out = compute_indicators(
        df_sec[cols].sort_values(COL_DATE) \
                    .rename(columns={COL_DATE:"date", COL_GVKEY:"gvkey", COL_IID:"iid",
                                     COL_GICS:"gics", COL_PRICE:"prc", COL_RET_RAW:"stock_ret"}),
        ma_window=MA_WINDOW,
        mom_lag=MOM_LAG,
        garch_params=GARCH_PARAMS,
        min_train=MIN_TRAIN,
    )
    out["year"]  = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    out.to_parquet(RESULTS_DIR, partition_cols=["year","month"], engine="pyarrow", compression="zstd")
    return 1

def run():
    df = load_utilities_dataframe()
    groups = list(df.groupby([COL_GVKEY, COL_IID], sort=False))
    Parallel(n_jobs=N_JOBS, prefer="processes", batch_size=BATCH_SIZE)(
        delayed(_process_one)(g) for g in groups
    )
    print(f"Done. Wrote Utilities signals to: {RESULTS_DIR}")

if __name__ == "__main__":
    run()
