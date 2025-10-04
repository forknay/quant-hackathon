from __future__ import annotations

import pandas as pd
from . import config as C


def align_to_month(filing_df: pd.DataFrame) -> pd.DataFrame:
    df = filing_df.copy()
    df[C.COL_GVKEY] = df[C.COL_GVKEY].astype(str)
    df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE])

    # Build month_end calendar per gvkey
    months = (
        df[[C.COL_GVKEY, C.COL_DATE]].assign(month_end=lambda x: x[C.COL_DATE].dt.to_period("M").dt.to_timestamp("M"))
        [[C.COL_GVKEY, "month_end"]]
        .drop_duplicates()
    )

    calendars = []
    for gvkey, grp in months.groupby(C.COL_GVKEY):
        mn, mx = grp["month_end"].min(), grp["month_end"].max()
        rng = pd.period_range(mn, mx, freq="M").to_timestamp("M")
        calendars.append(pd.DataFrame({C.COL_GVKEY: str(gvkey), "month_end": rng}))
    calendar = pd.concat(calendars, ignore_index=True)

    filings_sorted = df.sort_values([C.COL_GVKEY, C.COL_DATE])[[C.COL_GVKEY, C.COL_DATE]].drop_duplicates()
    aligned = pd.merge_asof(
        calendar.sort_values([C.COL_GVKEY, "month_end"]),
        filings_sorted.rename(columns={C.COL_DATE: "filing_date"}).sort_values([C.COL_GVKEY, "filing_date"]),
        by=C.COL_GVKEY,
        left_on="month_end",
        right_on="filing_date",
        direction="backward",
    )

    keyed = df.drop_duplicates(subset=[C.COL_GVKEY, C.COL_DATE])
    out = aligned.merge(keyed, left_on=[C.COL_GVKEY, "filing_date"], right_on=[C.COL_GVKEY, C.COL_DATE], how="left")

    out["days_since_last_filing"] = (out["month_end"] - out["filing_date"]).dt.days
    out["has_recent_text"] = out["filing_date"].notna().astype("int8")
    out = out.drop(columns=[C.COL_DATE], errors="ignore")
    return out
