import numpy as np
import pandas as pd
from arch.univariate import arch_model

def compute_indicators(df_sec: pd.DataFrame,
                                    ma_window: int,
                                    mom_lag: int,
                                    garch_params: dict,
                                    min_train: int) -> pd.DataFrame:
    """
    df_sec: rows for one (gvkey, iid), sorted by date; columns: date, gvkey, iid, gics, prc, stock_ret
    Returns: date, gvkey, iid, gics, ma_<w>, mom_<L>, garch_vol
    """
    df = df_sec.copy()
    w = int(ma_window)
    L = int(mom_lag)

    # --- Moving Average on price ---
    df[f"ma_{w}"] = df["prc"].rolling(w, min_periods=max(5, w // 4)).mean()

    # --- Momentum over L (percent change ratio) ---
    df[f"mom_{L}"] = (df["prc"] / df["prc"].shift(L)) - 1.0

    # --- GARCH(1,1) 1-step-ahead daily volatility forecasts ---
    # We refit on month-ends; assign forecast to the *next* trading day; forward-fill in between.
    df["garch_vol"] = np.nan

    # Month-end rows: last date of each month
    month_end_mask = (df["date"].dt.to_period("M") != df["date"].shift(-1).dt.to_period("M")).fillna(True)
    month_end_idx = df.index[month_end_mask].tolist()

    p = garch_params.get("p", 1)
    q = garch_params.get("q", 1)
    dist = garch_params.get("dist", "normal")

    for idx in month_end_idx:
        end_loc = df.index.get_loc(idx)
        ret_series = df["stock_ret"].iloc[: end_loc + 1].dropna()

        if len(ret_series) < min_train:
            continue  # don’t fit until we have enough history

        try:
            # Scale to percent to help optimizer stability
            am = arch_model(ret_series.values * 100.0, vol="Garch", p=p, q=q, dist=dist)
            res = am.fit(disp="off")

            # 1-step-ahead variance forecast (for the NEXT day)
            f = res.forecast(horizon=1, reindex=False)
            sigma2_next = f.variance.values[-1, 0]
            sigma_next = np.sqrt(sigma2_next) / 100.0  # back to daily return units

            if end_loc + 1 < len(df):
                df.iat[end_loc + 1, df.columns.get_loc("garch_vol")] = sigma_next
        except Exception:
            # leave NaN; next refit will try again
            pass

    # Forward-fill forecasts between refits
    df["garch_vol"] = df["garch_vol"].ffill()

    return df[["date", "gvkey", "iid", "gics", f"ma_{w}", f"mom_{L}", "garch_vol"]]
