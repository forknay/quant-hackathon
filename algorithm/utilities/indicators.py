import numpy as np
import pandas as pd
from arch.univariate import arch_model

def compute_indicators(df_sec: pd.DataFrame,
                                    ma_window: int,
                                    mom_lag: int,
                                    garch_params: dict,
                                    min_train: int = None) -> pd.DataFrame:
    """
    Compute technical indicators for a single security time series.
    
    Args:
        df_sec: rows for one (gvkey, iid), sorted by date
                columns: date, gvkey, iid, prc, stock_ret
        ma_window: Moving average window in days
        mom_lag: Momentum lookback period in days
        garch_params: GARCH model parameters dict
        min_train: Minimum training observations for GARCH
        
    Returns:
        DataFrame with: date, gvkey, iid, ma_<w>, ma_slope_<w>, mom_<L>, garch_vol
        
    Technical enhancements:
    - Respects MAX_TRAIN_WINDOW to prevent memory issues
    - Adds MA slope for trend detection
    - Robust GARCH fitting with error handling
    """
    df = df_sec.copy()
    w = int(ma_window)
    L = int(mom_lag)
    
    # Extract parameters with backward compatibility
    min_train = min_train or garch_params.get("min_train", 500)
    max_train_window = garch_params.get("max_train_window", 750)
    
    # --- Moving Average on price ---
    df[f"ma_{w}"] = df["prc"].rolling(w, min_periods=max(5, w // 4)).mean()
    
    # --- MA Slope (5-day change for trend detection) ---
    df[f"ma_slope_{w}"] = df[f"ma_{w}"].pct_change(5, fill_method=None)

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
        
        # Calculate training window with max cap per technical recommendations
        start_loc = max(0, end_loc + 1 - max_train_window)  # Respect MAX_TRAIN_WINDOW
        ret_series = df["stock_ret"].iloc[start_loc: end_loc + 1].dropna()

        if len(ret_series) < min_train:
            continue  # don't fit until we have enough history

        try:
            # Scale to percent to help optimizer stability
            am = arch_model(ret_series.values * 100.0, vol="Garch", p=p, q=q, dist=dist)
            res = am.fit(disp="off", show_warning=False)

            # 1-step-ahead variance forecast (for the NEXT day)
            f = res.forecast(horizon=1, reindex=False)
            sigma2_next = f.variance.values[-1, 0]
            sigma_next = np.sqrt(sigma2_next) / 100.0  # back to daily return units

            if end_loc + 1 < len(df):
                df.iat[end_loc + 1, df.columns.get_loc("garch_vol")] = sigma_next
        except Exception:
            # Robust error handling - continue processing
            pass

    # Forward-fill forecasts between refits
    df["garch_vol"] = df["garch_vol"].ffill()

    # Return only computed indicators (no sector-specific information)
    return df[["date", "gvkey", "iid", f"ma_{w}", f"ma_slope_{w}", f"mom_{L}", "garch_vol"]]
