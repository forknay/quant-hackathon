# McGill–FIAM Asset Management Hackathon: 4‑Layer Strategy & Deliverables

> **Objective:** Predict next‑month stock returns \(t \to t+1\) using only information known at time \(t\). All predictors must be time‑aligned and leakage‑free.

---

## 0) Data Governance & Leakage Guard (applies to all layers)

- **Target:** Next‑month return `stock_ret_{t+1}` predicted with information available at month‑end \(t\). (Toolkit and factor list are provided; base predictors are already lagged one month.)
- **Temporal split & retraining:** Use an **expanding window** with **rolling validation** and **monthly OOS predictions**.
  - Example split: Train 2005–2012, Validate 2013–2014, OOS Jan‑2015 → May‑2025. (If you adjust start years, update all references consistently.)
- **Portfolio rules:** Build a **global long–short** portfolio with **100–250 names**, rebalanced **monthly** (semi‑annual acceptable). Report **alpha vs S&P 500**, **Sharpe**, **drawdowns**, **turnover**.
- **Text linkage:** Map **10‑K/10‑Q** filings from **CIK ↔ gvkey** using the provided link table; only use filings **publicly available by time \(t\)**.
- **Deliverables (top‑level):** OOS \(R^2\) (competition definition), portfolio statistics, a **5‑page deck** (with content guidelines), and a **zipped code** folder.

---

## 1) Layer‑1 — Sector‑Aware MA/MOM/GARCH (Candidate Selection)

### Goal
For each month \(t\), per stock, compute:
- **MA trend** over a sector‑tuned window \(w\) (e.g., Utilities: 60 trading days).
- **Momentum (MOM)** over a lag \(L\) (e.g., Utilities: 120 trading days).
- **GARCH vol forecast** \(\hat{\sigma}_{t+1\mid t}\) from daily returns, **monthly re‑fit**, with `MIN_TRAIN=500`, `MAX_TRAIN_WINDOW=750`, **Student‑t** errors.

### Monthly workflow
- **Daily → monthly cut:** Compute indicators from daily data up to month‑end \(t\); **snapshot** values for month \(t\).
- **Sector knobs (by GICS sector):** Configure \(w\), \(L\), and distribution; defensives (UTIL, STAP) get **larger** windows; cyclicals (IT, CD) **smaller** windows.
- **Extremes selection (no look‑ahead):** Rank within each sector (or globally with sector caps). Pass only:
  - **Top‑K “uptrending”** (e.g., MA slope \(>0\), MOM above threshold, vol within band),
  - **Bottom‑K “downtrending”** (reverse signs).
- **Risk constraints:** Exclude illiquid names (zero‑trade flags), micro‑caps, or extreme/unstable vol tails. Keep **turnover** low (reported metric).

#### Example sector settings (illustrative)
| Sector        | GICS Prefix | MA Window \(w\) | MOM Lag \(L\) | GARCH Dist. | Notes                  |
|---|---:|---:|---:|---|---|
| Utilities     | `55`        | 60d              | 120d            | Student‑t    | More defensive         |
| Staples (TBD) | `30`        | 60–90d           | 120–180d        | Student‑t    | Defensive              |
| IT (TBD)      | `45`        | 20–40d           | 60–90d          | Student‑t    | Cyclical               |
| Discretionary (TBD) | `25`  | 20–40d           | 60–90d          | Student‑t    | Cyclical               |

**Output (per month \(t\))**: a **candidate set** for Layer‑2 containing ~**2–3×** the desired final book size, balanced long/short.

---

## 2) Layer‑2 — Signal‑Driven ML (Train on All Stocks; Score Only Candidates)

- **Expectation:** Predict next‑month returns using the **147 characteristics** with strict OOS protocol and OOS \(R^2\) reporting (baseline starter code includes LASSO/EN/Ridge and splits).
- **Modeling universe:** Train on the **full global panel** each month; **apply scores only to Layer‑1 candidates** for portfolio construction.
- **Features:** `factor_char_list.csv` (147 predictors), optionally **augment** with Layer‑1 signals (e.g., MOM, MA slope, recent realized vol) computed with **time‑t** data.
- **Preprocessing (cross‑sectional, monthly):** winsorize/robust‑scale **within month** (cleaner already available). **No future info** may leak.
- **Algorithms:** Start with **LASSO / Elastic Net / Ridge** (baseline). Consider **LightGBM/XGBoost** for nonlinearity while preserving the **expanding/OOS** procedure.
- **Tuning:** Validate on the **rolling validation** slice; **lock hyper‑parameters** for the next OOS year.
- **Outputs:** For month \(t\), predicted return \(\hat{r}_{i,t+1}\) for all stocks → **filter to candidates** → **rank** long/short baskets.
- **Report:** OOS \(R^2\) using the competition’s zero‑benchmark definition, then portfolio statistics.

---

## 3) Layer‑3 — LLMs on 10‑K/10‑Q Text (Incremental Edge)

- **Scope:** ~30GB of filings; focus on **MD&A** and **Risk Factors**, mapped **CIK ↔ gvkey**. Keep the scope manageable.
- **MVP feature pipeline:**
  1. **Featurize** each filing at time \(t\) via sentence embeddings (e.g., FinBERT/MPNet) + simple **sentiment/tone** scores (MD&A, Risk Factors).
  2. **Time‑t availability:** Use the filing’s **accepted date** to assign to month \(t\); never use \(t+1\) filings.
  3. **Fuse:** Aggregate to **firm‑month** features (mean embedding, tone, uncertainty, litigiousness keywords), then **join** to the panel on `(gvkey, year‑month)` using the link table.
- **Usage in models:** 
  - (a) **Augment Layer‑2** features; or 
  - (b) Build a small **secondary model** and **blend** predictions (stacking).
- **Stretch goal:** Predict fundamentals (e.g., **EBIT/Sales**, **ROA**) for \(t+1\) from **text + quant**, then use those predictions as **features** for return prediction (explicitly permitted as an “alternative target” workflow).

---

## 4) Layer‑4 — Neural Consolidation & Portfolio Construction

### Stacking / Meta‑Learner
- **Inputs:** Layer‑2 ML prediction, Layer‑3 text prediction, optionally Layer‑1 signals.
- **Model:** **Small MLP** or **ridge** meta‑learner trained **only** on train+validation; **freeze** and score OOS monthly.
- **Calibration:** Apply monotonic transform or **Platt/Isotonic** calibration on validation to keep **scores comparable through time**.

### Portfolio Build (meets rules)
- Each month, **rank OOS predictions** and form a **long–short** book with **100–250 names** (equal‑weight acceptable per toolkit). **Rebalance monthly.**

### Tracking & Reporting
- **Performance:** annualized return & stdev, **alpha vs S&P 500**, **Sharpe**, **Information Ratio**, **max drawdown**, **max 1‑month loss**, **turnover**.
- Use `mkt_ind.csv` for alpha/beta estimation.

### Risk Knobs (practical)
- **Sector/country neutrality caps**, **position caps** (e.g., 1% weight), **volatility targeting** (scale exposure to hit a target monthly vol), and **turnover penalties** to limit trading‑cost drag (explicitly called out).

---

## C) Final Package (per rules)

- **OOS \(R^2\):** competition definition, over **2015–2025** OOS window.
- **5‑page deck:**
  1. Executive summary
  2. Strategy & top holdings
  3. Data/methods & OOS \(R^2\)
  4. Portfolio statistics
  5. Discussion (+ short appendix if needed)
- **Zipped code folder:** main run script, clear comments, style guide. **No forward‑looking usage**; they will check.

---

## D) Practical Tips (speed & cleanliness)

- **Cache** daily→monthly Layer‑1 outputs so Layers 2–4 operate on small monthly tables.
- **Parallelize** by `(gvkey, iid)`; store as **Parquet**, partitioned by **year/month**.
- **Cap GARCH window** (e.g., 750) + **monthly re‑fit** to keep compute linear and avoid optimizer grind.
- Maintain **one config per sector** (Utilities `55` started); clone/adjust configs as you scale to other sectors.

---

### Notation
- \(\hat{\sigma}_{t+1\mid t}\): GARCH one‑step‑ahead volatility forecast using information up to \(t\).
- \(\hat{r}_{i,t+1}\): Predicted return for stock \(i\) for month \(t+1\), scored at \(t\).
- **OOS**: Out‑of‑sample.
