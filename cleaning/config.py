"""Configuration and categorization utilities for statistical cleaning of ret_sample.csv.

This file centralizes:
- File paths
- Column identifiers
- Category inference heuristics using provided Stock_features__Feature__Acronym_.csv mapping
- Clipping & transform rules
- Random seeds and chunk sizes
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

# ---- File & IO Settings ----
RAW_FILE = "Data/ret_sample.csv"  # large input
ACRONYM_FILE = "Data/Stock_features__Feature__Acronym_.csv"
PROFILE_JSON = "cleaning/profile_stats.json"
OUTPUT_PARQUET = "cleaned_all.parquet"

DATE_COL = "datadate"
TARGET_COL = "ret_eom"  # left untouched in cleaning
ID_COLS = ["id", "gvkey", "iid", "excntry", "tic", "cusip", "conm"]

# Processing parameters
CHUNKSIZE = 100_000  # reduced for lower peak memory
SAMPLE_PER_COL_PER_CHUNK = 4000
RANDOM_SEED = 42
FINAL_ABS_CLIP = 8.0  # robust z-score final guard; set None to disable

# Streaming behavior tweaks
FLUSH_EACH_CHUNK = True   # keep flushing so memory stable
ADD_MISSING_FLAGS = False  # disable missing flag columns to cut width & copies
CARRY_MAX_ROWS = 1_000_000  # tighter safety cap
CAST_FLOAT32 = True  # cast numeric features to float32 to halve memory

# Category specific clipping quantiles (low, high)
CLIP_RULES: Dict[str, Tuple[float, float]] = {
    "returns": (0.005, 0.995),
    "size": (0.01, 0.99),
    "price": (0.01, 0.99),
    "liquidity": (0.01, 0.99),
    "value": (0.01, 0.99),
    "quality": (0.01, 0.99),
    "growth": (0.01, 0.99),
    "leverage": (0.01, 0.99),
    "risk": (0.01, 0.99),
    "microstructure": (0.01, 0.99),
    "other": (0.01, 0.99),
}

# Keywords to detect categories when acronym mapping lacks explicit theme
CATEGORY_KEYWORDS: List[Tuple[str, str]] = [
    ("turnover", "liquidity"), ("dollar_vol", "liquidity"), ("volume", "liquidity"),
    ("bid", "microstructure"), ("ask", "microstructure"), ("bidsk", "microstructure"), ("amih", "microstructure"),
    ("me", "size"), ("market_equity", "size"), ("size", "size"),
    ("prc", "price"), ("price", "price"),
    ("bm", "value"), ("be_me", "value"), ("pb", "value"), ("intrinsic_value", "value"), ("bte_mev", "value"),
    ("roe", "quality"), ("roa", "quality"), ("gp_at", "quality"), ("gp_atl", "quality"), ("f_score", "quality"), ("o_score", "quality"),
    ("gr1", "growth"), ("gr2", "growth"), ("gr3", "growth"), ("growth", "growth"),
    ("lev", "leverage"), ("debt", "leverage"), ("netdebt", "leverage"),
    ("vol", "risk"), ("beta", "risk"), ("iskew", "risk"), ("coskew", "risk"), ("corr_", "risk"),
]

# Tags identifying return-like columns (treated tighter for clipping)
RETURNS_TAGS = ["ret_", "resff3", "rmax", "seas_"]

# Decide which transforms to apply per category
# log1p for positive, signed_log for symmetric heavy-tailed
LOG1P_CATEGORIES = {"size", "price", "liquidity", "microstructure"}
SIGNED_LOG_CATEGORIES = {"growth", "leverage"}

@dataclass
class ProfileStats:
    median: float
    q0005: float
    q01: float
    q99: float
    q995: float


def infer_category(column: str, theme_map: Dict[str, str]) -> str:
    """Infer feature category using explicit theme_map then keyword heuristics."""
    col = column.lower()
    if col in theme_map:
        t = theme_map[col]
        if t in CLIP_RULES:
            return t
    if any(tag in col for tag in RETURNS_TAGS):
        return "returns"
    for kw, cat in CATEGORY_KEYWORDS:
        if kw in col:
            return cat
    return "other"

__all__ = [
    'RAW_FILE','ACRONYM_FILE','PROFILE_JSON','OUTPUT_PARQUET','DATE_COL','TARGET_COL','ID_COLS',
    'CHUNKSIZE','SAMPLE_PER_COL_PER_CHUNK','RANDOM_SEED','FINAL_ABS_CLIP','CLIP_RULES','RETURNS_TAGS',
    'LOG1P_CATEGORIES','SIGNED_LOG_CATEGORIES','infer_category','ProfileStats'
]
