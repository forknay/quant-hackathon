from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib

from . import config as C


def aggregate_filing(chunk_outputs: List[Dict[str, np.ndarray]], doc_meta: Tuple[str, pd.Timestamp, int]) -> Dict[str, object]:
	gvkey, date, length_words = doc_meta
	probs = np.concatenate([o["probs"] for o in chunk_outputs], axis=0)
	cls = np.concatenate([o["cls"] for o in chunk_outputs], axis=0)

	neg, neu, pos = probs[:, 0], probs[:, 1], probs[:, 2]
	sentiment_mean = probs.mean(axis=0)
	sentiment_std = probs.std(axis=0)
	pos_ratio = (pos > 0.5).mean()
	neg_ratio = (neg > 0.5).mean()
	emb_mean = cls.mean(axis=0)

	return {
		"gvkey": str(gvkey),
		"date": pd.Timestamp(date),
		"length_words": int(length_words) if length_words == length_words else None,
		"sent_neg_mean": float(sentiment_mean[0]),
		"sent_neu_mean": float(sentiment_mean[1]),
		"sent_pos_mean": float(sentiment_mean[2]),
		"sent_neg_std": float(sentiment_std[0]),
		"sent_neu_std": float(sentiment_std[1]),
		"sent_pos_std": float(sentiment_std[2]),
		"pos_ratio": float(pos_ratio),
		"neg_ratio": float(neg_ratio),
		"emb_mean": emb_mean,
	}


def pca_fit_transform_embeddings(df: pd.DataFrame, pca_dim: int = C.PCA_OUT_DIM):
	embs = np.vstack(df["emb_mean"].values)
	pca = PCA(n_components=pca_dim, random_state=42)
	X = pca.fit_transform(embs)
	cols = [f"emb_pca_{i+1}" for i in range(X.shape[1])]
	for i, col in enumerate(cols):
		df[col] = X[:, i]
	df = df.drop(columns=["emb_mean"], errors="ignore")
	return df, pca


def pca_transform_embeddings(df: pd.DataFrame, pca: PCA):
	embs = np.vstack(df["emb_mean"].values)
	X = pca.transform(embs)
	cols = [f"emb_pca_{i+1}" for i in range(X.shape[1])]
	for i, col in enumerate(cols):
		df[col] = X[:, i]
	df = df.drop(columns=["emb_mean"], errors="ignore")
	return df


def save_pca(pca: PCA, path: str) -> None:
	joblib.dump(pca, path)


def load_pca(path: str) -> PCA:
	return joblib.load(path)

