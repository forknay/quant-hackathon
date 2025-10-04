from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from . import config as C


def normalize_text(s: str) -> str:
	if not isinstance(s, str):
		return ""
	return " ".join(s.split())


def read_textdata_parquets(root: Path) -> pd.DataFrame:
	files = list(Path(root).rglob("*.parquet"))
	if not files:
		raise FileNotFoundError(f"No parquet files found under {root}")
	dfs = []
	for fp in files:
		try:
			df = pd.read_parquet(fp, columns=[C.COL_DATE, C.COL_GVKEY, C.COL_TEXT])
			df["source_file"] = str(fp)
			dfs.append(df)
		except Exception:
			continue
	if not dfs:
		raise RuntimeError("No readable parquet files for required columns.")
	out = pd.concat(dfs, ignore_index=True)
	out[C.COL_DATE] = pd.to_datetime(out[C.COL_DATE], errors="coerce")
	out[C.COL_GVKEY] = out[C.COL_GVKEY].astype(str)
	out[C.COL_TEXT] = out[C.COL_TEXT].map(normalize_text)
	out = out.dropna(subset=[C.COL_DATE])
	out = out[out[C.COL_TEXT].str.len() > 0]
	out["length_words"] = out[C.COL_TEXT].str.split().str.len().astype("Int32")
	return out


class FinBertChunkDataset(Dataset):
	def __init__(self, docs: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int = C.MAX_TOKENS, doc_stride: int = C.DOC_STRIDE):
		self.tokenizer = tokenizer
		self.max_len = max_len
		self.doc_stride = doc_stride
		self.docs_meta: List[Tuple[str, pd.Timestamp]] = []
		self.chunks: List[Tuple[torch.Tensor, torch.Tensor, int, int]] = []

		for _, row in docs.iterrows():
			text = row[C.COL_TEXT]
			enc = tokenizer(
				text,
				truncation=True,
				max_length=max_len,
				stride=doc_stride,
				return_overflowing_tokens=True,
				padding="max_length",
				return_tensors="pt",
			)
			num_chunks = enc["input_ids"].size(0)
			doc_index = len(self.docs_meta)
			for chunk_idx in range(num_chunks):
				self.chunks.append(
					(
						enc["input_ids"][chunk_idx],
						enc["attention_mask"][chunk_idx],
						doc_index,
						chunk_idx,
					)
				)
			self.docs_meta.append((str(row[C.COL_GVKEY]), pd.Timestamp(row[C.COL_DATE])))

	def __len__(self) -> int:
		return len(self.chunks)

	def __getitem__(self, idx: int):
		return self.chunks[idx]

	def doc_count(self) -> int:
		return len(self.docs_meta)

