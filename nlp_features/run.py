from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

from . import config as C
from .dataset import read_textdata_parquets, FinBertChunkDataset, FinBertChunkIterableDataset
from .model import FinBertLightning
from .aggregate import (
    aggregate_filing,
    pca_fit_transform_embeddings,
    pca_transform_embeddings,
    save_pca,
    load_pca,
)
from .align import align_to_month


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FinBERT-based text features from TextData")
    p.add_argument("--text-root", type=str, default=str(C.TEXTDATA_DIR))
    p.add_argument("--outdir", type=str, default=str(C.OUTPUT_DIR))
    p.add_argument("--model", type=str, default=C.MODEL_NAME)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--max-tokens", type=int, default=C.MAX_TOKENS)
    p.add_argument("--doc-stride", type=int, default=C.DOC_STRIDE)
    p.add_argument("--pca-dim", type=int, default=C.PCA_OUT_DIM)
    p.add_argument("--pca-model", type=str, default=C.PCA_MODEL_NAME)
    p.add_argument("--reuse-pca", action="store_true")
    p.add_argument("--limit-docs", type=int, default=0)
    p.add_argument("--streaming", action="store_true", help="Use iterable dataset to reduce RAM usage")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    docs = read_textdata_parquets(Path(args.text_root))
    if args.limit_docs and args.limit_docs > 0:
        docs = docs.head(args.limit_docs).copy()

    meta = docs[[C.COL_GVKEY, C.COL_DATE, "length_words"]].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if args.streaming:
        ds = FinBertChunkIterableDataset(docs, tokenizer, max_len=args.max_tokens, doc_stride=args.doc_stride)
    else:
        ds = FinBertChunkDataset(docs, tokenizer, max_len=args.max_tokens, doc_stride=args.doc_stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=True)

    model = FinBertLightning(args.model)
    trainer = pl.Trainer(accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)

    probs_chunks: List[np.ndarray] = []
    cls_chunks: List[np.ndarray] = []
    doc_indices: List[int] = []

    model.eval()
    device = trainer.strategy.root_device if hasattr(trainer, "strategy") else torch.device("cpu")
    try:
        dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    except Exception:
        dev_name = str(device)
    print(f"Using device: {dev_name}")
    model.to(device)
    with torch.no_grad():
        for batch in dl:
            input_ids, attention_mask, doc_idx, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            probs_chunks.append(out["probs"].cpu().numpy())
            cls_chunks.append(out["cls"].cpu().numpy())
            doc_indices.extend(doc_idx.tolist())

    bucket: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for p, c, di in zip(probs_chunks, cls_chunks, doc_indices):
        bucket.setdefault(di, []).append({"probs": p, "cls": c})

    filings: List[Dict[str, object]] = []
    for di, chunks in bucket.items():
        gvkey = str(meta.iloc[di][C.COL_GVKEY])
        date = pd.Timestamp(meta.iloc[di][C.COL_DATE])
        lw = meta.iloc[di]["length_words"]
        filings.append(aggregate_filing(chunks, (gvkey, date, lw)))

    filing_df = pd.DataFrame(filings).sort_values([C.COL_GVKEY, C.COL_DATE]).reset_index(drop=True)

    pca_path = outdir / args.pca_model
    if args.reuse_pca and pca_path.exists():
        pca = load_pca(str(pca_path))
        filing_df = pca_transform_embeddings(filing_df, pca)
    else:
        filing_df, pca = pca_fit_transform_embeddings(filing_df, pca_dim=args.pca_dim)
        save_pca(pca, str(pca_path))

    filing_out = outdir / C.FILING_FEATURES_NAME
    filing_df.to_parquet(filing_out, index=False)

    monthly_df = align_to_month(filing_df)
    monthly_out = outdir / C.MONTHLY_FEATURES_NAME
    monthly_df.to_parquet(monthly_out, index=False)

    print(f"Saved filing-level features to: {filing_out}")
    print(f"Saved monthly features to: {monthly_out}")


if __name__ == "__main__":
    main()
