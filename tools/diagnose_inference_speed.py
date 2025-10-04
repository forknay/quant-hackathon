#!/usr/bin/env python3
"""Diagnostic tool to identify Phase 1 inference bottlenecks.

This script runs a quick inference benchmark to measure:
- GPU utilization and memory usage
- Tokenization speed (CPU)
- Model forward pass speed (GPU)
- Data loading overhead
- Batch processing throughput

Usage:
    python tools/diagnose_inference_speed.py
"""

import time
import torch
import pandas as pd
from pathlib import Path
import psutil
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add the parent directory to the path so we can import nlp_features
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp_features.dataset import read_textdata_parquets, FinBertChunkDataset, FinBertChunkIterableDataset
from nlp_features.model import FinBertLightning
from nlp_features import config as C

def check_system_resources():
    """Check CPU, GPU, and memory availability."""
    print("=== System Resources ===")
    print(f"CPU cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
    print(f"RAM total: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_total:.1f} GB)")
    else:
        print("CUDA available: No (CPU only - this will be very slow)")
    print()

def benchmark_tokenization(texts, tokenizer, max_len=512, doc_stride=32, n_samples=100):
    """Benchmark tokenization speed."""
    print("=== Tokenization Benchmark ===")
    sample_texts = texts[:n_samples] if len(texts) > n_samples else texts
    
    # Time single tokenization
    start = time.time()
    for text in sample_texts:
        _ = tokenizer(text, truncation=True, max_length=max_len, 
                     stride=doc_stride, return_overflowing_tokens=True, 
                     padding="max_length", return_tensors="pt")
    single_time = time.time() - start
    
    print(f"Tokenized {len(sample_texts)} docs in {single_time:.2f}s")
    print(f"Tokenization speed: {len(sample_texts) / single_time:.1f} docs/sec")
    
    # Estimate total chunks
    total_chunks = 0
    for text in sample_texts[:10]:  # Sample to estimate
        enc = tokenizer(text, truncation=True, max_length=max_len, 
                       stride=doc_stride, return_overflowing_tokens=True, 
                       padding="max_length", return_tensors="pt")
        total_chunks += enc["input_ids"].size(0)
    avg_chunks = total_chunks / min(10, len(sample_texts))
    print(f"Average chunks per doc: {avg_chunks:.1f}")
    print()
    
    return avg_chunks

def benchmark_model_inference(model, device, batch_size=8, n_batches=10):
    """Benchmark pure model inference speed."""
    print("=== Model Inference Benchmark ===")
    
    # Create dummy batch
    dummy_input_ids = torch.randint(1, 1000, (batch_size, 512), device=device)
    dummy_attention_mask = torch.ones((batch_size, 512), device=device)
    
    model.eval()
    model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(n_batches):
            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    total_samples = n_batches * batch_size
    
    print(f"Processed {total_samples} samples in {elapsed:.2f}s")
    print(f"Model inference speed: {total_samples / elapsed:.1f} samples/sec")
    print(f"Batch processing speed: {n_batches / elapsed:.1f} batches/sec")
    
    if device.type == "cuda":
        mem_used = torch.cuda.max_memory_allocated(device) / 1e9
        mem_reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"GPU memory used: {mem_used:.1f} GB")
        print(f"GPU memory reserved: {mem_reserved:.1f} GB")
    print()
    
    return total_samples / elapsed

def benchmark_dataloader(dataloader, device, n_batches=20):
    """Benchmark DataLoader overhead."""
    print("=== DataLoader Benchmark ===")
    
    start = time.time()
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
            
        input_ids, attention_mask, doc_idx, chunk_idx = batch
        
        # Time data transfer to GPU
        transfer_start = time.time()
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_time = time.time() - transfer_start
        
        batch_count += 1
        if batch_idx == 0:
            print(f"First batch shape: {input_ids.shape}")
            print(f"Data transfer time for first batch: {transfer_time*1000:.1f}ms")
    
    elapsed = time.time() - start
    print(f"Processed {batch_count} batches in {elapsed:.2f}s")
    print(f"DataLoader speed: {batch_count / elapsed:.1f} batches/sec")
    print()
    
    return batch_count / elapsed

def diagnose_bottlenecks():
    """Run comprehensive diagnosis of inference bottlenecks."""
    print("Phase 1 Inference Speed Diagnosis")
    print("=" * 50)
    
    # Check system resources
    check_system_resources()
    
    # Load a small sample of data
    try:
        docs = read_textdata_parquets(Path("TextData"))
        if len(docs) > 500:
            docs = docs.head(500).copy()
        print(f"Loaded {len(docs)} documents for testing")
    except Exception as e:
        print(f"Error loading TextData: {e}")
        print("Make sure you have TextData/ folder with Parquet files")
        return
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME, use_fast=True)
    model = FinBertLightning(C.MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Benchmark tokenization
    avg_chunks = benchmark_tokenization(docs[C.COL_TEXT].tolist(), tokenizer)
    
    # Benchmark model inference
    inference_speed = benchmark_model_inference(model, device)
    
    # Create dataset and dataloader for testing
    print("Creating dataset...")
    ds = FinBertChunkDataset(docs, tokenizer, max_len=C.MAX_TOKENS, doc_stride=C.DOC_STRIDE)
    
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    
    # Benchmark dataloader
    dataloader_speed = benchmark_dataloader(dl, device)
    
    # Calculate estimates for full run
    print("=== Performance Estimates ===")
    total_chunks = len(docs) * avg_chunks
    print(f"Estimated total chunks: {total_chunks:.0f}")
    
    # Estimate time based on model inference speed
    model_time_estimate = total_chunks / inference_speed
    print(f"Model inference time estimate: {model_time_estimate/60:.1f} minutes")
    
    # Estimate time based on dataloader speed (includes tokenization + transfer)
    batches_needed = total_chunks / 8  # assuming batch size 8
    dataloader_time_estimate = batches_needed / dataloader_speed
    print(f"Full pipeline time estimate: {dataloader_time_estimate/60:.1f} minutes")
    
    # Bottleneck analysis
    print("\n=== Bottleneck Analysis ===")
    if not torch.cuda.is_available():
        print("⚠️  MAJOR BOTTLENECK: No GPU detected. CPU inference is 10-100x slower.")
        print("   Solution: Attach a GPU in Lightning Studio")
    
    if inference_speed < 50:
        print("⚠️  Slow model inference detected")
        print("   Solutions: Use --fp16 flag, increase batch size, check GPU utilization")
    
    if dataloader_speed < 5:
        print("⚠️  Slow data loading detected")
        print("   Solutions: Use --streaming, reduce --workers, check tokenization speed")
    
    if avg_chunks > 10:
        print("⚠️  Very long documents detected")
        print(f"   Average {avg_chunks:.1f} chunks per doc creates many batches")
        print("   Solutions: Use --doc-stride 0 for testing, or --limit-docs for smaller runs")
    
    print("\n=== Recommended Settings ===")
    if torch.cuda.is_available():
        print("For fastest inference:")
        print("  python -m nlp_features.run --streaming --fp16 --batch-size 16 --workers 2 --doc-stride 32")
        print("If memory constrained:")
        print("  python -m nlp_features.run --streaming --fp16 --batch-size 8 --workers 0 --doc-stride 0")
    else:
        print("CPU-only (will be slow):")
        print("  python -m nlp_features.run --streaming --batch-size 4 --workers 0 --doc-stride 0")

if __name__ == "__main__":
    diagnose_bottlenecks()