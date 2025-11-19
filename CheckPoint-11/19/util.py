import os, json, math, random
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import Counter

MODEL_NAME = "rafalposwiata/deproberta-large-depression"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

MAX_TOKENS = 512

def chunk_text(text, max_len=MAX_TOKENS, tokenizer=tokenizer):
    # split long text into chunks that the model can handle
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunk_size = max_len - 2  # leave room for special tokens
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_ids = tokens[i:i+chunk_size]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks

def inspect_dataset(ds, label_col_candidates=None):
    print("Dataset preview (first 3 rows):")
    for i in range(min(3, len(ds))):
        print(ds[i])
    print("\nColumns:", ds.column_names)
    # find label column
    if label_col_candidates is None:
        label_col_candidates = ['label','labels','target','truth','y']
    found = [c for c in ds.column_names if c in label_col_candidates]
    if found:
        print("Likely label column(s):", found)
    else:
        print("No obvious label column found. All columns:", ds.column_names)
    # nulls and class balance
    for c in ds.column_names:
        values = [r[c] for r in ds.select(range(min(1000, len(ds))))]
        nnull = sum(1 for v in values if v is None)
        if nnull>0:
            print(f"Warning: Column {c} has nulls in sample of 1000 (count {nnull})")
    return
