"""Chronological splits by target date; earlier context is permitted at split edges."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def fetch_stock_data(ticker, start_date, end_date):
    import yfinance as yf

    data = yf.download(ticker, start=start_date, end=end_date,
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()


def load_csv(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if data.index.has_duplicates:
        raise ValueError("Duplicate dates in CSV")
    return data


def prepare_sequences(data, seq_len, pred_len, train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))
    if not (0 < train_cut < val_cut < n) or train_cut < seq_len + pred_len:
        raise ValueError("Insufficient data for requested chronology and window sizes")
    scaler = StandardScaler().fit(data.iloc[:train_cut].to_numpy())
    scaled = scaler.transform(data.to_numpy()).astype(np.float32)
    xs, ys, groups = [], [], []
    for start in range(n - seq_len - pred_len + 1):
        label_start = start + seq_len
        label_end = label_start + pred_len  # exclusive
        if label_end <= train_cut:
            split = "train"
        elif label_start >= train_cut and label_end <= val_cut:
            split = "val"
        elif label_start >= val_cut:
            split = "test"
        else:
            continue  # horizon straddles a boundary
        xs.append(scaled[start:label_start])
        ys.append(scaled[label_start:label_end])
        groups.append(split)
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))
    result = {}
    for split in ("train", "val", "test"):
        indices = [i for i, group in enumerate(groups) if group == split]
        if not indices:
            raise ValueError(f"No {split} windows; use more dates or a shorter horizon")
        result[split] = (x[indices], y[indices])
    return result, scaler
