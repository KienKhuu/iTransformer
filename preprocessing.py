import yfinance as yf
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance and adds Log Returns.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()

    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))

    data = data.dropna()
    print(f"Fetched {len(data)} rows of data.")
    return data


def prepare_sequences(data, seq_len, label_len, pred_len, split_ratio=0.8):
    """
    Sửa lỗi Data Leakage bằng cách chỉ fit Scaler trên tập Train.
    """
    split_idx = int(len(data) * split_ratio)
    train_data = data.iloc[:split_idx].values

    scaler = StandardScaler()
    scaler.fit(train_data)

    scaled_data = scaler.transform(data.values)

    X_enc, X_dec, Y = [], [], []

    for i in range(len(scaled_data) - seq_len - pred_len + 1):
        x_enc = scaled_data[i : i + seq_len]
        y_true = scaled_data[i + seq_len : i + seq_len + pred_len]

        x_dec_known = scaled_data[i + seq_len - label_len : i + seq_len]
        x_dec_zeros = np.zeros((pred_len, scaled_data.shape[1]))
        x_dec = np.vstack([x_dec_known, x_dec_zeros])

        X_enc.append(x_enc)
        X_dec.append(x_dec)
        Y.append(y_true)

    X_enc = torch.tensor(np.array(X_enc), dtype=torch.float32)
    X_dec = torch.tensor(np.array(X_dec), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    seq_split_idx = int(len(X_enc) * split_ratio)

    return X_enc, X_dec, Y, scaler, seq_split_idx
