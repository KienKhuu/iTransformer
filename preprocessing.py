import yfinance as yf
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data = data.dropna()
    print(f"Fetched {len(data)} rows of data.")
    return data


def prepare_sequences(data, seq_len, pred_len, train_ratio=0.7, val_ratio=0.15):
    """
    Tạo X_enc (lịch sử) và Y (tương lai). Đã loại bỏ hoàn toàn X_dec.
    """
    train_split_idx = int(len(data) * train_ratio)

    train_data = data.iloc[:train_split_idx].values
    scaler = StandardScaler()
    scaler.fit(train_data)

    scaled_data = scaler.transform(data.values)

    X_enc, Y = [], []

    for i in range(len(scaled_data) - seq_len - pred_len + 1):
        x_enc = scaled_data[i : i + seq_len]
        y_true = scaled_data[i + seq_len : i + seq_len + pred_len]

        X_enc.append(x_enc)
        Y.append(y_true)

    X_enc = torch.tensor(np.array(X_enc), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    total_seqs = len(X_enc)
    train_end = int(total_seqs * train_ratio)
    val_end = int(total_seqs * (train_ratio + val_ratio))

    return X_enc, Y, scaler, train_end, val_end
