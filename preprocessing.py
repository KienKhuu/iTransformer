import yfinance as yf
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Handle multi-index columns in newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"Fetched {len(data)} rows of data.")
    return data


def prepare_sequences(data, seq_len, label_len, pred_len):
    """
    Scales data and creates sequences required by the official repo models.
    Official models need:
    - x_enc: Encoder input (seq_len)
    - x_dec: Decoder input (label_len + pred_len)
    - y_true: Ground truth (pred_len)
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)

    X_enc, X_dec, Y = [], [], []

    for i in range(len(scaled_data) - seq_len - pred_len + 1):
        # Encoder input
        x_enc = scaled_data[i : i + seq_len]

        # Ground truth
        y_true = scaled_data[i + seq_len : i + seq_len + pred_len]

        # Decoder input: 'label_len' history + 'pred_len' zeros
        x_dec_known = scaled_data[i + seq_len - label_len : i + seq_len]
        x_dec_zeros = np.zeros((pred_len, scaled_data.shape[1]))
        x_dec = np.vstack([x_dec_known, x_dec_zeros])

        X_enc.append(x_enc)
        X_dec.append(x_dec)
        Y.append(y_true)

    X_enc = torch.tensor(np.array(X_enc), dtype=torch.float32)
    X_dec = torch.tensor(np.array(X_dec), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    return X_enc, X_dec, Y, scaler
