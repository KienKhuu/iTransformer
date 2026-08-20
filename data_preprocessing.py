import os
import pandas as pd
import numpy as np
import torch
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class StockDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
            seq_y, dtype=torch.float32
        )


def fetch_stock_data(ticker, start_date, end_date, save_path="data/stock_data.csv"):
    """
    Fetches OHLCV data from Yahoo Finance and saves it to a CSV.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(
            f"Failed to fetch data for {ticker}. Check ticker symbol and dates."
        )

    # Handle multi-index columns (yfinance sometimes returns multi-index for single tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the required OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    df.to_csv(save_path)
    print(f"Successfully downloaded {len(df)} days of data and saved to {save_path}")

    return save_path


def load_and_preprocess_data(
    csv_path, seq_len=96, pred_len=24, batch_size=32, train_ratio=0.8
):
    # 1. Load Data
    df = pd.read_csv(csv_path)

    # Keep only OHLCV (ignores Date column if it exists)
    features = ["Open", "High", "Low", "Close", "Volume"]
    df = df[features]

    # 2. Scale Data (Fit on train data only to prevent data leakage)
    train_size = int(len(df) * train_ratio)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(df.iloc[:train_size].values)
    test_data = scaler.transform(df.iloc[train_size:].values)

    # 3. Create Datasets and DataLoaders
    train_dataset = StockDataset(train_data, seq_len, pred_len)
    test_dataset = StockDataset(test_data, seq_len, pred_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader, scaler
