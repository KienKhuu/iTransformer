import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

# Import the official models from the cloned repository
from models.iTransformer import Model as iTransformerModel
from preprocessing import fetch_stock_data, prepare_sequences


class MockConfig:
    def __init__(self, seq_len, label_len, pred_len, num_variates):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = num_variates
        self.dec_in = num_variates
        self.c_out = num_variates
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.dropout = 0.1
        self.embed = "timeF"
        self.freq = "d"
        self.activation = "gelu"
        self.output_attention = False
        self.use_norm = True
        self.class_strategy = "projection"


def train_model(model, train_loader, epochs=50, lr=0.001, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_x_enc, batch_x_dec, batch_y in train_loader:
            optimizer.zero_grad()

            batch_x_enc = batch_x_enc.to(device)
            batch_x_dec = batch_x_dec.to(device)
            batch_y = batch_y.to(device)

            # Dummy time markers (no Time Covariates yet)
            batch_x_mark_enc = torch.zeros(
                batch_x_enc.shape[0], batch_x_enc.shape[1], 4
            ).to(device)
            batch_x_mark_dec = torch.zeros(
                batch_x_dec.shape[0], batch_x_dec.shape[1], 4
            ).to(device)

            outputs = model(
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec
            )
            outputs = outputs[:, -batch_y.shape[1] :, :]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


def evaluate_and_predict(
    model, X_enc, X_dec, Y_true, scaler, pred_len, close_idx=3, device="cpu"
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_enc = X_enc.to(device)
        X_dec = X_dec.to(device)

        X_mark_enc = torch.zeros(X_enc.shape[0], X_enc.shape[1], 4).to(device)
        X_mark_dec = torch.zeros(X_dec.shape[0], X_dec.shape[1], 4).to(device)

        preds = model(X_enc, X_mark_enc, X_dec, X_mark_dec)
        preds = preds[:, -pred_len:, :]

    B, _, N = preds.shape
    preds = preds.cpu()

    # Inverse transform
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, N)).reshape(
        B, pred_len, N
    )
    y_test_unscaled = scaler.inverse_transform(Y_true.reshape(-1, N)).reshape(
        B, pred_len, N
    )

    preds_close = preds_unscaled[:, :, close_idx]
    actual_close = y_test_unscaled[:, :, close_idx]

    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))

    return preds_close, actual_close, mae, rmse


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using Device: {device} ---")

    # 1. Setup Parameters
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2026-01-01"
    SEQ_LEN = 60  # Đổi thành 60 theo yêu cầu
    LABEL_LEN = 30  # Thường bằng 1/2 seq_len
    PRED_LEN = 1  # Dự báo 1 ngày tới
    BATCH_SIZE = 32
    SPLIT_RATIO = 0.8
    EPOCHS = 50  # Giảm epoch để tránh overfit vì pred_len = 1 dễ học thuộc

    # 2. Fetch Data
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    NUM_VARIATES = df.shape[
        1
    ]  # Lúc này là 6 (Open, High, Low, Close, Volume, Log_Return)
    CLOSE_IDX = 3  # Position of 'Close' vẫn là 3

    X_enc, X_dec, Y, scaler, seq_split_idx = prepare_sequences(
        df, SEQ_LEN, LABEL_LEN, PRED_LEN, SPLIT_RATIO
    )

    # Train/Test Split an toàn (Không leakage)
    X_enc_train, X_dec_train, Y_train = (
        X_enc[:seq_split_idx],
        X_dec[:seq_split_idx],
        Y[:seq_split_idx],
    )
    X_enc_test, X_dec_test, Y_test = (
        X_enc[seq_split_idx:],
        X_dec[seq_split_idx:],
        Y[seq_split_idx:],
    )

    train_dataset = TensorDataset(X_enc_train, X_dec_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize & Train
    configs = MockConfig(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        num_variates=NUM_VARIATES,
    )

    print("\n--- Training iTransformer ---")
    itransformer_model = iTransformerModel(configs)
    train_model(itransformer_model, train_loader, epochs=EPOCHS, device=device)

    # Lưu model
    model_save_path = "saved/itransformer_aapl_1day.pth"
    torch.save(itransformer_model.state_dict(), model_save_path)
    print(f"\n[+] Save model at: {model_save_path}")

    # 4. Evaluation
    print("\n--- Evaluation on Test Set ('Close' Price) ---")
    i_preds, actuals, i_mae, i_rmse = evaluate_and_predict(
        itransformer_model,
        X_enc_test,
        X_dec_test,
        Y_test,
        scaler,
        PRED_LEN,
        CLOSE_IDX,
        device=device,
    )
    print(f"Official iTransformer -> MAE: {i_mae:.4f}, RMSE: {i_rmse:.4f}")

    # 5. Plotting (Thay đổi logic plot cho PRED_LEN = 1)
    # Vì mỗi prediction chỉ có 1 ngày, ta sẽ nối tất cả các dự báo lại
    # và so sánh với giá thực tế của 100 ngày giao dịch cuối cùng trong tập Test.

    plot_actuals = actuals.flatten().numpy()
    plot_preds = i_preds.flatten().numpy()

    # Lấy 100 ngày cuối cùng để vẽ cho dễ nhìn
    plot_range = 100

    plt.figure(figsize=(12, 6))
    plt.plot(
        plot_actuals[-plot_range:],
        label="Actual Close Price",
        marker=".",
        color="black",
    )
    plt.plot(
        plot_preds[-plot_range:],
        label="iTransformer Prediction",
        marker=".",
        color="red",
        alpha=0.7,
    )

    plt.title(f"{TICKER} - 1-Day Ahead Prediction (Last {plot_range} Days of Test Set)")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    chart_save_path = "prediction_1day_chart.png"
    plt.savefig(chart_save_path, dpi=300)
    print(f"[+] Save chart at: {chart_save_path}")

    plt.show()
