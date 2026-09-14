import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from models.iTransformer import Model as iTransformerModel
from preprocessing import fetch_stock_data, prepare_sequences


# ---------------------------------------------------------
# Cấu hình và Tiện ích
# ---------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sinusoidal_encoding(seq_len, d_mark):
    """
    Tạo ma trận Sinusoidal Encoding dựa trên công thức của Transformer gốc.
    Shape trả về: [seq_len, d_mark]
    """
    pe = torch.zeros(seq_len, d_mark)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_mark, 2).float() * (-np.log(10000.0) / d_mark)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MockConfig:
    def __init__(self, seq_len, pred_len, num_variates):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 256
        self.factor = 1
        self.dropout = 0.1
        self.embed = "timeF"
        self.freq = "d"
        self.activation = "gelu"
        self.output_attention = False
        self.use_norm = True
        self.class_strategy = "projection"


# ---------------------------------------------------------
# Training, Evaluation & Baseline
# ---------------------------------------------------------
def naive_baseline(X_enc, Y_true, scaler, pred_len, close_idx=3):
    B, _, N = X_enc.shape
    last_known_scaled = X_enc[:, -1, :].numpy()
    last_known_unscaled = scaler.inverse_transform(last_known_scaled)
    last_close = last_known_unscaled[:, close_idx]

    preds_close = np.repeat(last_close[:, np.newaxis], pred_len, axis=1)

    y_unscaled = scaler.inverse_transform(Y_true.reshape(-1, N).numpy()).reshape(
        B, pred_len, N
    )
    actual_close = y_unscaled[:, :, close_idx]

    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))

    return preds_close, actual_close, mae, rmse


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    device="cpu",
    patience=5,
    close_idx=3,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    # Khởi tạo 1 lần Sinusoidal Tensor cho toàn bộ seq_len để tái sử dụng
    seq_len = train_loader.dataset.tensors[0].shape[1]
    base_sinusoidal = sinusoidal_encoding(seq_len, d_mark=4).to(device)

    for epoch in range(epochs):
        model.train()
        for batch_x_enc, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x_enc, batch_y = batch_x_enc.to(device), batch_y.to(device)

            # Mở rộng Sinusoidal cho khớp với Batch Size
            batch_x_mark_enc = base_sinusoidal.unsqueeze(0).repeat(
                batch_x_enc.shape[0], 1, 1
            )

            # Đẩy None vào vị trí của x_dec và x_mark_dec
            outputs = model(batch_x_enc, batch_x_mark_enc, None, None)
            outputs = outputs[:, -batch_y.shape[1] :, :]

            loss = criterion(outputs[:, :, close_idx], batch_y[:, :, close_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_enc, batch_y in val_loader:
                batch_x_enc, batch_y = batch_x_enc.to(device), batch_y.to(device)
                batch_x_mark_enc = base_sinusoidal.unsqueeze(0).repeat(
                    batch_x_enc.shape[0], 1, 1
                )

                outputs = model(batch_x_enc, batch_x_mark_enc, None, None)
                outputs = outputs[:, -batch_y.shape[1] :, :]

                v_loss = criterion(outputs[:, :, close_idx], batch_y[:, :, close_idx])
                val_loss += v_loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_model_wts)
                break


def evaluate_and_predict(
    model, X_enc, Y_true, scaler, pred_len, close_idx=3, device="cpu"
):
    model.to(device)
    model.eval()

    seq_len = X_enc.shape[1]
    base_sinusoidal = sinusoidal_encoding(seq_len, d_mark=4).to(device)

    with torch.no_grad():
        X_enc = X_enc.to(device)
        X_mark_enc = base_sinusoidal.unsqueeze(0).repeat(X_enc.shape[0], 1, 1)

        preds = model(X_enc, X_mark_enc, None, None)
        preds = preds[:, -pred_len:, :].cpu()

    B, _, N = preds.shape
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, N)).reshape(
        B, pred_len, N
    )
    y_test_unscaled = scaler.inverse_transform(Y_true.reshape(-1, N).numpy()).reshape(
        B, pred_len, N
    )

    preds_close = preds_unscaled[:, :, close_idx]
    actual_close = y_test_unscaled[:, :, close_idx]

    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))

    return preds_close, actual_close, mae, rmse


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using Device: {device} ---")

    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2026-01-01"
    SEQ_LEN = 60
    BATCH_SIZE = 32
    EPOCHS = 100
    SEEDS = [42, 123, 2026]
    PRED_LENS = [1, 5]

    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    NUM_VARIATES = df.shape[1]
    CLOSE_IDX = df.columns.get_loc("Close")

    for PRED_LEN in PRED_LENS:
        print(f"\n=======================================================")
        print(f" EXPERIMENT: PREDICTION LENGTH = {PRED_LEN} ")
        print(f"=======================================================")

        X_enc, Y, scaler, train_end, val_end = prepare_sequences(
            df, SEQ_LEN, PRED_LEN, train_ratio=0.7, val_ratio=0.1
        )

        X_enc_train, Y_train = X_enc[:train_end], Y[:train_end]
        X_enc_val, Y_val = X_enc[train_end:val_end], Y[train_end:val_end]
        X_enc_test, Y_test = X_enc[val_end:], Y[val_end:]

        train_loader = DataLoader(
            TensorDataset(X_enc_train, Y_train), batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_enc_val, Y_val), batch_size=BATCH_SIZE, shuffle=False
        )

        naive_preds, actuals, naive_mae, naive_rmse = naive_baseline(
            X_enc_test, Y_test, scaler, PRED_LEN, CLOSE_IDX
        )
        print(f"[NAIVE BASELINE] MAE: {naive_mae:.4f} | RMSE: {naive_rmse:.4f}")

        itrans_maes, itrans_rmses = [], []
        best_preds = None

        for seed in SEEDS:
            set_seed(seed)
            configs = MockConfig(
                seq_len=SEQ_LEN, pred_len=PRED_LEN, num_variates=NUM_VARIATES
            )
            model = iTransformerModel(configs)

            train_model(
                model,
                train_loader,
                val_loader,
                epochs=EPOCHS,
                lr=0.001,
                device=device,
                patience=5,
                close_idx=CLOSE_IDX,
            )

            i_preds, _, i_mae, i_rmse = evaluate_and_predict(
                model, X_enc_test, Y_test, scaler, PRED_LEN, CLOSE_IDX, device=device
            )

            itrans_maes.append(i_mae)
            itrans_rmses.append(i_rmse)
            best_preds = i_preds

            print(f" - Seed {seed:4d} -> MAE: {i_mae:.4f} | RMSE: {i_rmse:.4f}")

        mean_mae, std_mae = np.mean(itrans_maes), np.std(itrans_maes)
        mean_rmse, std_rmse = np.mean(itrans_rmses), np.std(itrans_rmses)

        print(
            f"\n[iTRANSFORMER] MAE: {mean_mae:.4f} ± {std_mae:.4f} | RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}"
        )

        plt.figure(figsize=(12, 6))

        if PRED_LEN == 1:
            plot_range = 100
            plt.plot(
                actuals.flatten()[-plot_range:],
                label="Actual Close",
                color="black",
                marker=".",
            )
            plt.plot(
                naive_preds.flatten()[-plot_range:],
                label="Naive Baseline",
                color="green",
                linestyle="--",
                alpha=0.7,
            )
            plt.plot(
                best_preds.flatten()[-plot_range:],
                label=f"iTransformer",
                color="red",
                alpha=0.8,
            )
            plt.title(f"{TICKER} - 1-Day Ahead Prediction (Last {plot_range} days)")
        else:
            sample_idx = -1
            hist_unscaled = scaler.inverse_transform(X_enc_test[sample_idx].numpy())[
                :, CLOSE_IDX
            ]
            time_hist = np.arange(SEQ_LEN)
            time_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

            plt.plot(time_hist, hist_unscaled, label="Historical Close", color="gray")
            plt.plot(
                time_pred,
                actuals[sample_idx],
                label="Actual Future",
                color="black",
                marker="o",
            )
            plt.plot(
                time_pred,
                naive_preds[sample_idx],
                label="Naive Baseline",
                color="green",
                linestyle="--",
                marker="x",
            )
            plt.plot(
                time_pred,
                best_preds[sample_idx],
                label="iTransformer",
                color="red",
                marker="s",
            )
            plt.title(f"{TICKER} - {PRED_LEN}-Days Horizon Prediction")

        plt.xlabel("Time Steps")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"figure/prediction_{PRED_LEN}day_chart.png", dpi=300)
        plt.close()
