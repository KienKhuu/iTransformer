import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import fetch_stock_data, prepare_sequences
from model.iTransformer import Model as iTransformerModel

from utils.cl import DataAugmentation, ProjectionHead, InfoNCELoss
from model.iTransformer_CL import iTransformer_CL


# ---------------------------------------------------------
# Helper Functions & Configs
# ---------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sinusoidal_encoding(seq_len, d_mark):
    pe = torch.zeros(seq_len, d_mark)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_mark, 2).float() * (-np.log(10000.0) / d_mark)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MockConfig_iTransformer:
    def __init__(self, seq_len, pred_len, num_variates):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = num_variates
        self.dec_in = num_variates
        self.c_out = num_variates
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
# Huấn luyện Mô hình Cơ bản (Baseline)
# ---------------------------------------------------------
def train_baseline(
    model, train_loader, val_loader, epochs, lr, device, patience, close_idx
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    seq_len = train_loader.dataset.tensors[0].shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    for epoch in range(epochs):
        model.train()
        for batch_x_enc, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x_enc, batch_y = batch_x_enc.to(device), batch_y.to(device)
            batch_x_mark_enc = base_sinusoidal.unsqueeze(0).repeat(
                batch_x_enc.shape[0], 1, 1
            )

            outputs = model(batch_x_enc, batch_x_mark_enc, None, None)
            outputs = outputs[:, -batch_y.shape[1] :, :]
            loss = criterion(outputs[:, :, close_idx], batch_y[:, :, close_idx])

            loss.backward()
            optimizer.step()

        # Validation
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
                val_loss += criterion(
                    outputs[:, :, close_idx], batch_y[:, :, close_idx]
                ).item()

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


# ---------------------------------------------------------
# Huấn luyện Mô hình với Contrastive Learning
# ---------------------------------------------------------
def train_with_cl(
    model,
    proj_head,
    train_loader,
    val_loader,
    epochs,
    lr,
    lamda,
    device,
    patience,
    close_idx,
):
    criterion_forecast = nn.MSELoss()
    criterion_cl = InfoNCELoss(temperature=0.2)
    augmentor = DataAugmentation(jitter_sigma=0.03, mask_ratio=0.10)

    # Optimizer tối ưu chung cả Model Forecast và Projection Head
    optimizer = optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()), lr=lr
    )

    model.to(device)
    proj_head.to(device)
    augmentor.to(device)
    criterion_cl.to(device)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    seq_len = train_loader.dataset.tensors[0].shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    for epoch in range(epochs):
        model.train()
        proj_head.train()
        for batch_x_enc, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x_enc, batch_y = batch_x_enc.to(device), batch_y.to(device)
            batch_x_mark_enc = base_sinusoidal.unsqueeze(0).repeat(
                batch_x_enc.shape[0], 1, 1
            )

            # 1. Sinh 2 Augmented Views
            x1, x2 = augmentor(batch_x_enc)

            # 2. Encode Original Input cho Forecasting
            h_orig, means, stdev = model.encode(batch_x_enc, batch_x_mark_enc)
            preds = model.forecast_from_representation(h_orig, means, stdev)

            # Tính Forecasting Loss
            loss_forecast = criterion_forecast(
                preds[:, :, close_idx], batch_y[:, :, close_idx]
            )

            # 3. Encode Augmented Views cho Contrastive Learning
            h1, _, _ = model.encode(x1, batch_x_mark_enc)
            h2, _, _ = model.encode(x2, batch_x_mark_enc)

            # iTransformer shape của h là [B, N, d_model]. Ta Mean Pooling dọc theo variates (N)
            h1_pooled = h1.mean(dim=1)  # [B, d_model]
            h2_pooled = h2.mean(dim=1)

            # 4. Projection & L2 Normalization
            z1 = F.normalize(proj_head(h1_pooled), dim=1)
            z2 = F.normalize(proj_head(h2_pooled), dim=1)

            # 5. Tính CL Loss
            loss_cl = criterion_cl(z1, z2)

            # 6. Combine Losses & Backprop
            loss_total = loss_forecast + (lamda * loss_cl)
            loss_total.backward()
            optimizer.step()

        # --- Validation (Chỉ đánh giá Forecasting Loss) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_enc, batch_y in val_loader:
                batch_x_enc, batch_y = batch_x_enc.to(device), batch_y.to(device)
                batch_x_mark_enc = base_sinusoidal.unsqueeze(0).repeat(
                    batch_x_enc.shape[0], 1, 1
                )

                h_orig, means, stdev = model.encode(batch_x_enc, batch_x_mark_enc)
                preds = model.forecast_from_representation(h_orig, means, stdev)

                val_loss += criterion_forecast(
                    preds[:, :, close_idx], batch_y[:, :, close_idx]
                ).item()

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


# ---------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------
def evaluate_and_predict(
    model, X_enc, Y_true, scaler, pred_len, close_idx=3, device="cpu", is_cl=False
):
    model.to(device)
    model.eval()
    seq_len = X_enc.shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    with torch.no_grad():
        X_enc = X_enc.to(device)
        X_mark_enc = base_sinusoidal.unsqueeze(0).repeat(X_enc.shape[0], 1, 1)

        if is_cl:
            h, means, stdev = model.encode(X_enc, X_mark_enc)
            preds = model.forecast_from_representation(h, means, stdev)
        else:
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
    SEEDS = [42, 2024, 8888]
    PRED_LENS = [1]

    # CL Hyperparameters
    LAMDA = 0.1

    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    NUM_VARIATES = df.shape[1]
    CLOSE_IDX = df.columns.get_loc("Close")

    for PRED_LEN in PRED_LENS:
        print(f"\n=======================================================")
        print(f" EXPERIMENT: PREDICTION LENGTH = {PRED_LEN} ")
        print(f"=======================================================")

        X_enc, Y, scaler, train_end, val_end = prepare_sequences(
            df, SEQ_LEN, PRED_LEN, train_ratio=0.7, val_ratio=0.15
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

        results = {
            "iTransformer": {"maes": [], "rmses": [], "best_preds": None},
            "iTransformer+CL": {"maes": [], "rmses": [], "best_preds": None},
        }

        for seed in SEEDS:
            print(f"\n--- Running Seed: {seed} ---")
            set_seed(seed)
            configs = MockConfig_iTransformer(SEQ_LEN, PRED_LEN, NUM_VARIATES)

            # 1. Train Original iTransformer
            model_base = iTransformerModel(configs)
            train_baseline(
                model_base,
                train_loader,
                val_loader,
                EPOCHS,
                0.001,
                device,
                5,
                CLOSE_IDX,
            )
            preds_base, actuals, mae_base, rmse_base = evaluate_and_predict(
                model_base,
                X_enc_test,
                Y_test,
                scaler,
                PRED_LEN,
                CLOSE_IDX,
                device,
                is_cl=False,
            )
            results["iTransformer"]["maes"].append(mae_base)
            results["iTransformer"]["rmses"].append(rmse_base)
            results["iTransformer"]["best_preds"] = preds_base
            print(f" iTransformer    -> MAE: {mae_base:.4f} | RMSE: {rmse_base:.4f}")

            # 2. Train iTransformer + Contrastive Learning
            set_seed(seed)  # Reset seed
            model_cl = iTransformer_CL(configs)
            proj_head = ProjectionHead(
                input_dim=configs.d_model, hidden_dim=64, out_dim=32
            )

            train_with_cl(
                model_cl,
                proj_head,
                train_loader,
                val_loader,
                EPOCHS,
                0.001,
                LAMDA,
                device,
                5,
                CLOSE_IDX,
            )
            preds_cl, _, mae_cl, rmse_cl = evaluate_and_predict(
                model_cl,
                X_enc_test,
                Y_test,
                scaler,
                PRED_LEN,
                CLOSE_IDX,
                device,
                is_cl=True,
            )
            results["iTransformer+CL"]["maes"].append(mae_cl)
            results["iTransformer+CL"]["rmses"].append(rmse_cl)
            results["iTransformer+CL"]["best_preds"] = preds_cl
            print(f" iTransformer+CL -> MAE: {mae_cl:.4f} | RMSE: {rmse_cl:.4f}")

        # --- In Tổng kết ---
        print(f"\n[FINAL RESULTS PRED_LEN={PRED_LEN}]")
        for m_name in results.keys():
            mean_mae, std_mae = np.mean(results[m_name]["maes"]), np.std(
                results[m_name]["maes"]
            )
            mean_rmse, std_rmse = np.mean(results[m_name]["rmses"]), np.std(
                results[m_name]["rmses"]
            )
            print(
                f"{m_name:15s} -> MAE: {mean_mae:.4f} ± {std_mae:.4f} | RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}"
            )

        # --- Plotting ---
        plt.figure(figsize=(14, 7))
        if PRED_LEN == 1:
            plot_range = 100
            plt.plot(
                actuals.flatten()[-plot_range:],
                label="Actual Close",
                color="black",
                marker=".",
                linewidth=2,
            )
            plt.plot(
                results["iTransformer"]["best_preds"].flatten()[-plot_range:],
                label="iTransformer",
                color="blue",
                alpha=0.8,
            )
            plt.plot(
                results["iTransformer+CL"]["best_preds"].flatten()[-plot_range:],
                label="iTransformer + CL",
                color="red",
                alpha=0.8,
            )
        else:
            sample_idx = -1
            time_pred = np.arange(PRED_LEN)
            plt.plot(
                time_pred,
                actuals[sample_idx],
                label="Actual Future",
                color="black",
                marker="o",
                linewidth=2,
            )
            plt.plot(
                time_pred,
                results["iTransformer"]["best_preds"][sample_idx],
                label="iTransformer",
                color="blue",
                marker="x",
            )
            plt.plot(
                time_pred,
                results["iTransformer+CL"]["best_preds"][sample_idx],
                label="iTransformer + CL",
                color="red",
                marker="s",
            )

        plt.title(f"{TICKER} - {PRED_LEN}-Day Ahead Prediction (CL vs Baseline)")
        plt.xlabel("Time Steps")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"prediction_{PRED_LEN}day_CL_compare.png", dpi=300)
        plt.close()
