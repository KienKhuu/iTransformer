import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from model.iTransformer import Model as iTransformerModel
from model.PatchTST import Model as PatchTSTModel
from model.LSTM import Model as LSTMModel
from model.TSMixer import Model as TSMixerModel
from preprocessing import fetch_stock_data, prepare_sequences
from contrastive.trainer import contrastive_pretrain


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
    def __init__(self, seq_len, pred_len, num_variates=None):
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


class MockConfig_PatchTST:
    def __init__(self, seq_len, pred_len, num_variates):
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = num_variates
        self.c_out = num_variates
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 256
        self.dropout = 0.1
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.activation = "gelu"
        self.output_attention = False


class MockConfig_TSMixer:
    def __init__(self, seq_len, pred_len, num_variates):
        self.task_name = "long_term_forecast"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = num_variates
        self.d_model = 64
        self.e_layers = 2
        self.dropout = 0.1


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
    epochs=50,
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

    seq_len = train_loader.dataset.tensors[0].shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x_enc, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x_enc, batch_y = batch_x_enc.to(device), batch_y.to(device)
            batch_x_mark_enc = base_sinusoidal.unsqueeze(0).repeat(
                batch_x_enc.shape[0], 1, 1
            )

            outputs = model(batch_x_enc, batch_x_mark_enc)
            outputs = outputs[:, -batch_y.shape[1] :, :]

            # Train on "Close"
            loss = criterion(outputs[:, :, close_idx], batch_y[:, :, close_idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_enc, batch_y in val_loader:
                batch_x_enc, batch_y = batch_x_enc.to(device), batch_y.to(device)
                batch_x_mark_enc = base_sinusoidal.unsqueeze(0).repeat(
                    batch_x_enc.shape[0], 1, 1
                )

                outputs = model(batch_x_enc, batch_x_mark_enc)
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
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    with torch.no_grad():
        X_enc = X_enc.to(device)
        X_mark_enc = base_sinusoidal.unsqueeze(0).repeat(X_enc.shape[0], 1, 1)

        preds = model(X_enc, X_mark_enc)
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
    SEEDS = [42, 2024]
    PRED_LENS = [1]

    # Contrastive pretraining configuration.
    CL_EPOCHS = 20
    CL_LR = 1e-3
    CL_PROJECTION_DIM = 64
    CL_TEMPERATURE = 0.2
    CL_JITTER_STD = 0.02
    CL_MASK_RATIO = 0.10

    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    NUM_VARIATES = df.shape[1]
    CLOSE_IDX = df.columns.get_loc("Close")
    os.makedirs("figure", exist_ok=True)

    for PRED_LEN in PRED_LENS:
        print("\n=======================================================")
        print(f" EXPERIMENT: PREDICTION LENGTH = {PRED_LEN} ")
        print("=======================================================")

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

        naive_preds, actuals, naive_mae, naive_rmse = naive_baseline(
            X_enc_test, Y_test, scaler, PRED_LEN, CLOSE_IDX
        )
        print(f"[NAIVE BASELINE] MAE: {naive_mae:.4f} | RMSE: {naive_rmse:.4f}")

        model_factories = {
            "iTransformer": lambda: iTransformerModel(
                MockConfig_iTransformer(SEQ_LEN, PRED_LEN, NUM_VARIATES)
            ),
            "PatchTST": lambda: PatchTSTModel(
                MockConfig_PatchTST(SEQ_LEN, PRED_LEN, NUM_VARIATES)
            ),
            "TSMixer": lambda: TSMixerModel(
                MockConfig_TSMixer(SEQ_LEN, PRED_LEN, NUM_VARIATES)
            ),
            "LSTM": lambda: LSTMModel(NUM_VARIATES, SEQ_LEN, PRED_LEN),
        }

        results = {
            name: {
                "baseline": {
                    "maes": [],
                    "rmses": [],
                    "best_preds": None,
                    "best_mae": float("inf"),
                },
                "cl": {
                    "maes": [],
                    "rmses": [],
                    "best_preds": None,
                    "best_mae": float("inf"),
                },
            }
            for name in model_factories
        }

        for seed in SEEDS:
            print(f"\n==================== SEED {seed} ====================")

            for model_name, factory in model_factories.items():
                print(f"\n--- {model_name}: Baseline vs Contrastive Pretraining ---")

                # Create one initialization and clone it so baseline and +CL start
                # from exactly the same weights for this seed.
                set_seed(seed)
                initial_model = factory()
                baseline_model = copy.deepcopy(initial_model)
                cl_model = copy.deepcopy(initial_model)

                # ---------------- Baseline ----------------
                set_seed(seed)
                train_model(
                    baseline_model,
                    train_loader,
                    val_loader,
                    epochs=EPOCHS,
                    lr=0.001,
                    device=device,
                    patience=5,
                    close_idx=CLOSE_IDX,
                )

                baseline_preds, _, baseline_mae, baseline_rmse = evaluate_and_predict(
                    baseline_model,
                    X_enc_test,
                    Y_test,
                    scaler,
                    PRED_LEN,
                    CLOSE_IDX,
                    device=device,
                )

                baseline_result = results[model_name]["baseline"]
                baseline_result["maes"].append(baseline_mae)
                baseline_result["rmses"].append(baseline_rmse)
                if baseline_mae < baseline_result["best_mae"]:
                    baseline_result["best_mae"] = baseline_mae
                    baseline_result["best_preds"] = baseline_preds

                print(
                    f"  Baseline      -> MAE: {baseline_mae:.4f} | "
                    f"RMSE: {baseline_rmse:.4f}"
                )

                # ---------------- CL pretrain + same forecasting fine-tune ----------------
                set_seed(seed)
                cl_model, cl_history = contrastive_pretrain(
                    cl_model,
                    train_loader,
                    epochs=CL_EPOCHS,
                    lr=CL_LR,
                    device=device,
                    projection_dim=CL_PROJECTION_DIM,
                    temperature=CL_TEMPERATURE,
                    jitter_std=CL_JITTER_STD,
                    mask_ratio=CL_MASK_RATIO,
                )

                # Projection head was local to contrastive_pretrain and is already
                # discarded. Fine-tuning below uses the original forecasting model.
                train_model(
                    cl_model,
                    train_loader,
                    val_loader,
                    epochs=EPOCHS,
                    lr=0.001,
                    device=device,
                    patience=5,
                    close_idx=CLOSE_IDX,
                )

                cl_preds, _, cl_mae, cl_rmse = evaluate_and_predict(
                    cl_model,
                    X_enc_test,
                    Y_test,
                    scaler,
                    PRED_LEN,
                    CLOSE_IDX,
                    device=device,
                )

                cl_result = results[model_name]["cl"]
                cl_result["maes"].append(cl_mae)
                cl_result["rmses"].append(cl_rmse)
                if cl_mae < cl_result["best_mae"]:
                    cl_result["best_mae"] = cl_mae
                    cl_result["best_preds"] = cl_preds

                print(
                    f"  +CL           -> MAE: {cl_mae:.4f} | RMSE: {cl_rmse:.4f} "
                    f"| final InfoNCE: {cl_history[-1]:.4f}"
                )

        print(f"\n[FINAL RESULTS PRED_LEN={PRED_LEN}]")
        print(
            f"{'Model':15s} | {'Setting':10s} | {'MAE':18s} | "
            f"{'RMSE':18s} | {'MAE Δ vs base':>13s}"
        )
        print("-" * 87)

        for model_name, model_results in results.items():
            base = model_results["baseline"]
            cl = model_results["cl"]

            base_mae = np.mean(base["maes"])
            base_mae_std = np.std(base["maes"])
            base_rmse = np.mean(base["rmses"])
            base_rmse_std = np.std(base["rmses"])

            cl_mae = np.mean(cl["maes"])
            cl_mae_std = np.std(cl["maes"])
            cl_rmse = np.mean(cl["rmses"])
            cl_rmse_std = np.std(cl["rmses"])

            mae_improvement = 100.0 * (base_mae - cl_mae) / base_mae

            print(
                f"{model_name:15s} | {'Baseline':10s} | "
                f"{base_mae:.4f} ± {base_mae_std:.4f} | "
                f"{base_rmse:.4f} ± {base_rmse_std:.4f} | {'-':>13s}"
            )
            print(
                f"{model_name:15s} | {'+CL':10s} | "
                f"{cl_mae:.4f} ± {cl_mae_std:.4f} | "
                f"{cl_rmse:.4f} ± {cl_rmse_std:.4f} | "
                f"{mae_improvement:+12.2f}%"
            )

        # ---------------------------------------------------------
        # Prediction comparison plot: solid = baseline, dashed = +CL
        # ---------------------------------------------------------
        plt.figure(figsize=(16, 8))
        colors = {
            "iTransformer": "red",
            "PatchTST": "blue",
            "LSTM": "orange",
            "TSMixer": "purple",
        }

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
                naive_preds.flatten()[-plot_range:],
                label="Naive Baseline",
                color="gray",
                linestyle=":",
                alpha=0.8,
            )

            for model_name, color in colors.items():
                plt.plot(
                    results[model_name]["baseline"]["best_preds"].flatten()[
                        -plot_range:
                    ],
                    label=model_name,
                    color=color,
                    alpha=0.70,
                )
                plt.plot(
                    results[model_name]["cl"]["best_preds"].flatten()[-plot_range:],
                    label=f"{model_name}+CL",
                    color=color,
                    linestyle="--",
                    alpha=0.95,
                )

            plt.title(
                f"{TICKER} - 1-Day Ahead: Baselines vs Contrastive Pretraining "
                f"(Last {plot_range} days)"
            )
        else:
            sample_idx = -1
            hist_unscaled = scaler.inverse_transform(X_enc_test[sample_idx].numpy())[::, CLOSE_IDX]
            time_hist = np.arange(SEQ_LEN)
            time_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

            plt.plot(
                time_hist[-30:],
                hist_unscaled[-30:],
                label="Historical Close",
                color="gray",
            )
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
                naive_preds[sample_idx],
                label="Naive Baseline",
                color="gray",
                linestyle=":",
                marker="x",
            )

            for model_name, color in colors.items():
                plt.plot(
                    time_pred,
                    results[model_name]["baseline"]["best_preds"][sample_idx],
                    label=model_name,
                    color=color,
                    alpha=0.70,
                )
                plt.plot(
                    time_pred,
                    results[model_name]["cl"]["best_preds"][sample_idx],
                    label=f"{model_name}+CL",
                    color=color,
                    linestyle="--",
                    alpha=0.95,
                )

            plt.title(
                f"{TICKER} - {PRED_LEN}-Days Horizon: Baselines vs Contrastive Pretraining"
            )

        plt.xlabel("Time Steps")
        plt.ylabel("Price (USD)")
        plt.legend(ncol=2)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        figure_path = f"figure/prediction_{PRED_LEN}day_baseline_vs_cl.png"
        plt.savefig(figure_path, dpi=300)
        plt.close()
        print(f"[+] Đã lưu biểu đồ: {figure_path}")
