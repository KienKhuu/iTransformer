import copy
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from model.iTransformer import Model as iTransformerModel
from model.PatchTST import Model as PatchTSTModel
from model.LSTM import Model as LSTMModel
from model.TSMixer import Model as TSMixerModel
from preprocessing import fetch_stock_data, prepare_sequences
from contrastive import ContrastiveWrapper, make_contrastive_views, info_nce_loss


# =========================================================
# Reproducibility & shared helpers
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


# =========================================================
# Model configs - original model source files stay unchanged
# =========================================================
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


def build_models(seq_len, pred_len, num_variates):
    """Build all original forecasting backbones in one deterministic order."""
    return {
        "iTransformer": iTransformerModel(
            MockConfig_iTransformer(seq_len, pred_len, num_variates)
        ),
        "PatchTST": PatchTSTModel(MockConfig_PatchTST(seq_len, pred_len, num_variates)),
        "TSMixer": TSMixerModel(MockConfig_TSMixer(seq_len, pred_len, num_variates)),
        "LSTM": LSTMModel(num_variates, seq_len, pred_len),
    }


# =========================================================
# Baseline
# =========================================================
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
    """Vanilla forecasting training."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    seq_len = train_loader.dataset.tensors[0].shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_mark = base_sinusoidal.unsqueeze(0).repeat(batch_x.size(0), 1, 1)

            optimizer.zero_grad()
            outputs = model(batch_x, x_mark)
            outputs = outputs[:, -batch_y.shape[1] :, :]

            loss = criterion(outputs[:, :, close_idx], batch_y[:, :, close_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                x_mark = base_sinusoidal.unsqueeze(0).repeat(batch_x.size(0), 1, 1)

                outputs = model(batch_x, x_mark)
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
                break

    # Always evaluate the best validation checkpoint.
    model.load_state_dict(best_model_wts)


# =========================================================
# Contrastive Learning
# =========================================================
def train_model_cl(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    device="cpu",
    patience=5,
    close_idx=3,
    lambda_cl=0.1,
    temperature=0.2,
    jitter_sigma=0.01,
    mask_ratio=0.10,
):
    """Joint forecasting + contrastive training."""
    criterion = nn.MSELoss()
    model.to(device)

    seq_len = train_loader.dataset.tensors[0].shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    # Materialize LazyLinear before optimizer construction.
    init_x = train_loader.dataset.tensors[0][:2].to(device)
    init_mark = base_sinusoidal.unsqueeze(0).repeat(init_x.size(0), 1, 1)
    model.initialize_projection(init_x, init_mark)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epochs):
        model.train()
        sum_total = 0.0
        sum_forecast = 0.0
        sum_cl = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_mark = base_sinusoidal.unsqueeze(0).repeat(batch_x.size(0), 1, 1)

            optimizer.zero_grad()

            outputs = model(batch_x, x_mark)
            outputs = outputs[:, -batch_y.shape[1] :, :]
            forecast_loss = criterion(
                outputs[:, :, close_idx], batch_y[:, :, close_idx]
            )

            if batch_x.size(0) >= 2:
                view1, view2 = make_contrastive_views(
                    batch_x,
                    jitter_sigma=jitter_sigma,
                    mask_ratio=mask_ratio,
                )
                z1 = model.encode(view1, x_mark)
                z2 = model.encode(view2, x_mark)
                cl_loss = info_nce_loss(z1, z2, temperature=temperature)
            else:
                cl_loss = forecast_loss.new_zeros(())

            loss = forecast_loss + lambda_cl * cl_loss
            loss.backward()
            optimizer.step()

            sum_total += loss.item()
            sum_forecast += forecast_loss.item()
            sum_cl += cl_loss.item()
            n_batches += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                x_mark = base_sinusoidal.unsqueeze(0).repeat(batch_x.size(0), 1, 1)
                outputs = model(batch_x, x_mark)
                outputs = outputs[:, -batch_y.shape[1] :, :]
                val_loss += criterion(
                    outputs[:, :, close_idx], batch_y[:, :, close_idx]
                ).item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            denom = max(n_batches, 1)
            print(
                f"      epoch={epoch+1:03d} "
                f"total={sum_total/denom:.5f} "
                f"forecast={sum_forecast/denom:.5f} "
                f"cl={sum_cl/denom:.5f} "
                f"val={val_loss:.5f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_model_wts)


# =========================================================
# Shared evaluation
# =========================================================
def evaluate_and_predict(
    model, X_enc, Y_true, scaler, pred_len, close_idx=3, device="cpu"
):
    model.to(device)
    model.eval()

    seq_len = X_enc.shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    preds_all = []
    eval_loader = DataLoader(
        TensorDataset(X_enc, Y_true), batch_size=256, shuffle=False
    )

    with torch.no_grad():
        for batch_x, _ in eval_loader:
            batch_x = batch_x.to(device)
            x_mark = base_sinusoidal.unsqueeze(0).repeat(batch_x.size(0), 1, 1)
            preds = model(batch_x, x_mark)
            preds_all.append(preds[:, -pred_len:, :].cpu())

    preds = torch.cat(preds_all, dim=0)
    B, _, N = preds.shape

    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, N).numpy()).reshape(
        B, pred_len, N
    )
    y_unscaled = scaler.inverse_transform(Y_true.reshape(-1, N).numpy()).reshape(
        B, pred_len, N
    )

    preds_close = preds_unscaled[:, :, close_idx]
    actual_close = y_unscaled[:, :, close_idx]

    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))
    return preds_close, actual_close, mae, rmse


# =========================================================
# Results helpers
# =========================================================
def empty_result_store(model_names):
    return {
        model_name: {
            "baseline": {"maes": [], "rmses": [], "last_preds": None},
            "cl": {"maes": [], "rmses": [], "last_preds": None},
        }
        for model_name in model_names
    }


def metric_summary(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def improvement_percent(base, improved):
    if base == 0:
        return np.nan
    return 100.0 * (base - improved) / base


def print_final_results(results, pred_len):
    print(f"\n[FINAL COMPARISON PRED_LEN={pred_len}]")
    print("-" * 110)

    rows = []
    for model_name, model_results in results.items():
        base_mae, base_mae_std = metric_summary(model_results["baseline"]["maes"])
        base_rmse, base_rmse_std = metric_summary(model_results["baseline"]["rmses"])
        cl_mae, cl_mae_std = metric_summary(model_results["cl"]["maes"])
        cl_rmse, cl_rmse_std = metric_summary(model_results["cl"]["rmses"])

        delta_mae = improvement_percent(base_mae, cl_mae)
        delta_rmse = improvement_percent(base_rmse, cl_rmse)

        print(
            f"{model_name:15s} | "
            f"BASE MAE {base_mae:.4f} ± {base_mae_std:.4f} | "
            f"CL MAE {cl_mae:.4f} ± {cl_mae_std:.4f} | "
            f"ΔMAE {delta_mae:+.2f}%"
        )
        print(
            f"{'':15s} | "
            f"BASE RMSE {base_rmse:.4f} ± {base_rmse_std:.4f} | "
            f"CL RMSE {cl_rmse:.4f} ± {cl_rmse_std:.4f} | "
            f"ΔRMSE {delta_rmse:+.2f}%"
        )

        rows.append(
            {
                "pred_len": pred_len,
                "model": model_name,
                "base_mae": base_mae,
                "base_mae_std": base_mae_std,
                "cl_mae": cl_mae,
                "cl_mae_std": cl_mae_std,
                "delta_mae_percent": delta_mae,
                "base_rmse": base_rmse,
                "base_rmse_std": base_rmse_std,
                "cl_rmse": cl_rmse,
                "cl_rmse_std": cl_rmse_std,
                "delta_rmse_percent": delta_rmse,
            }
        )

    return rows


def save_results_csv(rows, filepath):
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_prediction_comparison(
    results,
    X_test,
    actuals,
    naive_preds,
    scaler,
    pred_len,
    seq_len,
    close_idx,
    ticker,
    output_dir="figure",
):
    """Save separate baseline and CL plots to avoid one overcrowded figure."""
    os.makedirs(output_dir, exist_ok=True)

    styles = {
        "iTransformer": "red",
        "PatchTST": "blue",
        "LSTM": "orange",
        "TSMixer": "purple",
    }

    for variant, title_suffix in [("baseline", "Baseline"), ("cl", "+CL")]:
        plt.figure(figsize=(14, 7))

        if pred_len == 1:
            plot_range = min(100, len(actuals.flatten()))
            plt.plot(
                actuals.flatten()[-plot_range:],
                label="Actual Close",
                color="black",
                linewidth=2,
            )

            if variant == "baseline":
                plt.plot(
                    naive_preds.flatten()[-plot_range:],
                    label="Naive",
                    color="gray",
                    linestyle="--",
                    alpha=0.7,
                )

            for model_name, color in styles.items():
                preds = results[model_name][variant]["last_preds"]
                if preds is not None:
                    plt.plot(
                        preds.flatten()[-plot_range:],
                        label=f"{model_name}{' + CL' if variant == 'cl' else ''}",
                        color=color,
                        alpha=0.8,
                    )

            plt.title(
                f"{ticker} - 1-Day Ahead Prediction ({title_suffix}, Last {plot_range} Days)"
            )
        else:
            sample_idx = -1
            hist_unscaled = scaler.inverse_transform(X_test[sample_idx].numpy())[
                :, close_idx
            ]
            time_hist = np.arange(seq_len)
            time_pred = np.arange(seq_len, seq_len + pred_len)

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

            if variant == "baseline":
                plt.plot(
                    time_pred,
                    naive_preds[sample_idx],
                    label="Naive",
                    color="gray",
                    linestyle="--",
                    marker="x",
                )

            for model_name, color in styles.items():
                preds = results[model_name][variant]["last_preds"]
                if preds is not None:
                    plt.plot(
                        time_pred,
                        preds[sample_idx],
                        label=f"{model_name}{' + CL' if variant == 'cl' else ''}",
                        color=color,
                        marker="o",
                    )

            plt.title(f"{ticker} - {pred_len}-Day Prediction ({title_suffix})")

        plt.xlabel("Time Steps")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        path = os.path.join(output_dir, f"prediction_{pred_len}day_{variant}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[+] Saved: {path}")


# =========================================================
# Main experiment
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using Device: {device} ---")

    # -------------------------
    # Dataset / training config
    # -------------------------
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2026-01-01"
    SEQ_LEN = 60
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 5
    LEARNING_RATE = 0.001
    SEEDS = [42, 2024]
    PRED_LENS = [1]

    # -------------------------
    # CL config - one place only
    # -------------------------
    CL_CONFIG = {
        "lambda_cl": 0.10,
        "temperature": 0.20,
        "jitter_sigma": 0.01,
        "mask_ratio": 0.10,
        "projection_dim": 64,
        "projection_hidden": 128,
    }

    # Optional manual overrides. Leave None to auto-detect.
    REPRESENTATION_LAYERS = {
        "iTransformer": None,
        "PatchTST": None,
        "TSMixer": None,
        "LSTM": None,
    }

    # Both are True by default so deltas are computed from the SAME run/protocol.
    RUN_BASELINE = True
    RUN_CL = True

    os.makedirs("figure", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    NUM_VARIATES = df.shape[1]
    CLOSE_IDX = df.columns.get_loc("Close")
    MODEL_NAMES = ["iTransformer", "PatchTST", "TSMixer", "LSTM"]

    for PRED_LEN in PRED_LENS:
        print("\n" + "=" * 72)
        print(f"EXPERIMENT: PREDICTION LENGTH = {PRED_LEN}")
        print("=" * 72)

        X_enc, Y, scaler, train_end, val_end = prepare_sequences(
            df,
            SEQ_LEN,
            PRED_LEN,
            train_ratio=0.7,
            val_ratio=0.15,
        )

        X_train, Y_train = X_enc[:train_end], Y[:train_end]
        X_val, Y_val = X_enc[train_end:val_end], Y[train_end:val_end]
        X_test, Y_test = X_enc[val_end:], Y[val_end:]

        # One shared loader protocol for baseline and CL.
        train_loader = DataLoader(
            TensorDataset(X_train, Y_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        naive_preds, actuals, naive_mae, naive_rmse = naive_baseline(
            X_test, Y_test, scaler, PRED_LEN, CLOSE_IDX
        )
        print(f"[NAIVE] MAE: {naive_mae:.4f} | RMSE: {naive_rmse:.4f}")

        results = empty_result_store(MODEL_NAMES)

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            # Build paired backbone sets from the same RNG state.
            if RUN_BASELINE:
                set_seed(seed)
                baseline_models = build_models(SEQ_LEN, PRED_LEN, NUM_VARIATES)
            else:
                baseline_models = {}

            if RUN_CL:
                set_seed(seed)
                cl_backbones = build_models(SEQ_LEN, PRED_LEN, NUM_VARIATES)
            else:
                cl_backbones = {}

            for model_name in MODEL_NAMES:
                # -------------------------
                # Vanilla baseline
                # -------------------------
                if RUN_BASELINE:
                    print(f"\n[{model_name} BASELINE]")
                    set_seed(seed)
                    model = baseline_models.pop(model_name)

                    train_model(
                        model,
                        train_loader,
                        val_loader,
                        epochs=EPOCHS,
                        lr=LEARNING_RATE,
                        device=device,
                        patience=PATIENCE,
                        close_idx=CLOSE_IDX,
                    )

                    preds, _, mae, rmse = evaluate_and_predict(
                        model,
                        X_test,
                        Y_test,
                        scaler,
                        PRED_LEN,
                        CLOSE_IDX,
                        device=device,
                    )

                    results[model_name]["baseline"]["maes"].append(mae)
                    results[model_name]["baseline"]["rmses"].append(rmse)
                    results[model_name]["baseline"]["last_preds"] = preds
                    print(f"  MAE={mae:.4f} | RMSE={rmse:.4f}")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # -------------------------
                # Same backbone + CL
                # -------------------------
                if RUN_CL:
                    print(f"\n[{model_name} + CL]")
                    set_seed(seed)
                    backbone = cl_backbones.pop(model_name)
                    wrapped = ContrastiveWrapper(
                        backbone=backbone,
                        model_name=model_name,
                        projection_dim=CL_CONFIG["projection_dim"],
                        projection_hidden=CL_CONFIG["projection_hidden"],
                        representation_layer=REPRESENTATION_LAYERS[model_name],
                    ).to(device)

                    print(f"  representation hook: {wrapped.representation_layer}")

                    train_model_cl(
                        wrapped,
                        train_loader,
                        val_loader,
                        epochs=EPOCHS,
                        lr=LEARNING_RATE,
                        device=device,
                        patience=PATIENCE,
                        close_idx=CLOSE_IDX,
                        lambda_cl=CL_CONFIG["lambda_cl"],
                        temperature=CL_CONFIG["temperature"],
                        jitter_sigma=CL_CONFIG["jitter_sigma"],
                        mask_ratio=CL_CONFIG["mask_ratio"],
                    )

                    preds, _, mae, rmse = evaluate_and_predict(
                        wrapped,
                        X_test,
                        Y_test,
                        scaler,
                        PRED_LEN,
                        CLOSE_IDX,
                        device=device,
                    )

                    results[model_name]["cl"]["maes"].append(mae)
                    results[model_name]["cl"]["rmses"].append(rmse)
                    results[model_name]["cl"]["last_preds"] = preds
                    print(f"  MAE={mae:.4f} | RMSE={rmse:.4f}")

                    wrapped.close()
                    del wrapped
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Dynamic comparison: no BASELINE_RESULTS hard-code.
        if RUN_BASELINE and RUN_CL:
            rows = print_final_results(results, PRED_LEN)
            csv_path = f"results/comparison_pred_len_{PRED_LEN}.csv"
            save_results_csv(rows, csv_path)
            print(f"[+] Saved: {csv_path}")

            plot_prediction_comparison(
                results=results,
                X_test=X_test,
                actuals=actuals,
                naive_preds=naive_preds,
                scaler=scaler,
                pred_len=PRED_LEN,
                seq_len=SEQ_LEN,
                close_idx=CLOSE_IDX,
                ticker=TICKER,
            )
        else:
            print(
                "\nRun both RUN_BASELINE=True and RUN_CL=True to compute "
                "automatic improvement percentages."
            )
