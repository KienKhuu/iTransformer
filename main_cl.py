import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.iTransformer import Model as iTransformerModel
from model.PatchTST import Model as PatchTSTModel
from model.LSTM import Model as LSTMModel
from model.TSMixer import Model as TSMixerModel
from preprocessing import fetch_stock_data, prepare_sequences

# Reuse the exact baseline configs/helpers without editing main.py.
from main import (
    set_seed,
    generate_sinusoidal_encoding,
    MockConfig_iTransformer,
    MockConfig_PatchTST,
    MockConfig_TSMixer,
)
from contrastive import (
    ContrastiveWrapper,
    make_contrastive_views,
    info_nce_loss,
)


BASELINE_RESULTS = {
    "iTransformer": {"mae": 3.6855, "rmse": 5.0306},
    "PatchTST": {"mae": 3.1563, "rmse": 4.3838},
    "LSTM": {"mae": 23.6285, "rmse": 27.7660},
    "TSMixer": {"mae": 4.7892, "rmse": 6.3808},
}


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
    criterion = nn.MSELoss()
    model.to(device)

    seq_len = train_loader.dataset.tensors[0].shape[1]
    base_sinusoidal = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)

    # Materialize LazyLinear using one real batch BEFORE constructing optimizer.
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
            # Last batch with one sample cannot form in-batch negatives.
            if batch_x.size(0) < 2:
                continue

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_mark = base_sinusoidal.unsqueeze(0).repeat(batch_x.size(0), 1, 1)

            optimizer.zero_grad()

            # 1) Forecasting branch: ORIGINAL input, unchanged backbone.
            outputs = model(batch_x, x_mark)
            outputs = outputs[:, -batch_y.shape[1] :, :]
            forecast_loss = criterion(
                outputs[:, :, close_idx], batch_y[:, :, close_idx]
            )

            # 2) Contrastive branch: same backbone, two perturbed views.
            view1, view2 = make_contrastive_views(
                batch_x,
                jitter_sigma=jitter_sigma,
                mask_ratio=mask_ratio,
            )
            z1 = model.encode(view1, x_mark)
            z2 = model.encode(view2, x_mark)
            cl_loss = info_nce_loss(z1, z2, temperature=temperature)

            # 3) Joint objective.
            loss = forecast_loss + lambda_cl * cl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sum_total += loss.item()
            sum_forecast += forecast_loss.item()
            sum_cl += cl_loss.item()
            n_batches += 1

        # Early stopping remains based ONLY on forecasting validation loss.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                x_mark = base_sinusoidal.unsqueeze(0).repeat(
                    batch_x.size(0), 1, 1
                )
                outputs = model(batch_x, x_mark)
                outputs = outputs[:, -batch_y.shape[1] :, :]
                val_loss += criterion(
                    outputs[:, :, close_idx], batch_y[:, :, close_idx]
                ).item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            denom = max(n_batches, 1)
            print(
                f"    epoch={epoch+1:03d} "
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


def evaluate_cl_model(
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

    preds_unscaled = scaler.inverse_transform(
        preds.reshape(-1, N).numpy()
    ).reshape(B, pred_len, N)
    y_unscaled = scaler.inverse_transform(
        Y_true.reshape(-1, N).numpy()
    ).reshape(B, pred_len, N)

    preds_close = preds_unscaled[:, :, close_idx]
    actual_close = y_unscaled[:, :, close_idx]

    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))
    return mae, rmse


def build_models(seq_len, pred_len, num_variates):
    # These are the ORIGINAL four model classes. No source file is modified.
    return {
        "iTransformer": iTransformerModel(
            MockConfig_iTransformer(seq_len, pred_len, num_variates)
        ),
        "PatchTST": PatchTSTModel(
            MockConfig_PatchTST(seq_len, pred_len, num_variates)
        ),
        "LSTM": LSTMModel(num_variates, seq_len, pred_len),
        "TSMixer": TSMixerModel(
            MockConfig_TSMixer(seq_len, pred_len, num_variates)
        ),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using Device: {device} ---")

    # Keep protocol aligned with your current baseline run.
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2026-01-01"
    SEQ_LEN = 60
    BATCH_SIZE = 32
    EPOCHS = 100
    SEEDS = [42, 2024]
    PRED_LENS = [1]

    # First-pass CL hyperparameters. Do not tune until the pipeline is verified.
    LAMBDA_CL = 0.10
    TEMPERATURE = 0.20
    JITTER_SIGMA = 0.01
    MASK_RATIO = 0.10

    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    num_variates = df.shape[1]
    close_idx = df.columns.get_loc("Close")

    for pred_len in PRED_LENS:
        X_enc, Y, scaler, train_end, val_end = prepare_sequences(
            df, SEQ_LEN, pred_len, train_ratio=0.7, val_ratio=0.15
        )

        X_train, Y_train = X_enc[:train_end], Y[:train_end]
        X_val, Y_val = X_enc[train_end:val_end], Y[train_end:val_end]
        X_test, Y_test = X_enc[val_end:], Y[val_end:]

        train_loader = DataLoader(
            TensorDataset(X_train, Y_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        results = {
            name: {"maes": [], "rmses": []}
            for name in ["iTransformer", "PatchTST", "LSTM", "TSMixer"]
        }

        for seed in SEEDS:
            print(f"\n=== Seed {seed} ===")
            set_seed(seed)
            base_models = build_models(SEQ_LEN, pred_len, num_variates)

            for model_name, backbone in base_models.items():
                print(f"\n[{model_name} + CL]")

                wrapped = ContrastiveWrapper(
                    backbone=backbone,
                    model_name=model_name,
                    projection_dim=64,
                    projection_hidden=128,
                ).to(device)

                print(
                    f"  representation hook: {wrapped.representation_layer}"
                )

                train_model_cl(
                    wrapped,
                    train_loader,
                    val_loader,
                    epochs=EPOCHS,
                    lr=0.001,
                    device=device,
                    patience=5,
                    close_idx=close_idx,
                    lambda_cl=LAMBDA_CL,
                    temperature=TEMPERATURE,
                    jitter_sigma=JITTER_SIGMA,
                    mask_ratio=MASK_RATIO,
                )

                mae, rmse = evaluate_cl_model(
                    wrapped,
                    X_test,
                    Y_test,
                    scaler,
                    pred_len,
                    close_idx,
                    device,
                )
                results[model_name]["maes"].append(mae)
                results[model_name]["rmses"].append(rmse)
                print(f"  MAE={mae:.4f} | RMSE={rmse:.4f}")

                wrapped.close()
                del wrapped
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"\n[FINAL CL RESULTS PRED_LEN={pred_len}]")
        for model_name, values in results.items():
            mae_mean = np.mean(values["maes"])
            mae_std = np.std(values["maes"])
            rmse_mean = np.mean(values["rmses"])
            rmse_std = np.std(values["rmses"])

            base = BASELINE_RESULTS.get(model_name)
            if base is not None:
                delta_mae = 100.0 * (base["mae"] - mae_mean) / base["mae"]
                delta_rmse = 100.0 * (base["rmse"] - rmse_mean) / base["rmse"]
                delta_text = (
                    f" | ΔMAE={delta_mae:+.2f}% | ΔRMSE={delta_rmse:+.2f}%"
                )
            else:
                delta_text = ""

            print(
                f"{model_name:15s} -> "
                f"MAE: {mae_mean:.4f} ± {mae_std:.4f} | "
                f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}"
                f"{delta_text}"
            )
