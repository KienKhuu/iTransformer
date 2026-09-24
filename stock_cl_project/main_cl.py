import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from contrastive.trainer import pretrain_contrastive
from main import (
    MockConfig_PatchTST,
    MockConfig_TSMixer,
    MockConfig_iTransformer,
    evaluate_and_predict,
    set_seed,
    train_model,
)
from model.LSTM import Model as LSTMModel
from model.PatchTST import Model as PatchTSTModel
from model.TSMixer import Model as TSMixerModel
from model.iTransformer import Model as iTransformerModel
from preprocessing import fetch_stock_data, prepare_sequences


def build_models(seq_len, pred_len, num_variates):
    return {
        "iTransformer": iTransformerModel(
            MockConfig_iTransformer(seq_len, pred_len, num_variates)
        ),
        "PatchTST": PatchTSTModel(
            MockConfig_PatchTST(seq_len, pred_len, num_variates)
        ),
        "TSMixer": TSMixerModel(
            MockConfig_TSMixer(seq_len, pred_len, num_variates)
        ),
        "LSTM": LSTMModel(num_variates, seq_len, pred_len),
    }


def sanity_check_model(model, x, pred_len, num_variates, device):
    """Cheap shape/finite-value check before expensive training."""
    model = model.to(device)
    model.eval()
    x = x[: min(4, len(x))].to(device)
    seq_len = x.shape[1]

    # Keep the same 4-column sinusoidal covariate format as main.py.
    from main import generate_sinusoidal_encoding

    base_mark = generate_sinusoidal_encoding(seq_len, d_mark=4).to(device)
    x_mark = base_mark.unsqueeze(0).repeat(x.shape[0], 1, 1)

    with torch.no_grad():
        pred = model(x, x_mark)
        h = model.encode(x, x_mark)

    assert pred.shape == (x.shape[0], pred_len, num_variates), (
        f"Bad prediction shape: {pred.shape}"
    )
    assert h.ndim == 2 and h.shape[0] == x.shape[0], f"Bad encoding shape: {h.shape}"
    assert h.shape[1] == model.representation_dim, (
        h.shape,
        model.representation_dim,
    )
    assert torch.isfinite(pred).all(), "NaN/Inf in predictions"
    assert torch.isfinite(h).all(), "NaN/Inf in representations"
    return tuple(pred.shape), tuple(h.shape)


def append_result(results, name, mae, rmse):
    if name not in results:
        results[name] = {"maes": [], "rmses": []}
    results[name]["maes"].append(float(mae))
    results[name]["rmses"].append(float(rmse))


def summarize(results):
    summary = {}
    print("\n================ FINAL SUMMARY ================")
    for name, vals in results.items():
        mae_mean = float(np.mean(vals["maes"]))
        mae_std = float(np.std(vals["maes"]))
        rmse_mean = float(np.mean(vals["rmses"]))
        rmse_std = float(np.std(vals["rmses"]))
        summary[name] = {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
        }
        print(
            f"{name:20s} -> MAE {mae_mean:.4f} ± {mae_std:.4f} | "
            f"RMSE {rmse_mean:.4f} ± {rmse_std:.4f}"
        )
    return summary


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using Device: {device} ---")

    # Same dataset / forecasting setup as baseline main.py
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2026-01-01"
    SEQ_LEN = 60
    PRED_LEN = 1
    BATCH_SIZE = 32
    SEEDS = [42, 2024]

    # Supervised fine-tuning: intentionally matched to baseline
    FT_EPOCHS = 100
    FT_LR = 1e-3
    FT_PATIENCE = 5

    # Contrastive pretraining hyperparameters
    CL_EPOCHS = 40
    CL_LR = 1e-3
    CL_PATIENCE = 7
    TEMPERATURE = 0.2
    JITTER_SIGMA = 0.01
    MASK_RATIO = 0.10
    PROJECTION_HIDDEN_DIM = 128
    PROJECTION_DIM = 64

    # True gives an especially clean paired experiment: baseline and CL model
    # start from exactly the same initial weights for each seed.
    RUN_PAIRED_BASELINES = True

    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    num_variates = df.shape[1]
    close_idx = df.columns.get_loc("Close")

    X_enc, Y, scaler, train_end, val_end = prepare_sequences(
        df, SEQ_LEN, PRED_LEN, train_ratio=0.7, val_ratio=0.15
    )
    X_train, Y_train = X_enc[:train_end], Y[:train_end]
    X_val, Y_val = X_enc[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X_enc[val_end:], Y[val_end:]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    results = {}
    cl_histories = {}

    for seed in SEEDS:
        print(f"\n================ SEED {seed} ================")
        set_seed(seed)
        initialized_models = build_models(SEQ_LEN, PRED_LEN, num_variates)

        for model_name, initial_model in initialized_models.items():
            print(f"\n---------- {model_name} ----------")
            pred_shape, rep_shape = sanity_check_model(
                initial_model, X_train, PRED_LEN, num_variates, device
            )
            print(f"  sanity: pred={pred_shape}, representation={rep_shape}")

            # Make two identical copies BEFORE either training procedure.
            baseline_model = copy.deepcopy(initial_model)
            cl_model = copy.deepcopy(initial_model)

            if RUN_PAIRED_BASELINES:
                print("  [1/3] Supervised baseline")
                set_seed(seed)
                train_model(
                    baseline_model,
                    train_loader,
                    val_loader,
                    epochs=FT_EPOCHS,
                    lr=FT_LR,
                    device=device,
                    patience=FT_PATIENCE,
                    close_idx=close_idx,
                )
                _, _, mae, rmse = evaluate_and_predict(
                    baseline_model,
                    X_test,
                    Y_test,
                    scaler,
                    PRED_LEN,
                    close_idx,
                    device=device,
                )
                append_result(results, model_name, mae, rmse)
                print(f"    {model_name}: MAE={mae:.4f} RMSE={rmse:.4f}")

            print("  [2/3] Self-supervised contrastive pretraining")
            set_seed(seed)
            cl_model, history = pretrain_contrastive(
                cl_model,
                train_loader,
                val_loader,
                epochs=CL_EPOCHS,
                lr=CL_LR,
                device=device,
                patience=CL_PATIENCE,
                temperature=TEMPERATURE,
                jitter_sigma=JITTER_SIGMA,
                mask_ratio=MASK_RATIO,
                projection_hidden_dim=PROJECTION_HIDDEN_DIM,
                projection_dim=PROJECTION_DIM,
                verbose=True,
            )
            cl_histories[f"seed_{seed}_{model_name}"] = history

            print("  [3/3] Supervised fine-tuning of CL-pretrained backbone")
            # Reset RNG so supervised minibatch/dropout randomness follows the same
            # seed as the paired baseline. Only initialization differs due to CL.
            set_seed(seed)
            train_model(
                cl_model,
                train_loader,
                val_loader,
                epochs=FT_EPOCHS,
                lr=FT_LR,
                device=device,
                patience=FT_PATIENCE,
                close_idx=close_idx,
            )
            _, _, mae, rmse = evaluate_and_predict(
                cl_model,
                X_test,
                Y_test,
                scaler,
                PRED_LEN,
                close_idx,
                device=device,
            )
            cl_name = f"CL-{model_name}"
            append_result(results, cl_name, mae, rmse)
            print(f"    {cl_name}: MAE={mae:.4f} RMSE={rmse:.4f}")

    summary = summarize(results)
    output = {
        "config": {
            "ticker": TICKER,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "batch_size": BATCH_SIZE,
            "seeds": SEEDS,
            "cl_epochs": CL_EPOCHS,
            "cl_lr": CL_LR,
            "temperature": TEMPERATURE,
            "jitter_sigma": JITTER_SIGMA,
            "mask_ratio": MASK_RATIO,
        },
        "raw_results": results,
        "summary": summary,
        "cl_history": cl_histories,
    }
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "contrastive_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")
