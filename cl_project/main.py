"""Run matched raw / random frozen / contrastive frozen forecasting experiments.

Requires the original project's layers/ directory beside this file.
"""
import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from contrastive import TemporalEncoder, pretrain_encoder
from preprocessing import fetch_stock_data, load_csv, prepare_sequences
from model.iTransformer import Model as ITransformer
from model.PatchTST import Model as PatchTST
from model.TSMixer import Model as TSMixer
from model.LSTM import Model as LSTM


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IConfig:
    def __init__(self, seq_len, pred_len, channels):
        self.seq_len, self.pred_len = seq_len, pred_len
        self.d_model, self.n_heads, self.e_layers, self.d_ff = 64, 4, 2, 256
        self.factor, self.dropout = 1, 0.1
        self.embed, self.freq, self.activation = "timeF", "d", "gelu"
        self.output_attention, self.use_norm = False, True
        self.class_strategy = "projection"


class PConfig:
    def __init__(self, seq_len, pred_len, channels):
        self.seq_len, self.pred_len = seq_len, pred_len
        self.label_len, self.enc_in, self.c_out = 0, channels, channels
        self.patch_len, self.stride, self.padding_patch = 16, 8, "end"
        self.revin, self.affine, self.subtract_last = 1, 0, 0
        self.decomposition, self.kernel_size, self.individual = 0, 25, 0
        self.d_model, self.n_heads, self.e_layers, self.d_ff = 64, 4, 2, 256
        self.dropout = self.fc_dropout = self.head_dropout = 0.1
        self.activation, self.output_attention = "gelu", False


class MConfig:
    def __init__(self, seq_len, pred_len, channels):
        self.task_name, self.seq_len, self.pred_len = "long_term_forecast", seq_len, pred_len
        self.enc_in, self.d_model, self.e_layers, self.dropout = channels, 64, 2, 0.1


def build_backbone(name, seq_len, pred_len, channels):
    if name == "iTransformer":
        return ITransformer(IConfig(seq_len, pred_len, channels))
    if name == "PatchTST":
        return PatchTST(PConfig(seq_len, pred_len, channels))
    if name == "TSMixer":
        return TSMixer(MConfig(seq_len, pred_len, channels))
    if name == "LSTM":
        return LSTM(channels, seq_len, pred_len)
    raise ValueError(name)


class ForecastingSystem(nn.Module):
    """All branches forecast a change from the last observed scaled Close."""
    def __init__(self, backbone, close_idx, encoder=None, latent_dim=None):
        super().__init__()
        self.backbone, self.encoder, self.close_idx = backbone, encoder, close_idx
        if encoder is not None:
            encoder.requires_grad_(False)
            encoder.eval()
        self.readout = nn.Linear(latent_dim, 1) if encoder is not None else None

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def forward(self, raw_x):
        # Frozen encoder never uses targets, and is kept in eval mode.
        if self.encoder is None:
            features = raw_x
        else:
            with torch.no_grad():
                features = self.encoder(raw_x)
        positions = torch.arange(features.size(1), device=features.device, dtype=features.dtype)
        frequencies = torch.exp(torch.arange(0, 4, 2, device=features.device,
                                               dtype=features.dtype) * (-np.log(10000.0) / 4))
        marks = torch.zeros(features.size(1), 4, device=features.device, dtype=features.dtype)
        marks[:, 0::2] = torch.sin(positions[:, None] * frequencies)
        marks[:, 1::2] = torch.cos(positions[:, None] * frequencies)
        output = self.backbone(features, marks.unsqueeze(0).expand(len(raw_x), -1, -1))
        if isinstance(output, tuple):
            output = output[0]
        residual = (self.readout(output).squeeze(-1) if self.readout is not None
                    else output[:, :, self.close_idx])
        return raw_x[:, -1, self.close_idx].unsqueeze(1) + residual


def train_forecaster(model, train_data, val_data, device, batch_size=32,
                     epochs=100, patience=10, lr=1e-3, close_idx=3):
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    loaders = [DataLoader(TensorDataset(*split), batch_size=batch_size, shuffle=shuffle)
               for split, shuffle in ((train_data, True), (val_data, False))]
    best_loss, best_state, stalled = float("inf"), None, 0
    for epoch in range(epochs):
        losses = []
        for phase, loader in enumerate(loaders):
            model.train(phase == 0)
            running, count = 0.0, 0
            context = torch.enable_grad() if phase == 0 else torch.no_grad()
            with context:
                for x, y in loader:
                    x, target = x.to(device), y[:, :, close_idx].to(device)
                    pred = model(x)
                    loss = nn.functional.mse_loss(pred, target)
                    if phase == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running += loss.item() * len(x)
                    count += len(x)
            losses.append(running / count)
        if losses[1] < best_loss:
            best_loss, best_state, stalled = losses[1], copy.deepcopy(model.state_dict()), 0
        else:
            stalled += 1
        if stalled >= patience:
            break
    model.load_state_dict(best_state)  # also restore when all epochs complete
    return best_loss


def evaluate(model, split, scaler, close_idx, device, batch_size=256):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(*split), batch_size=batch_size):
            preds.append(model(x.to(device)).cpu().numpy())
            actuals.append(y[:, :, close_idx].numpy())
    scale = scaler.scale_[close_idx]
    mean = scaler.mean_[close_idx]
    pred = np.concatenate(preds) * scale + mean
    actual = np.concatenate(actuals) * scale + mean
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Date-indexed OHLCV CSV; auto-adjusted prices recommended")
    source.add_argument("--ticker", help="Yahoo Finance symbol, e.g. AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cl-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024])
    parser.add_argument("--models", nargs="+", default=["iTransformer", "PatchTST", "TSMixer", "LSTM"],
                        choices=["iTransformer", "PatchTST", "TSMixer", "LSTM"])
    parser.add_argument("--output", type=Path, default=Path("results_cl.csv"))
    args = parser.parse_args()
    if args.seq_len < 16 or args.pred_len < 1 or args.latent_dim < 2:
        parser.error("seq-len must be >=16, pred-len >=1, and latent-dim >=2")
    data = load_csv(args.csv) if args.csv else fetch_stock_data(args.ticker, args.start, args.end)
    splits, scaler = prepare_sequences(data, args.seq_len, args.pred_len)
    channels, close_idx = data.shape[1], data.columns.get_loc("Close")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; windows: " + ", ".join(f"{k}={len(v[0])}" for k, v in splits.items()))
    test_x, test_y = splits["test"]
    scale, mean = scaler.scale_[close_idx], scaler.mean_[close_idx]
    persistence = np.repeat(test_x[:, -1, close_idx].numpy()[:, None], args.pred_len, axis=1) * scale + mean
    actual = test_y[:, :, close_idx].numpy() * scale + mean
    rows = [{"seed": "all", "representation": "persistence", "model": "last Close",
             "mae": float(np.mean(np.abs(persistence - actual))),
             "rmse": float(np.sqrt(np.mean((persistence - actual) ** 2)))}]
    print("Persistence:", rows[0])
    for seed in args.seeds:
        seed_everything(seed)
        initial_encoder = TemporalEncoder(channels, args.latent_dim)
        random_encoder = copy.deepcopy(initial_encoder)
        cl_encoder = pretrain_encoder(splits["train"][0], channels, seed, device,
                                      latent_dim=args.latent_dim, epochs=args.cl_epochs,
                                      batch_size=args.batch_size, encoder=initial_encoder)
        for name in args.models:
            for variant in ("raw", "random_frozen", "cl_frozen"):
                # Reset downstream initialization and batch order across variants.
                seed_everything(seed + 1000 * args.models.index(name))
                encoder = {"raw": None, "random_frozen": random_encoder,
                           "cl_frozen": cl_encoder}[variant]
                encoder = copy.deepcopy(encoder) if encoder is not None else None
                dim = channels if encoder is None else args.latent_dim
                backbone = build_backbone(name, args.seq_len, args.pred_len, dim)
                system = ForecastingSystem(backbone, close_idx, encoder, args.latent_dim)
                best_val = train_forecaster(system, splits["train"], splits["val"], device,
                                            batch_size=args.batch_size, epochs=args.epochs,
                                            close_idx=close_idx)
                mae, rmse = evaluate(system, splits["test"], scaler, close_idx, device)
                rows.append({"seed": seed, "representation": variant, "model": name,
                             "mae": mae, "rmse": rmse})
                print(f"{name:14} {variant:14} seed={seed} val_mse={best_val:.5f} "
                      f"MAE={mae:.4f} RMSE={rmse:.4f}")
                del system
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "representation", "model", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
