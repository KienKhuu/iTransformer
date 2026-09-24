"""Run with `python smoke_check.py` after installing dependencies; no layers/ needed."""
import numpy as np
import pandas as pd
import torch

from contrastive import TemporalEncoder, overlapping_views, hierarchical_loss
from preprocessing import prepare_sequences


def main():
    torch.manual_seed(42)
    n, seq_len, pred_len = 200, 16, 3
    values = np.arange(n, dtype=float)
    data = pd.DataFrame({c: values + j for j, c in
                         enumerate(["Open", "High", "Low", "Close", "Volume"])})
    splits, scaler = prepare_sequences(data, seq_len, pred_len)
    train_cut, val_cut = int(n * 0.7), int(n * 0.85)
    for split, (x, y) in splits.items():
        first_target = (y[:, 0, 0].numpy() * scaler.scale_[0] + scaler.mean_[0]).round().astype(int)
        last_target = first_target + pred_len - 1
        if split == "train":
            assert (last_target < train_cut).all()
        elif split == "val":
            assert (first_target >= train_cut).all() and (last_target < val_cut).all()
        else:
            assert (first_target >= val_cut).all()
    batch = splits["train"][0][:8]
    encoder = TemporalEncoder(input_dim=5, latent_dim=16)
    a, b, ia, ib = overlapping_views(batch)
    z1, z2 = encoder(a)[:, ia], encoder(b)[:, ib]
    assert z1.shape == z2.shape and z1.shape[0] == 8
    loss = hierarchical_loss(z1, z2)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in encoder.parameters())
    # Output at an earlier time may never depend on observations after it.
    encoder.eval()
    changed = batch.clone()
    changed[:, 8:] += 100
    torch.testing.assert_close(encoder(batch)[:, :8], encoder(changed)[:, :8])
    print("Split boundaries, contrastive backward pass, and causal encoding: OK")


if __name__ == "__main__":
    main()
