import copy
import numpy as np
import torch
import torch.optim as optim

from .augmentations import TimeSeriesAugmenter
from .losses import ProjectionHead, nt_xent_loss


def _generate_sinusoidal_encoding(seq_len, d_mark, device):
    pe = torch.zeros(seq_len, d_mark, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_mark, 2, dtype=torch.float, device=device)
        * (-np.log(10000.0) / d_mark)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _make_time_marks(batch_size, seq_len, device):
    base = _generate_sinusoidal_encoding(seq_len, d_mark=4, device=device)
    return base.unsqueeze(0).repeat(batch_size, 1, 1)


def _infer_representation_dim(model, train_loader, device):
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(train_loader))
        x = first_batch[0].to(device)
        x_mark = _make_time_marks(x.size(0), x.size(1), device)
        h = model.encode(x, x_mark)
    if h.ndim != 2:
        raise ValueError(
            f"{model.__class__.__name__}.encode() must return [B, D], got {tuple(h.shape)}"
        )
    return h.size(1)


def contrastive_pretrain(
    model,
    train_loader,
    epochs=20,
    lr=1e-3,
    device="cpu",
    projection_dim=64,
    temperature=0.2,
    jitter_std=0.02,
    mask_ratio=0.10,
):
    """
    Self-supervised contrastive pretraining of an existing forecasting backbone.

    The projection head exists only during pretraining and is discarded afterwards.
    The returned model keeps the same forecasting forward()/forecast() behavior.
    """
    if not hasattr(model, "encode"):
        raise AttributeError(
            f"{model.__class__.__name__} must implement encode(x, x_mark_enc) for CL."
        )

    model = model.to(device)
    representation_dim = _infer_representation_dim(model, train_loader, device)
    projector = ProjectionHead(representation_dim, projection_dim).to(device)
    augmenter = TimeSeriesAugmenter(jitter_std=jitter_std, mask_ratio=mask_ratio)

    # Forecast heads that are not touched by encode() simply receive no gradients.
    optimizer = optim.Adam(
        list(model.parameters()) + list(projector.parameters()), lr=lr
    )

    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(epochs):
        model.train()
        projector.train()
        running_loss = 0.0
        valid_batches = 0

        for batch in train_loader:
            x = batch[0].to(device)
            if x.size(0) < 2:
                # InfoNCE needs negatives; skip a singleton final batch.
                continue

            x1, x2 = augmenter.make_views(x)
            x_mark = _make_time_marks(x.size(0), x.size(1), device)

            optimizer.zero_grad()
            h1 = model.encode(x1, x_mark)
            h2 = model.encode(x2, x_mark)
            z1 = projector(h1)
            z2 = projector(h2)

            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            valid_batches += 1

        if valid_batches == 0:
            raise RuntimeError(
                "No valid contrastive batches. Use batch_size >= 2 and enough training data."
            )

        epoch_loss = running_loss / valid_batches
        history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"    CL epoch {epoch + 1:02d}/{epochs:02d} | "
            f"InfoNCE: {epoch_loss:.4f}"
        )

    model.load_state_dict(best_model_wts)
    return model, history
