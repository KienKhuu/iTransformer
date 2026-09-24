import copy
import numpy as np
import torch
import torch.optim as optim

from .augmentations import make_contrastive_views
from .loss import nt_xent_loss
from .wrapper import ContrastiveWrapper


def _sinusoidal_encoding(seq_len, d_mark=4, device="cpu"):
    pe = torch.zeros(seq_len, d_mark, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_mark, 2, dtype=torch.float, device=device)
        * (-np.log(10000.0) / d_mark)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _run_epoch(
    wrapper,
    loader,
    optimizer,
    device,
    temperature,
    jitter_sigma,
    mask_ratio,
    train=True,
):
    wrapper.train(train)
    total_loss = 0.0
    used_batches = 0
    seq_len = loader.dataset.tensors[0].shape[1]
    base_mark = _sinusoidal_encoding(seq_len, d_mark=4, device=device)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_x, _ in loader:
            if batch_x.shape[0] < 2:
                continue
            batch_x = batch_x.to(device)
            x_mark = base_mark.unsqueeze(0).repeat(batch_x.shape[0], 1, 1)
            view1, view2 = make_contrastive_views(
                batch_x, jitter_sigma=jitter_sigma, mask_ratio=mask_ratio
            )

            if train:
                optimizer.zero_grad(set_to_none=True)

            z1 = wrapper(view1, x_mark)
            z2 = wrapper(view2, x_mark)

            if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
                raise FloatingPointError("NaN/Inf detected in contrastive projections.")

            loss = nt_xent_loss(z1, z2, temperature=temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError("NaN/Inf detected in contrastive loss.")

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            used_batches += 1

    if used_batches == 0:
        raise RuntimeError("No valid CL batch (need batch size >= 2).")
    return total_loss / used_batches


def pretrain_contrastive(
    model,
    train_loader,
    val_loader,
    epochs=40,
    lr=1e-3,
    device="cpu",
    patience=7,
    temperature=0.2,
    jitter_sigma=0.01,
    mask_ratio=0.10,
    projection_hidden_dim=128,
    projection_dim=64,
    verbose=True,
):
    """Self-supervised CL pretraining. Returns the SAME forecasting model instance.

    The projection head exists only inside this function and is discarded afterwards.
    Best model selection is based on validation contrastive loss.
    """
    model.to(device)
    wrapper = ContrastiveWrapper(
        model, hidden_dim=projection_hidden_dim, output_dim=projection_dim
    ).to(device)
    optimizer = optim.Adam(wrapper.parameters(), lr=lr)

    best_val = float("inf")
    best_backbone_state = copy.deepcopy(model.state_dict())
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            wrapper, train_loader, optimizer, device, temperature,
            jitter_sigma, mask_ratio, train=True
        )
        val_loss = _run_epoch(
            wrapper, val_loader, optimizer=None, device=device, temperature=temperature,
            jitter_sigma=jitter_sigma, mask_ratio=mask_ratio, train=False
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"    CL epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_backbone_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    CL early stopping at epoch {epoch}; best val={best_val:.4f}")
                break

    model.load_state_dict(best_backbone_state)
    return model, history
