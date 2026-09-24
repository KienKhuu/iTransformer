"""Small TS2Vec-inspired timestamp encoder and hierarchical contrastive loss.

This is an adaptation, not an exact reproduction of the TS2Vec paper.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # Left padding only: no representation at t uses observations after t.
        residual = x
        x = self.conv(F.pad(x, (self.pad, 0)))
        return residual + F.gelu(self.norm(x.transpose(1, 2)).transpose(1, 2))


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input = nn.Conv1d(input_dim, latent_dim, kernel_size=1)
        self.blocks = nn.Sequential(*(CausalBlock(latent_dim, d) for d in (1, 2, 4)))

    def forward(self, x):
        z = self.blocks(self.input(x.transpose(1, 2)))
        return z.transpose(1, 2)


def overlapping_views(x, crop_len=None, mask_prob=0.1):
    """Same temporal indices align within the shared overlap of two crops."""
    batch, length, _ = x.shape
    crop_len = crop_len or max(4, int(length * 0.75))
    if crop_len > length or 2 * crop_len <= length:
        raise ValueError("Crop must exceed half the input length")
    a = torch.randint(length - crop_len + 1, (1,)).item()
    b = torch.randint(length - crop_len + 1, (1,)).item()
    left, right = max(a, b), min(a + crop_len, b + crop_len)
    v1 = x[:, a:a + crop_len].clone()
    v2 = x[:, b:b + crop_len].clone()
    for v in (v1, v2):
        mask = torch.rand(batch, crop_len, device=x.device) < mask_prob
        v[mask] = 0.0  # zero is the train-fitted feature mean after scaling
    return v1, v2, slice(left - a, right - a), slice(left - b, right - b)


def contrastive_loss(z1, z2, temperature=0.2):
    """Cross-view InfoNCE: same timestamp and series are positive.

    Temporal negatives are other times in that series. Instance negatives are
    other samples at the aligned time. Both directions are averaged.
    """
    b, t, _ = z1.shape
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    loss = z1.new_zeros(())
    terms = 0
    if t > 1:
        scores = torch.bmm(z1, z2.transpose(1, 2)) / temperature
        labels = torch.arange(t, device=z1.device).repeat(b)
        loss = loss + (F.cross_entropy(scores.reshape(b * t, t), labels) +
                       F.cross_entropy(scores.transpose(1, 2).reshape(b * t, t), labels)) / 2
        terms += 1
    if b > 1:
        scores = torch.bmm(z1.transpose(0, 1), z2.transpose(0, 1).transpose(1, 2)) / temperature
        labels = torch.arange(b, device=z1.device).repeat(t)
        loss = loss + (F.cross_entropy(scores.reshape(t * b, b), labels) +
                       F.cross_entropy(scores.transpose(1, 2).reshape(t * b, b), labels)) / 2
        terms += 1
    if not terms:
        raise ValueError("At least two timestamps or two instances are required")
    return loss / terms


def hierarchical_loss(z1, z2, levels=4):
    losses = []
    for level in range(levels):
        losses.append(contrastive_loss(z1, z2))
        if level + 1 == levels or z1.shape[1] < 2:
            break
        z1 = F.max_pool1d(z1.transpose(1, 2), 2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), 2).transpose(1, 2)
    return torch.stack(losses).mean()


def pretrain_encoder(train_x, input_dim, seed, device, latent_dim=32,
                     epochs=30, batch_size=32, lr=1e-3, encoder=None):
    # Reseeding outside this function makes the comparison reproducible.
    encoder = (encoder if encoder is not None else TemporalEncoder(input_dim, latent_dim)).to(device)
    projector = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                              nn.Linear(latent_dim, latent_dim)).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr)
    loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True,
                        drop_last=len(train_x) % batch_size == 1)
    for epoch in range(epochs):
        encoder.train()
        projector.train()
        total = 0.0
        for (x,) in loader:
            x = x.to(device)
            a, b, ia, ib = overlapping_views(x)
            z1 = projector(encoder(a)[:, ia])
            z2 = projector(encoder(b)[:, ib])
            loss = hierarchical_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  CL seed={seed} epoch={epoch + 1}/{epochs} loss={total / len(loader):.4f}")
    encoder.eval()
    return copy.deepcopy(encoder).cpu()
