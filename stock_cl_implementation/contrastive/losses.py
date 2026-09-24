import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=64):
        super().__init__()
        hidden_dim = min(256, max(128, input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature=0.2):
    """Symmetric SimCLR/InfoNCE loss over the 2B representations in a batch."""
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1 and z2 must have shape [batch, embedding_dim].")
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape.")

    batch_size = z1.size(0)
    if batch_size < 2:
        raise ValueError("Contrastive learning needs at least 2 samples per batch.")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    logits = torch.matmul(z, z.T) / temperature
    diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diagonal, float("-inf"))

    # Positive of i is i+B, and positive of i+B is i.
    targets = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (
        2 * batch_size
    )
    return F.cross_entropy(logits, targets)
