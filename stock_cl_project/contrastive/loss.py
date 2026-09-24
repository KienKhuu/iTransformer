import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """Symmetric NT-Xent / InfoNCE loss using all other samples as in-batch negatives.

    Args:
        z1, z2: [B, D] representations of two views of the same B windows.
        temperature: softmax temperature > 0.
    """
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError(f"Expected z1/z2 with same [B,D] shape, got {z1.shape} and {z2.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if z1.shape[0] < 2:
        raise ValueError("NT-Xent requires batch size >= 2 to provide negatives.")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    batch_size = z1.shape[0]

    logits = torch.matmul(z, z.T) / temperature  # cosine similarity because z is normalized
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    # Positive for i in first half is i+B; positive for second half is i-B.
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)

    return F.cross_entropy(logits, targets)
